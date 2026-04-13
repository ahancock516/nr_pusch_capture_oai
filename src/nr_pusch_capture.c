/*
 * PUSCH IQ Dataset Capture Plugin for Sionna Research Kit
 *
 * Hooks into the OAI receiver plugin interface to capture raw PUSCH data:
 *   - Frequency-domain IQ samples (per antenna, per OFDM symbol)
 *   - Channel estimates from DMRS interpolation
 *   - Unscrambled LLR output from the default receiver
 *   - Full slot metadata (RNTI, MCS, DMRS config, PRB allocation, ...)
 *
 * The plugin collects exactly N slot captures (configurable via config file)
 * and writes them to a single binary dataset file for offline analysis.
 *
 * Enable with:  --loader.receiver.shlibversion _pusch_capture
 *
 * Binary format: see pusch_capture_format.h or the companion read_dataset.py
 */

#define _GNU_SOURCE
#include "openair1/PHY/TOOLS/tools_defs.h"
#include "openair1/PHY/defs_gNB.h"
#include "openair1/PHY/NR_REFSIG/dmrs_nr.h"
#include "PHY/sse_intrin.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <time.h>
#include <stdatomic.h>

/* --------------------------------------------------------------------------
 * Configuration
 * -------------------------------------------------------------------------- */

#define DEFAULT_MAX_CAPTURES    100
#define CONFIG_FILE             "plugins/nr_pusch_capture/capture_config.txt"
#define OUTPUT_FILE             "plugins/nr_pusch_capture/data/pusch_dataset.bin"

/* Upper bounds for static buffers (NR maximums) */
#define MAX_SYMBOLS_PER_SLOT    14
#define MAX_RB_SIZE             273
#define MAX_RE_PER_SYM          (MAX_RB_SIZE * 12)   /* 3276 */
#define MAX_MOD_ORDER           8                     /* 256-QAM */

/* --------------------------------------------------------------------------
 * Binary file format structures  (all little-endian, packed)
 * -------------------------------------------------------------------------- */

#define PUSCH_FILE_MAGIC        0x50555343  /* "PUSC" */
#define PUSCH_FORMAT_VERSION    1
#define FILE_HEADER_BYTES       64
#define CAPTURE_HEADER_BYTES    128

typedef struct __attribute__((packed)) {
    uint32_t magic;                 /*  0: 0x50555343                       */
    uint32_t version;               /*  4: format version                   */
    uint32_t max_captures;          /*  8: configured N                     */
    uint32_t num_captures;          /* 12: captures written so far          */
    uint8_t  reserved[48];          /* 16-63: pad to 64 bytes               */
} pusch_file_header_t;

_Static_assert(sizeof(pusch_file_header_t) == FILE_HEADER_BYTES,
               "file header must be 64 bytes");

typedef struct __attribute__((packed)) {
    /* ---- record envelope ---- */
    uint32_t record_bytes;          /*  0: total bytes of this record       */
    uint32_t capture_idx;           /*  4: 0-based capture index            */

    /* ---- 8-byte aligned fields ---- */
    int64_t  frame;                 /*  8: radio frame number               */
    int64_t  timestamp_ns;          /* 16: CLOCK_MONOTONIC nanoseconds      */

    /* ---- slot / UE identification ---- */
    int32_t  slot;                  /* 24                                   */
    uint16_t rnti;                  /* 28                                   */
    uint8_t  qam_mod_order;         /* 30: 2=QPSK, 4=16QAM, 6=64QAM, 8=256QAM */
    uint8_t  num_layers;            /* 31                                   */

    /* ---- PUSCH resource allocation ---- */
    int32_t  start_symbol;          /* 32                                   */
    int32_t  num_symbols;           /* 36                                   */
    int32_t  rb_size;               /* 40: number of allocated PRBs         */
    int32_t  rb_start;              /* 44                                   */
    int32_t  bwp_start;             /* 48                                   */

    /* ---- DMRS configuration ---- */
    uint32_t ul_dmrs_symb_pos;      /* 52: DMRS symbol bitmask              */
    int32_t  scid;                  /* 56                                   */
    int32_t  ul_dmrs_scrambling_id; /* 60                                   */
    int32_t  data_scrambling_id;    /* 64                                   */

    /* ---- PHY parameters ---- */
    int32_t  ofdm_symbol_size;      /* 68: FFT size                         */
    int32_t  first_carrier_offset;  /* 72                                   */
    int32_t  nb_re_per_sym;         /* 76: NR_NB_SC_PER_RB * rb_size        */
    int32_t  output_shift;          /* 80: log2_maxh                        */
    uint32_t nvar;                  /* 84: noise variance estimate          */

    /* ---- per-symbol valid RE counts ---- */
    int16_t  valid_re[MAX_SYMBOLS_PER_SLOT]; /* 88: ul_valid_re_per_slot     */
                                    /* 88 + 28 = 116                        */

    /* ---- payload sizes in bytes ---- */
    int32_t  iq_bytes;              /* 116: total IQ payload bytes          */
    int32_t  chest_bytes;           /* 120: total channel-est payload bytes */
    int32_t  llr_bytes;             /* 124: total LLR payload bytes         */
} pusch_capture_header_t;           /* 128 total                            */

_Static_assert(sizeof(pusch_capture_header_t) == CAPTURE_HEADER_BYTES,
               "capture header must be 128 bytes");

/* --------------------------------------------------------------------------
 * Plugin state
 * -------------------------------------------------------------------------- */

static FILE          *g_outfile;
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;
static uint32_t       g_max_captures = DEFAULT_MAX_CAPTURES;
static atomic_uint    g_capture_count = 0;
static int            g_done = 0;  /* set once N captures reached */

/* Buffers for the two-phase capture (input IQ/chest → output LLR) */
static pusch_capture_header_t g_pending_hdr;
static int16_t g_iq_buf  [MAX_SYMBOLS_PER_SLOT * MAX_RE_PER_SYM * 2]; /* real,imag */
static int16_t g_chest_buf[MAX_SYMBOLS_PER_SLOT * MAX_RE_PER_SYM * 2];
static int     g_input_captured = 0;   /* flag: input phase done, waiting for output */

/* --------------------------------------------------------------------------
 * Helpers
 * -------------------------------------------------------------------------- */

static uint32_t read_max_captures(const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f)
        return DEFAULT_MAX_CAPTURES;
    uint32_t n = 0;
    if (fscanf(f, "%u", &n) != 1 || n == 0)
        n = DEFAULT_MAX_CAPTURES;
    fclose(f);
    return n;
}

static void write_file_header(void)
{
    pusch_file_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.magic        = PUSCH_FILE_MAGIC;
    hdr.version      = PUSCH_FORMAT_VERSION;
    hdr.max_captures = g_max_captures;
    hdr.num_captures = 0;
    fwrite(&hdr, sizeof(hdr), 1, g_outfile);
    fflush(g_outfile);
}

static void update_capture_count(uint32_t count)
{
    long pos = ftell(g_outfile);
    fseek(g_outfile, offsetof(pusch_file_header_t, num_captures), SEEK_SET);
    fwrite(&count, sizeof(count), 1, g_outfile);
    fseek(g_outfile, pos, SEEK_SET);
    fflush(g_outfile);
}

/* --------------------------------------------------------------------------
 * Plugin lifecycle
 * -------------------------------------------------------------------------- */

int32_t receiver_init(void)
{
    g_max_captures = read_max_captures(CONFIG_FILE);
    printf("[nr_pusch_capture] Initializing — will capture %u slots\n",
           g_max_captures);
    printf("[nr_pusch_capture] Output: %s\n", OUTPUT_FILE);
    fflush(stdout);

    pthread_mutex_init(&g_lock, NULL);

    g_outfile = fopen(OUTPUT_FILE, "wb");
    AssertFatal(g_outfile != NULL,
                "[nr_pusch_capture] Cannot open %s for writing\n", OUTPUT_FILE);
    write_file_header();

    return 0;
}

int32_t receiver_init_thread(void)
{
    return 0;
}

int32_t receiver_shutdown(void)
{
    uint32_t final_count = atomic_load(&g_capture_count);
    printf("[nr_pusch_capture] Shutting down — captured %u / %u slots\n",
           final_count, g_max_captures);
    fflush(stdout);

    if (g_outfile) {
        update_capture_count(final_count);
        fclose(g_outfile);
        g_outfile = NULL;
    }
    pthread_mutex_destroy(&g_lock);
    return 0;
}

/* --------------------------------------------------------------------------
 * Receiver plugin entry point
 *
 * Called TWICE per PUSCH slot by nr_ulsch_demodulation.c:
 *   1) ul_chs != NULL  →  input phase  (raw IQ + channel estimates available)
 *   2) ul_chs == NULL  →  output phase (unscrambled LLRs available)
 *
 * Return values:
 *    0  → plugin did NOT handle decoding; OAI runs default inner_rx
 *   -1  → plugin captured input; requests output callback after default rx
 *    1  → plugin fully handled decoding (not used here)
 * -------------------------------------------------------------------------- */

int receiver_compute_llr(PHY_VARS_gNB *gNB,
                         int ulsch_id,
                         int slot,
                         frame_t frame,
                         NR_DL_FRAME_PARMS *frame_parms,
                         NR_gNB_PUSCH *pusch_vars,
                         nfapi_nr_pusch_pdu_t *rel15_ul,
                         c16_t **rxFs,
                         c16_t **ul_chs,
                         int16_t *llr,
                         int soffset,
                         int16_t const *lengths,
                         int start_symbol,
                         int num_symbols,
                         int output_shift,
                         uint32_t nvar)
{
    /* Fast path: done collecting */
    if (g_done)
        return 0;

    int nb_re_per_sym = NR_NB_SC_PER_RB * rel15_ul->rb_size;
    int start_re = (frame_parms->first_carrier_offset
                    + (rel15_ul->rb_start + rel15_ul->bwp_start) * NR_NB_SC_PER_RB)
                   % frame_parms->ofdm_symbol_size;
    int re_wrap = frame_parms->ofdm_symbol_size;

    /* ==================================================================
     * INPUT PHASE: capture raw IQ and channel estimates
     * ================================================================== */
    if (ul_chs) {
        pthread_mutex_lock(&g_lock);

        if (g_done) {
            pthread_mutex_unlock(&g_lock);
            return 0;
        }

        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);

        /* Fill capture header */
        memset(&g_pending_hdr, 0, sizeof(g_pending_hdr));
        g_pending_hdr.capture_idx           = atomic_load(&g_capture_count);
        g_pending_hdr.frame                 = (int64_t)frame;
        g_pending_hdr.timestamp_ns          = (int64_t)ts.tv_sec * 1000000000LL
                                            + (int64_t)ts.tv_nsec;
        g_pending_hdr.slot                  = slot;
        g_pending_hdr.rnti                  = rel15_ul->rnti;
        g_pending_hdr.qam_mod_order         = rel15_ul->qam_mod_order;
        g_pending_hdr.num_layers            = rel15_ul->nrOfLayers;
        g_pending_hdr.start_symbol          = start_symbol;
        g_pending_hdr.num_symbols           = num_symbols;
        g_pending_hdr.rb_size               = rel15_ul->rb_size;
        g_pending_hdr.rb_start              = rel15_ul->rb_start;
        g_pending_hdr.bwp_start             = rel15_ul->bwp_start;
        g_pending_hdr.ul_dmrs_symb_pos      = rel15_ul->ul_dmrs_symb_pos;
        g_pending_hdr.scid                  = rel15_ul->scid;
        g_pending_hdr.ul_dmrs_scrambling_id = rel15_ul->ul_dmrs_scrambling_id;
        g_pending_hdr.data_scrambling_id    = rel15_ul->data_scrambling_id;
        g_pending_hdr.ofdm_symbol_size      = frame_parms->ofdm_symbol_size;
        g_pending_hdr.first_carrier_offset  = frame_parms->first_carrier_offset;
        g_pending_hdr.nb_re_per_sym         = nb_re_per_sym;
        g_pending_hdr.output_shift          = output_shift;
        g_pending_hdr.nvar                  = nvar;

        /* Per-symbol valid RE counts */
        for (int s = 0; s < MAX_SYMBOLS_PER_SLOT; s++)
            g_pending_hdr.valid_re[s] = (s < num_symbols)
                ? pusch_vars->ul_valid_re_per_slot[start_symbol + s] : 0;

        /* Capture IQ and channel estimates per OFDM symbol */
        int iq_off = 0;
        for (int s = 0; s < num_symbols; s++) {
            int symbol = start_symbol + s;

            /* Resolve DMRS symbol index for channel estimate addressing */
            int dmrs_symbol = symbol;
            if (gNB->chest_time == 0)
                dmrs_symbol = (rel15_ul->ul_dmrs_symb_pos >> symbol) & 0x01
                    ? symbol
                    : get_valid_dmrs_idx_for_channel_est(
                          rel15_ul->ul_dmrs_symb_pos, symbol);
            else {
                int end_sym = rel15_ul->start_symbol_index
                            + rel15_ul->nr_of_symbols;
                dmrs_symbol = get_next_dmrs_symbol_in_slot(
                    rel15_ul->ul_dmrs_symb_pos,
                    rel15_ul->start_symbol_index, end_sym);
            }

            c16_t *rxF = (c16_t *)rxFs[0]
                         + symbol * frame_parms->ofdm_symbol_size + soffset;
            c16_t *ul_ch_est = (c16_t *)pusch_vars->ul_ch_estimates[0]
                               + dmrs_symbol * frame_parms->ofdm_symbol_size;

            /* Copy subcarriers (handle wrap-around at FFT edge) */
            for (int i = 0, k = start_re; i < nb_re_per_sym;
                 k = (k + 1 < re_wrap ? k + 1 : 0), ++i) {
                g_iq_buf[iq_off * 2 + 0]    = rxF[k].r;
                g_iq_buf[iq_off * 2 + 1]    = rxF[k].i;
                g_chest_buf[iq_off * 2 + 0]  = ul_ch_est[i].r;
                g_chest_buf[iq_off * 2 + 1]  = ul_ch_est[i].i;
                iq_off++;
            }
        }

        int total_iq_samples = iq_off;
        g_pending_hdr.iq_bytes    = total_iq_samples * 2 * (int32_t)sizeof(int16_t);
        g_pending_hdr.chest_bytes = total_iq_samples * 2 * (int32_t)sizeof(int16_t);
        g_pending_hdr.llr_bytes   = 0;  /* filled in output phase */

        g_input_captured = 1;

        /* Return -1: let OAI run default rx, then call us again for LLR output */
        return -1;
    }

    /* ==================================================================
     * OUTPUT PHASE: capture unscrambled LLRs
     * ================================================================== */
    if (!g_input_captured) {
        /* Spurious output call without a preceding input capture */
        return 0;
    }
    g_input_captured = 0;

    /* Gather LLR data per symbol */
    int total_llr_count = 0;
    for (int s = 0; s < g_pending_hdr.num_symbols; s++) {
        int symbol = g_pending_hdr.start_symbol + s;
        int nb_re_pusch = pusch_vars->ul_valid_re_per_slot[symbol];
        total_llr_count += nb_re_pusch * g_pending_hdr.qam_mod_order
                         * g_pending_hdr.num_layers;
    }
    g_pending_hdr.llr_bytes = total_llr_count * (int32_t)sizeof(int16_t);

    /* Compute total record size */
    g_pending_hdr.record_bytes = (uint32_t)(CAPTURE_HEADER_BYTES
                                  + g_pending_hdr.iq_bytes
                                  + g_pending_hdr.chest_bytes
                                  + g_pending_hdr.llr_bytes);

    /* Write the complete capture record */
    fwrite(&g_pending_hdr, CAPTURE_HEADER_BYTES, 1, g_outfile);
    fwrite(g_iq_buf,    1, g_pending_hdr.iq_bytes,    g_outfile);
    fwrite(g_chest_buf, 1, g_pending_hdr.chest_bytes, g_outfile);

    /* Write LLR data: walk per-symbol using llr_offset */
    for (int s = 0; s < g_pending_hdr.num_symbols; s++) {
        int symbol = g_pending_hdr.start_symbol + s;
        int nb_re_pusch = pusch_vars->ul_valid_re_per_slot[symbol];
        int llr_count = nb_re_pusch * g_pending_hdr.qam_mod_order
                      * g_pending_hdr.num_layers;
        int16_t *llr_ptr = &llr[pusch_vars->llr_offset[symbol]
                                * g_pending_hdr.num_layers];
        fwrite(llr_ptr, sizeof(int16_t), llr_count, g_outfile);
    }
    fflush(g_outfile);

    uint32_t idx = atomic_fetch_add(&g_capture_count, 1) + 1;
    update_capture_count(idx);

    if (idx % 50 == 0 || idx == g_max_captures) {
        printf("[nr_pusch_capture] Captured %u / %u slots\n",
               idx, g_max_captures);
        fflush(stdout);
    }

    if (idx >= g_max_captures) {
        g_done = 1;
        printf("[nr_pusch_capture] Dataset complete (%u captures). "
               "Plugin now passthrough.\n", idx);
        fflush(stdout);
    }

    pthread_mutex_unlock(&g_lock);
    return 0;
}

int receiver_symbols_requested(NR_DL_FRAME_PARMS *frame_parms)
{
    /* Return -1 to request all symbols in the slot at once */
    return -1;
}
