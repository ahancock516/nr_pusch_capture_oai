/*
 * PUSCH IQ + Channel-Estimate Dataset Capture Plugin for Sionna Research Kit
 *
 * Hooks into the OAI receiver plugin interface to capture, per accepted slot:
 *   - Frequency-domain IQ samples (per OFDM symbol, per allocated subcarrier)
 *   - gNB's own UL channel estimate for the same REs (v5+; antenna 0 / layer
 *     0 only, zero-filled on non-DMRS symbols — see pusch_capture_header_t)
 *   - Slot metadata needed to evaluate DMRS symbol placement and comb layout
 *
 * The plugin collects exactly N accepted slot captures (configured via config
 * file) and writes them to a single binary dataset file for offline analysis.
 * A capture is accepted only when the allocation window contains DMRS symbols,
 * the active PUSCH configuration exposes a strongly visible DMRS RE comb and
 * that comb is detected in the received IQ, and the resulting IQ payload is
 * not a duplicate of a previously accepted capture. Duplicate detection is
 * based on the raw IQ payload only, not the derived channel estimate.
 *
 * Enable with:  --loader.receiver.shlibversion _pusch_capture
 */

#define _GNU_SOURCE
#include "openair1/PHY/TOOLS/tools_defs.h"
#include "openair1/PHY/defs_gNB.h"
#include "PHY/sse_intrin.h"
#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdatomic.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <signal.h>
#include <unistd.h>

/* --------------------------------------------------------------------------
 * Configuration
 * -------------------------------------------------------------------------- */

#define DEFAULT_MAX_CAPTURES    100
#define CONFIG_FILE             "plugins/nr_pusch_capture/capture_config.txt"
#define OUTPUT_FILE             "plugins/nr_pusch_capture/data/pusch_dataset.bin"
#define LABEL_SOCKET_PATH       "/tmp/pusch_label.sock"
#define IMSI_MAX_LEN            16

/* Upper bounds for static buffers (NR maximums) */
#define MAX_SYMBOLS_PER_SLOT    14
#define MAX_RB_SIZE             273
#define MAX_RE_PER_SYM          (MAX_RB_SIZE * 12)

#define FNV1A64_OFFSET_BASIS    UINT64_C(14695981039346656037)
#define FNV1A64_PRIME           UINT64_C(1099511628211)

/* --------------------------------------------------------------------------
 * Binary file format structures  (all little-endian, packed)
 * -------------------------------------------------------------------------- */

#define PUSCH_FILE_MAGIC        0x50555343
#define PUSCH_FORMAT_VERSION    5
#define FILE_HEADER_BYTES       64
#define CAPTURE_HEADER_BYTES    148
#define DMRS_COMB_POWER_RATIO_THRESHOLD 1.50

typedef struct __attribute__((packed)) {
    uint32_t magic;
    uint32_t version;
    uint32_t max_captures;
    uint32_t num_captures;
    uint8_t  reserved[48];
} pusch_file_header_t;

_Static_assert(sizeof(pusch_file_header_t) == FILE_HEADER_BYTES,
               "file header must be 64 bytes");

typedef struct __attribute__((packed)) {
    uint32_t record_bytes;
    uint32_t capture_idx;
    int64_t  frame;
    int64_t  timestamp_ns;
    int32_t  slot;
    uint16_t rnti;
    uint8_t  qam_mod_order;
    uint8_t  num_layers;
    int32_t  start_symbol;
    int32_t  num_symbols;
    int32_t  rb_size;
    int32_t  rb_start;
    int32_t  bwp_start;
    uint32_t ul_dmrs_symb_pos;
    int32_t  scid;
    int32_t  ul_dmrs_scrambling_id;
    int32_t  data_scrambling_id;
    uint8_t  transform_precoding;
    uint8_t  dmrs_config_type;
    uint8_t  num_dmrs_cdm_grps_no_data;
    uint8_t  reserved0;
    uint16_t dmrs_ports;
    uint16_t reserved1;
    int32_t  ofdm_symbol_size;
    int32_t  first_carrier_offset;
    int32_t  nb_re_per_sym;
    int32_t  output_shift;
    uint32_t nvar;
    int16_t  valid_re[MAX_SYMBOLS_PER_SLOT];
    int32_t  iq_bytes;
    char     imsi[IMSI_MAX_LEN];
    /* v5: gNB's own LS/interpolated UL channel estimate (ul_ch_estimates),
     * antenna 0 / layer 0 only (matches the existing rxFs[0] IQ capture —
     * this gNB RU is 1T1R). Same [num_symbols][nb_re_per_sym] shape as the
     * IQ block; zero-filled on OFDM symbols without a DMRS (the estimate is
     * only ever computed at DMRS symbol positions, never interpolated
     * across the whole capture window). */
    int32_t  chest_bytes;
} pusch_capture_header_t;

_Static_assert(sizeof(pusch_capture_header_t) == CAPTURE_HEADER_BYTES,
               "capture header must be 148 bytes");

/* --------------------------------------------------------------------------
 * Plugin state
 * -------------------------------------------------------------------------- */

static FILE            *g_outfile;
static pthread_mutex_t  g_lock = PTHREAD_MUTEX_INITIALIZER;
static uint32_t         g_max_captures = DEFAULT_MAX_CAPTURES;
static atomic_uint      g_capture_count = 0;   /* accepted (controls g_done) */
static uint32_t         g_written_count  = 0;  /* written to disk (file header) */
static int              g_done = 0;
static uint64_t        *g_seen_hashes;
static uint32_t         g_seen_hash_count = 0;
static uint32_t         g_duplicate_skip_count = 0;
static uint32_t         g_dmrs_skip_count = 0;
static uint32_t         g_dmrs_comb_skip_count = 0;
static int16_t          g_iq_buf[MAX_SYMBOLS_PER_SLOT * MAX_RE_PER_SYM * 2];
static int16_t          g_chest_buf[MAX_SYMBOLS_PER_SLOT * MAX_RE_PER_SYM * 2];

/* Forward declarations for helpers defined later in this file */
static void update_capture_count(uint32_t count);
static void label_get(uint16_t rnti, char *out);

/* --------------------------------------------------------------------------
 * Pending queue — accepted captures whose IMSI is not yet known.
 * Flushed to disk once the label thread resolves the RNTI→IMSI mapping.
 * All access under g_lock.
 * -------------------------------------------------------------------------- */

typedef struct {
    uint8_t  *buf;    /* allocated record buffer; NULL = slot free */
    size_t    size;
    uint16_t  rnti;
} pending_capture_t;

static pending_capture_t *g_pending;   /* array of g_max_captures entries */
static uint32_t           g_pending_len = 0;

/* Write one pending capture with its IMSI filled in and update the file header. */
static void pending_write_one(pending_capture_t *p, const char *imsi)
{
    pusch_capture_header_t *hdr = (pusch_capture_header_t *)p->buf;
    memcpy(hdr->imsi, imsi, IMSI_MAX_LEN - 1);
    hdr->imsi[IMSI_MAX_LEN - 1] = '\0';
    fwrite(p->buf, 1, p->size, g_outfile);
    fflush(g_outfile);
    free(p->buf);
    p->buf = NULL;
    update_capture_count(++g_written_count);
}

/* Flush all pending captures for a given RNTI now that its IMSI is known.
 * Called from the label thread — must NOT hold g_label_lock when called. */
static void pending_flush_rnti(uint16_t rnti, const char *imsi)
{
    pthread_mutex_lock(&g_lock);
    uint32_t flushed = 0;
    for (uint32_t i = 0; i < g_pending_len; i++) {
        if (g_pending[i].buf && g_pending[i].rnti == rnti) {
            pending_write_one(&g_pending[i], imsi);
            flushed++;
        }
    }
    /* Compact: remove NULL slots from the front */
    uint32_t dst = 0;
    for (uint32_t i = 0; i < g_pending_len; i++) {
        if (g_pending[i].buf)
            g_pending[dst++] = g_pending[i];
    }
    g_pending_len = dst;
    pthread_mutex_unlock(&g_lock);

    if (flushed)
        printf("[nr_pusch_capture] Flushed %u pending capture(s) for "
               "RNTI 0x%04x → IMSI %s\n", flushed, rnti, imsi);
}

/* Write all remaining pending captures at shutdown (labeled or unlabeled). */
static void pending_flush_all(void)
{
    for (uint32_t i = 0; i < g_pending_len; i++) {
        if (!g_pending[i].buf)
            continue;
        char imsi[IMSI_MAX_LEN] = {0};
        label_get(g_pending[i].rnti, imsi);  /* may be empty if never resolved */
        pending_write_one(&g_pending[i], imsi);
    }
    g_pending_len = 0;
}

/* --------------------------------------------------------------------------
 * RNTI → IMSI label table (updated asynchronously by label_thread)
 * -------------------------------------------------------------------------- */

static char            g_imsi_table[65536][IMSI_MAX_LEN];
static pthread_mutex_t g_label_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_t       g_label_thread;
static atomic_int      g_label_stop = 0;
static pid_t           g_label_monitor_pid = -1;

#define LABEL_MONITOR_SCRIPT "plugins/nr_pusch_capture/scripts/label_monitor.py"
#define LABEL_MONITOR_LOG    "/tmp/label_monitor_pusch.log"

static void start_label_monitor(void)
{
    if (access(LABEL_MONITOR_SCRIPT, R_OK) != 0) {
        printf("[nr_pusch_capture] %s not found, running without auto-labeling\n",
               LABEL_MONITOR_SCRIPT);
        fflush(stdout);
        return;
    }

    g_label_monitor_pid = fork();
    if (g_label_monitor_pid == 0) {
        /* Child: redirect output to log file */
        FILE *lf = fopen(LABEL_MONITOR_LOG, "w");
        if (lf) {
            dup2(fileno(lf), STDOUT_FILENO);
            dup2(fileno(lf), STDERR_FILENO);
            fclose(lf);
        }
        execl("/usr/bin/python3", "python3", LABEL_MONITOR_SCRIPT, NULL);
        _exit(1);
    } else if (g_label_monitor_pid < 0) {
        printf("[nr_pusch_capture] Warning: fork failed, "
               "running without auto-labeling\n");
        fflush(stdout);
        g_label_monitor_pid = -1;
        return;
    }

    printf("[nr_pusch_capture] Started label_monitor.py "
           "(PID %d, log: %s)\n", g_label_monitor_pid, LABEL_MONITOR_LOG);
    fflush(stdout);

    /* Give label_monitor time to open the socket before the label thread
     * makes its first connection attempt. */
    usleep(500000);
}

static void stop_label_monitor(void)
{
    if (g_label_monitor_pid <= 0)
        return;
    kill(g_label_monitor_pid, SIGTERM);
    waitpid(g_label_monitor_pid, NULL, 0);
    printf("[nr_pusch_capture] label_monitor.py stopped\n");
    fflush(stdout);
    g_label_monitor_pid = -1;
}

static void label_set(uint16_t rnti, const char *imsi)
{
    pthread_mutex_lock(&g_label_lock);
    memcpy(g_imsi_table[rnti], imsi, IMSI_MAX_LEN - 1);
    g_imsi_table[rnti][IMSI_MAX_LEN - 1] = '\0';
    pthread_mutex_unlock(&g_label_lock);
}


static void label_get(uint16_t rnti, char *out)
{
    pthread_mutex_lock(&g_label_lock);
    strncpy(out, g_imsi_table[rnti], IMSI_MAX_LEN - 1);
    out[IMSI_MAX_LEN - 1] = '\0';
    pthread_mutex_unlock(&g_label_lock);
}

/* Read one newline-terminated line from fd using single-byte reads — avoids
 * stdio buffering issues on socket fds.  Returns line length, 0 on EOF, -1 on error. */
static ssize_t read_line(int fd, char *buf, size_t maxlen)
{
    size_t n = 0;
    while (n < maxlen - 1) {
        char c;
        ssize_t rc = read(fd, &c, 1);
        if (rc <= 0)
            return rc;
        buf[n++] = c;
        if (c == '\n')
            break;
    }
    buf[n] = '\0';
    return (ssize_t)n;
}

static void *label_thread_fn(void *arg)
{
    (void)arg;
    while (!atomic_load(&g_label_stop)) {
        int fd = socket(AF_UNIX, SOCK_STREAM, 0);
        if (fd < 0) { sleep(1); continue; }

        struct sockaddr_un addr;
        memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, LABEL_SOCKET_PATH, sizeof(addr.sun_path) - 1);

        if (connect(fd, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
            close(fd);
            sleep(1);
            continue;
        }

        printf("[nr_pusch_capture] Connected to label monitor at %s\n",
               LABEL_SOCKET_PATH);
        fflush(stdout);

        char line[256];
        while (!atomic_load(&g_label_stop) && read_line(fd, line, sizeof(line)) > 0) {
            line[strcspn(line, "\r\n")] = '\0';
            if (line[0] == '\0' || line[0] == 'S')
                continue;

            unsigned int rnti_val;
            char imsi[IMSI_MAX_LEN];

            if (line[0] == 'A'
                && sscanf(line + 1, " %x %15s", &rnti_val, imsi) == 2) {
                label_set((uint16_t)rnti_val, imsi);
                /* g_label_lock now released — safe to acquire g_lock for flush */
                pending_flush_rnti((uint16_t)rnti_val, imsi);
                printf("[nr_pusch_capture] Label: RNTI 0x%04x -> IMSI %s\n",
                       rnti_val, imsi);
                fflush(stdout);
            } else if (line[0] == 'R'
                       && sscanf(line + 1, " %x", &rnti_val) == 1) {
                /* NAS session ended but the MAC RNTI may still be active.
                 * Flush any pending captures using the current label, then
                 * keep the label alive so subsequent captures for this RNTI
                 * are written rather than silently accumulated as unlabeled. */
                char cur_imsi[IMSI_MAX_LEN];
                label_get((uint16_t)rnti_val, cur_imsi);
                if (cur_imsi[0] != '\0')
                    pending_flush_rnti((uint16_t)rnti_val, cur_imsi);
            }
        }

        close(fd);
        if (!atomic_load(&g_label_stop)) {
            printf("[nr_pusch_capture] Lost connection to label monitor, "
                   "reconnecting...\n");
            fflush(stdout);
            sleep(1);
        }
    }
    return NULL;
}

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
    hdr.magic = PUSCH_FILE_MAGIC;
    hdr.version = PUSCH_FORMAT_VERSION;
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

static int has_dmrs_in_capture_window(uint32_t dmrs_mask,
                                      int start_symbol,
                                      int num_symbols)
{
    for (int rel = 0; rel < num_symbols; rel++) {
        int symbol = start_symbol + rel;
        if ((dmrs_mask >> symbol) & 0x01)
            return 1;
    }
    return 0;
}

static int dmrs_type_is_supported(uint8_t dmrs_config_type)
{
    return dmrs_config_type == NFAPI_NR_DMRS_TYPE1
        || dmrs_config_type == NFAPI_NR_DMRS_TYPE2;
}

static int dmrs_type_is_type1(uint8_t dmrs_config_type)
{
    return dmrs_config_type == NFAPI_NR_DMRS_TYPE1;
}

static int get_dmrs_port_delta(uint8_t dmrs_config_type,
                               uint8_t dmrs_port,
                               uint8_t *delta_out)
{
    static const uint8_t type1_deltas[8] = {0, 0, 1, 1, 0, 0, 1, 1};
    static const uint8_t type2_deltas[12] = {0, 0, 2, 2, 4, 4, 0, 0, 2, 2, 4, 4};

    if (dmrs_type_is_type1(dmrs_config_type)) {
        if (dmrs_port >= 8)
            return 0;
        *delta_out = type1_deltas[dmrs_port];
        return 1;
    }

    if (dmrs_config_type != NFAPI_NR_DMRS_TYPE2 || dmrs_port >= 12)
        return 0;
    *delta_out = type2_deltas[dmrs_port];
    return 1;
}

static void mark_dmrs_group_bins(uint8_t dmrs_config_type,
                                 uint8_t group_index,
                                 uint8_t bins[NR_NB_SC_PER_RB])
{
    if (dmrs_type_is_type1(dmrs_config_type)) {
        for (int k = group_index; k < NR_NB_SC_PER_RB; k += 2)
            bins[k] = 1;
        return;
    }

    static const uint8_t type2_group_bins[3][4] = {
        {0, 1, 6, 7},
        {2, 3, 8, 9},
        {4, 5, 10, 11},
    };

    if (group_index >= 3)
        return;
    for (int i = 0; i < 4; i++)
        bins[type2_group_bins[group_index][i]] = 1;
}

static void mark_dmrs_active_bins(uint8_t dmrs_config_type,
                                  uint8_t delta,
                                  uint8_t bins[NR_NB_SC_PER_RB])
{
    if (dmrs_type_is_type1(dmrs_config_type)) {
        for (int k = delta; k < NR_NB_SC_PER_RB; k += 2)
            bins[k] = 1;
        return;
    }

    static const uint8_t type2_offsets[4] = {0, 1, 6, 7};
    for (int i = 0; i < 4; i++) {
        uint8_t bin = delta + type2_offsets[i];
        if (bin < NR_NB_SC_PER_RB)
            bins[bin] = 1;
    }
}

static int build_dmrs_quiet_bins(const pusch_capture_header_t *hdr,
                                 uint8_t active_bins[NR_NB_SC_PER_RB],
                                 uint8_t quiet_bins[NR_NB_SC_PER_RB])
{
    uint8_t reserved_bins[NR_NB_SC_PER_RB] = {0};
    memset(active_bins, 0, NR_NB_SC_PER_RB);
    memset(quiet_bins, 0, NR_NB_SC_PER_RB);

    if (!dmrs_type_is_supported(hdr->dmrs_config_type))
        return 0;

    int max_groups = dmrs_type_is_type1(hdr->dmrs_config_type) ? 2 : 3;
    if (hdr->num_dmrs_cdm_grps_no_data < 1
        || hdr->num_dmrs_cdm_grps_no_data > max_groups)
        return 0;

    for (uint8_t group = 0; group < hdr->num_dmrs_cdm_grps_no_data; group++)
        mark_dmrs_group_bins(hdr->dmrs_config_type, group, reserved_bins);

    if (hdr->dmrs_ports == 0)
        return 0;

    int max_ports = dmrs_type_is_type1(hdr->dmrs_config_type) ? 8 : 12;
    for (int port = 0; port < max_ports; port++) {
        if (((uint16_t)hdr->dmrs_ports & (uint16_t)(1u << port)) == 0)
            continue;
        uint8_t delta = 0;
        if (!get_dmrs_port_delta(hdr->dmrs_config_type, (uint8_t)port, &delta))
            return 0;
        mark_dmrs_active_bins(hdr->dmrs_config_type, delta, active_bins);
    }

    int active_count = 0;
    int quiet_count = 0;
    for (int bin = 0; bin < NR_NB_SC_PER_RB; bin++) {
        if (active_bins[bin]) {
            if (!reserved_bins[bin])
                return 0;
            active_count++;
        } else if (reserved_bins[bin]) {
            quiet_bins[bin] = 1;
            quiet_count++;
        }
    }

    /* quiet_count==0 is valid when num_dmrs_cdm_grps_no_data==1 (single CDM
     * group, all reserved subcarriers carry active DMRS — e.g. SISO PUSCH).
     * In that case the comb power check is skipped; we still require at least
     * one active DMRS bin to flag a valid layout. */
    return active_count > 0;
}

static int has_expected_dmrs_comb(const pusch_capture_header_t *hdr,
                                  const int16_t *iq_buf,
                                  const uint8_t active_bins[NR_NB_SC_PER_RB],
                                  const uint8_t quiet_bins[NR_NB_SC_PER_RB])
{
    double active_power_sum = 0.0;
    double quiet_power_sum = 0.0;
    uint32_t active_sample_count = 0;
    uint32_t quiet_sample_count = 0;

    for (int rel_symbol = 0; rel_symbol < hdr->num_symbols; rel_symbol++) {
        int symbol = hdr->start_symbol + rel_symbol;
        if (((hdr->ul_dmrs_symb_pos >> symbol) & 0x01) == 0)
            continue;

        size_t symbol_offset = (size_t)rel_symbol * (size_t)hdr->nb_re_per_sym;
        for (int re = 0; re < hdr->nb_re_per_sym; re++) {
            size_t sample_index = (symbol_offset + (size_t)re) * 2;
            double i = (double)iq_buf[sample_index + 0];
            double q = (double)iq_buf[sample_index + 1];
            double power = i * i + q * q;
            uint8_t bin = (uint8_t)(re % NR_NB_SC_PER_RB);

            if (active_bins[bin]) {
                active_power_sum += power;
                active_sample_count++;
            } else if (quiet_bins[bin]) {
                quiet_power_sum += power;
                quiet_sample_count++;
            }
        }
    }

    if (active_sample_count == 0)
        return 0;

    double active_mean = active_power_sum / (double)active_sample_count;

    if (quiet_sample_count == 0)
        return active_mean > 0.0;

    double quiet_mean = quiet_power_sum / (double)quiet_sample_count;
    if (quiet_mean <= 0.0)
        return active_mean > 0.0;

    return active_mean >= quiet_mean * DMRS_COMB_POWER_RATIO_THRESHOLD;
}

static uint64_t fnv1a64_update(uint64_t hash, const void *data, size_t len)
{
    const uint8_t *bytes = (const uint8_t *)data;
    for (size_t i = 0; i < len; i++) {
        hash ^= bytes[i];
        hash *= FNV1A64_PRIME;
    }
    return hash;
}

static uint64_t compute_capture_identity_hash(const pusch_capture_header_t *hdr,
                                              const uint8_t *payload,
                                              size_t payload_bytes)
{
    pusch_capture_header_t hash_hdr = *hdr;
    hash_hdr.capture_idx = 0;
    hash_hdr.frame = 0;
    hash_hdr.timestamp_ns = 0;
    hash_hdr.slot = 0;

    uint64_t hash = FNV1A64_OFFSET_BASIS;
    hash = fnv1a64_update(hash, &hash_hdr, sizeof(hash_hdr));
    hash = fnv1a64_update(hash, payload, payload_bytes);
    return hash;
}

static int capture_hash_seen(uint64_t hash)
{
    for (uint32_t i = 0; i < g_seen_hash_count; i++) {
        if (g_seen_hashes[i] == hash)
            return 1;
    }
    return 0;
}

static void remember_capture_hash(uint64_t hash)
{
    AssertFatal(g_seen_hash_count < g_max_captures,
                "[nr_pusch_capture] hash table overflow (%u / %u)\n",
                g_seen_hash_count, g_max_captures);
    g_seen_hashes[g_seen_hash_count++] = hash;
}

static void maybe_log_skip(const char *reason,
                           uint32_t skip_count,
                           int64_t frame,
                           int slot)
{
    if (skip_count <= 5 || skip_count % 100 == 0) {
        printf("[nr_pusch_capture] Skipping %s (count=%u, frame=%ld, slot=%d)\n",
               reason, skip_count, (long)frame, slot);
        fflush(stdout);
    }
}

/* --------------------------------------------------------------------------
 * Plugin lifecycle
 * -------------------------------------------------------------------------- */

int32_t receiver_init(void)
{
    g_max_captures = read_max_captures(CONFIG_FILE);
    g_seen_hash_count = 0;
    g_duplicate_skip_count = 0;
    g_dmrs_skip_count = 0;
    g_dmrs_comb_skip_count = 0;
    g_done = 0;
    atomic_store(&g_capture_count, 0);

    printf("[nr_pusch_capture] Initializing - will capture %u accepted slots\n",
           g_max_captures);
    printf("[nr_pusch_capture] Output: %s\n", OUTPUT_FILE);
    fflush(stdout);

    pthread_mutex_init(&g_lock, NULL);
    pthread_mutex_init(&g_label_lock, NULL);
    memset(g_imsi_table, 0, sizeof(g_imsi_table));
    atomic_store(&g_label_stop, 0);

    g_pending = calloc(g_max_captures, sizeof(*g_pending));
    AssertFatal(g_pending != NULL,
                "[nr_pusch_capture] Cannot allocate pending queue\n");
    g_pending_len   = 0;
    g_written_count = 0;

    start_label_monitor();
    pthread_create(&g_label_thread, NULL, label_thread_fn, NULL);

    g_seen_hashes = calloc(g_max_captures, sizeof(*g_seen_hashes));
    AssertFatal(g_seen_hashes != NULL,
                "[nr_pusch_capture] Cannot allocate duplicate filter state\n");

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
    printf("[nr_pusch_capture] Shutting down - captured %u / %u accepted slots\n",
           final_count, g_max_captures);
    printf("[nr_pusch_capture] Skipped %u captures without DMRS symbols, %u without a strongly visible DMRS comb, and %u duplicate captures\n",
           g_dmrs_skip_count, g_dmrs_comb_skip_count, g_duplicate_skip_count);
    fflush(stdout);

    atomic_store(&g_label_stop, 1);
    pthread_join(g_label_thread, NULL);
    pthread_mutex_destroy(&g_label_lock);
    stop_label_monitor();

    /* Flush any captures still waiting for an IMSI */
    if (g_pending_len > 0) {
        printf("[nr_pusch_capture] Flushing %u pending capture(s) at shutdown\n",
               g_pending_len);
        fflush(stdout);
        pending_flush_all();
    }
    free(g_pending);
    g_pending = NULL;

    if (g_outfile) {
        update_capture_count(g_written_count);
        fclose(g_outfile);
        g_outfile = NULL;
    }
    free(g_seen_hashes);
    g_seen_hashes = NULL;
    pthread_mutex_destroy(&g_lock);
    return 0;
}

/* --------------------------------------------------------------------------
 * Receiver plugin entry point
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
    (void)gNB;
    (void)ulsch_id;
    (void)llr;
    (void)lengths;

    if (g_done || !ul_chs)
        return 0;

    int nb_re_per_sym = NR_NB_SC_PER_RB * rel15_ul->rb_size;
    int start_re = (frame_parms->first_carrier_offset
                    + (rel15_ul->rb_start + rel15_ul->bwp_start) * NR_NB_SC_PER_RB)
                   % frame_parms->ofdm_symbol_size;
    int re_wrap = frame_parms->ofdm_symbol_size;

    pthread_mutex_lock(&g_lock);
    if (g_done) {
        pthread_mutex_unlock(&g_lock);
        return 0;
    }

    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);

    pusch_capture_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    hdr.capture_idx = atomic_load(&g_capture_count);
    hdr.frame = (int64_t)frame;
    hdr.timestamp_ns = (int64_t)ts.tv_sec * 1000000000LL + (int64_t)ts.tv_nsec;
    hdr.slot = slot;
    hdr.rnti = rel15_ul->rnti;
    hdr.qam_mod_order = rel15_ul->qam_mod_order;
    hdr.num_layers = rel15_ul->nrOfLayers;
    hdr.start_symbol = start_symbol;
    hdr.num_symbols = num_symbols;
    hdr.rb_size = rel15_ul->rb_size;
    hdr.rb_start = rel15_ul->rb_start;
    hdr.bwp_start = rel15_ul->bwp_start;
    hdr.ul_dmrs_symb_pos = rel15_ul->ul_dmrs_symb_pos;
    hdr.scid = rel15_ul->scid;
    hdr.ul_dmrs_scrambling_id = rel15_ul->ul_dmrs_scrambling_id;
    hdr.data_scrambling_id = rel15_ul->data_scrambling_id;
    hdr.transform_precoding = rel15_ul->transform_precoding;
    hdr.dmrs_config_type = rel15_ul->dmrs_config_type;
    hdr.num_dmrs_cdm_grps_no_data = rel15_ul->num_dmrs_cdm_grps_no_data;
    hdr.dmrs_ports = rel15_ul->dmrs_ports;
    hdr.ofdm_symbol_size = frame_parms->ofdm_symbol_size;
    hdr.first_carrier_offset = frame_parms->first_carrier_offset;
    hdr.nb_re_per_sym = nb_re_per_sym;
    hdr.output_shift = output_shift;
    hdr.nvar = nvar;

    if (!has_dmrs_in_capture_window(hdr.ul_dmrs_symb_pos, start_symbol, num_symbols)) {
        g_dmrs_skip_count++;
        maybe_log_skip("capture without DMRS in allocation window",
                       g_dmrs_skip_count,
                       hdr.frame,
                       hdr.slot);
        pthread_mutex_unlock(&g_lock);
        return 0;
    }

    for (int s = 0; s < MAX_SYMBOLS_PER_SLOT; s++) {
        hdr.valid_re[s] = (s < num_symbols)
            ? pusch_vars->ul_valid_re_per_slot[start_symbol + s] : 0;
    }

    int iq_off = 0;
    for (int s = 0; s < num_symbols; s++) {
        int symbol = start_symbol + s;
        c16_t *rxF = (c16_t *)rxFs[0]
                     + symbol * frame_parms->ofdm_symbol_size + soffset;

        for (int i = 0, k = start_re; i < nb_re_per_sym;
             k = (k + 1 < re_wrap ? k + 1 : 0), ++i) {
            g_iq_buf[iq_off * 2 + 0] = rxF[k].r;
            g_iq_buf[iq_off * 2 + 1] = rxF[k].i;
            iq_off++;
        }
    }

    uint8_t dmrs_active_bins[NR_NB_SC_PER_RB];
    uint8_t dmrs_quiet_bins[NR_NB_SC_PER_RB];
    if (!build_dmrs_quiet_bins(&hdr, dmrs_active_bins, dmrs_quiet_bins)) {
        g_dmrs_comb_skip_count++;
        maybe_log_skip("capture without a supportable visible DMRS RE comb",
                       g_dmrs_comb_skip_count,
                       hdr.frame,
                       hdr.slot);
        pthread_mutex_unlock(&g_lock);
        return 0;
    }

    if (!has_expected_dmrs_comb(&hdr, g_iq_buf, dmrs_active_bins, dmrs_quiet_bins)) {
        g_dmrs_comb_skip_count++;
        maybe_log_skip("capture without a strongly visible DMRS RE comb",
                       g_dmrs_comb_skip_count,
                       hdr.frame,
                       hdr.slot);
        pthread_mutex_unlock(&g_lock);
        return 0;
    }

    hdr.iq_bytes = iq_off * 2 * (int32_t)sizeof(int16_t);

    /* gNB's own channel estimate for the same REs, antenna 0 / layer 0.
     * ul_chs[0] is the already-computed LS/interpolated estimate (see
     * nr_pusch_antenna_processing() in nr_ul_channel_estimation.c) — a
     * dense, 0-based array of nb_re_per_sym values per OFDM symbol,
     * starting at offset (ofdm_symbol_size * symbol). No wraparound needed
     * here (unlike the raw IQ read above), and only DMRS symbols are
     * populated — zero elsewhere, matching the underlying OAI buffer. */
    memset(g_chest_buf, 0, (size_t)num_symbols * nb_re_per_sym * 2 * sizeof(int16_t));
    c16_t *ul_ch0 = ul_chs[0];
    for (int s = 0; s < num_symbols; s++) {
        int symbol = start_symbol + s;
        if (((hdr.ul_dmrs_symb_pos >> symbol) & 0x01) == 0)
            continue;   /* no DMRS on this symbol — leave zero-filled */

        c16_t *ch = ul_ch0 + (size_t)symbol * frame_parms->ofdm_symbol_size;
        for (int re = 0; re < nb_re_per_sym; re++) {
            g_chest_buf[(s * nb_re_per_sym + re) * 2 + 0] = ch[re].r;
            g_chest_buf[(s * nb_re_per_sym + re) * 2 + 1] = ch[re].i;
        }
    }
    hdr.chest_bytes = num_symbols * nb_re_per_sym * 2 * (int32_t)sizeof(int16_t);

    hdr.record_bytes = (uint32_t)(CAPTURE_HEADER_BYTES + hdr.iq_bytes + hdr.chest_bytes);
    label_get(rel15_ul->rnti, hdr.imsi);

    size_t record_bytes = hdr.record_bytes;
    uint8_t *record_buf = malloc(record_bytes);
    AssertFatal(record_buf != NULL,
                "[nr_pusch_capture] Cannot allocate record buffer (%zu bytes)\n",
                record_bytes);

    memcpy(record_buf, &hdr, CAPTURE_HEADER_BYTES);
    memcpy(record_buf + CAPTURE_HEADER_BYTES, g_iq_buf, hdr.iq_bytes);
    memcpy(record_buf + CAPTURE_HEADER_BYTES + hdr.iq_bytes, g_chest_buf, hdr.chest_bytes);

    uint64_t hash = compute_capture_identity_hash(
        &hdr,
        record_buf + CAPTURE_HEADER_BYTES,
        (size_t)hdr.iq_bytes);
    if (capture_hash_seen(hash)) {
        free(record_buf);
        g_duplicate_skip_count++;
        maybe_log_skip("duplicate capture payload",
                       g_duplicate_skip_count,
                       hdr.frame,
                       hdr.slot);
        pthread_mutex_unlock(&g_lock);
        return 0;
    }

    remember_capture_hash(hash);

    /* Accept the capture — count it regardless of whether IMSI is known yet */
    uint32_t idx = atomic_fetch_add(&g_capture_count, 1) + 1;

    if (hdr.imsi[0] != '\0') {
        /* IMSI known — write immediately */
        fwrite(record_buf, 1, record_bytes, g_outfile);
        fflush(g_outfile);
        free(record_buf);
        update_capture_count(++g_written_count);
    } else {
        /* IMSI unknown — queue until label thread resolves it */
        AssertFatal(g_pending_len < g_max_captures,
                    "[nr_pusch_capture] Pending queue overflow\n");
        g_pending[g_pending_len].buf  = record_buf;
        g_pending[g_pending_len].size = record_bytes;
        g_pending[g_pending_len].rnti = rel15_ul->rnti;
        g_pending_len++;
    }

    if (idx % 50 == 0 || idx == g_max_captures) {
        printf("[nr_pusch_capture] Captured %u / %u accepted slots "
               "(%u written, %u pending)\n",
               idx, g_max_captures, g_written_count, g_pending_len);
        fflush(stdout);
    }

    if (idx >= g_max_captures) {
        g_done = 1;
        printf("[nr_pusch_capture] Dataset complete (%u accepted). "
               "Plugin now passthrough.\n", idx);
        fflush(stdout);
    }

    pthread_mutex_unlock(&g_lock);
    return 0;
}

int receiver_symbols_requested(NR_DL_FRAME_PARMS *frame_parms)
{
    (void)frame_parms;
    return -1;
}
