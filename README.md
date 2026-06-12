# NR PUSCH Capture Plugin — OAI Integration Layer

An OAI gNB receiver-side plugin that captures frequency-domain PUSCH IQ with
real-time per-device IMSI labeling. Forked from
[C2A2-at-Florida-Atlantic-University/nr_pusch_capture](https://github.com/C2A2-at-Florida-Atlantic-University/nr_pusch_capture)
and extended with an OAI-specific labeling pipeline.

## What this fork adds

| Feature | Upstream (v3) | This fork (v4) |
|---|---|---|
| Dataset format | v3 — IQ + DMRS metadata | v4 — adds `imsi[16]` field per capture |
| IMSI labeling | post-hoc via `post_label.py` | embedded at capture time via Unix socket |
| Label monitor | run separately | auto-started by plugin at init |
| Pending queue | — | holds captures until IMSI known, then flushes |
| `read_dataset.py` | v1/v2/v3 | v1/v2/v3/v4 + IMSI column in summary |
| `plot_capture.py` | title: frame/slot/RNTI | title: + IMSI for v4 datasets |

## How labeling works

```
gNB plugin (nr_pusch_capture.c)
  ├── forks label_monitor.py at init
  ├── label thread connects to /tmp/pusch_label.sock
  │     receives: "A <rnti_hex> <imsi>\n"  (add)
  │               "R <rnti_hex>\n"          (remove)
  └── at capture time:
        IMSI known   → write immediately with IMSI embedded
        IMSI unknown → hold in pending queue
                        flush when label arrives from socket
                        flush at shutdown (labeled or unlabeled)

label_monitor.py
  ├── tails docker AMF logs for NAS registration events (IMSI → ran_id)
  ├── polls nrRRC_stats.log for RNTI → cu_ue_id mapping
  ├── joins: ran_ue_ngap_id == cu_ue_id → RNTI → IMSI
  └── broadcasts updates to plugin over Unix socket in real time
```

The pending queue is bounded by the AMF registration window (~15 seconds)
so memory overhead is constant regardless of total capture count.

## What is captured

Each accepted capture contains:
- frequency-domain PUSCH IQ samples (per OFDM symbol, per allocated subcarrier)
- slot and allocation metadata
- DMRS configuration metadata
- per-symbol valid RE counts
- **IMSI of the transmitting device** (v4 datasets)

Captures are accepted only if:
- the allocation window contains DMRS symbols
- the DMRS configuration exposes a supportable active-vs-quiet comb layout
- active-bin mean power ≥ `1.50×` quiet-bin mean power
- the IQ payload is not a duplicate of a previously accepted capture

## Repo layout

```
nr_pusch_capture/
    CMakeLists.txt
    README.md
    capture_config.txt
    src/
        nr_pusch_capture.c       ← v4 format, socket label thread, pending queue
    scripts/
        label_monitor.py         ← Unix socket server, real-time RNTI→IMSI push
        post_label.py            ← retroactive labeling fallback for v3 datasets
        read_dataset.py          ← v1/v2/v3/v4 reader with IMSI column
        evaluate_dataset.py
        plot_capture.py          ← IMSI in title for v4 datasets
        generate_capture_video.py
        backup/
            label_monitor_original.py   ← pre-socket polling-only version
            label_monitor_socket_v1.py  ← current socket version snapshot
            post_label_original.py      ← original post-hoc labeling script
    data/
        pusch_dataset.bin
```

## OAI deployment (OAIBOX with X410)

### Configure capture count

```bash
echo 100 > plugins/nr_pusch_capture/capture_config.txt
```

The plugin stops after `N` accepted captures and becomes passthrough.

### Build the plugin

```bash
sudo ninja -C /home/user/openairinterface5g/cmake_targets/ran_build/build \
    receiver_pusch_capture
```

### Run with plugin enabled

```bash
cd /home/user/openairinterface5g
sudo cmake_targets/ran_build/build/nr-softmodem \
    -O targets/PROJECTS/GENERIC-NR-5GC/CONF/oaibox.yaml \
    --usrp-tx-thread-config 1 \
    --T_stdout 2 --T_nowait \
    --loader.receiver.shlibversion _pusch_capture
```

The plugin automatically starts `label_monitor.py` and connects to it.
Watch for these lines confirming the labeling pipeline is active:

```
[nr_pusch_capture] Started label_monitor.py (PID XXXX, log: /tmp/label_monitor_pusch.log)
[nr_pusch_capture] Connected to label monitor at /tmp/pusch_label.sock
[nr_pusch_capture] Label: RNTI 0x1ce8 -> IMSI 001010000000010
[nr_pusch_capture] Flushed 45 pending capture(s) for RNTI 0x1ce8 → IMSI 001010000000010
```

Only one receiver plugin can be active at a time.

## Inspect captures

Print a summary table (v4 shows IMSI column):

```bash
python3 scripts/read_dataset.py data/pusch_dataset.bin
```

Evaluate DMRS presence, comb visibility, and duplicates:

```bash
python3 scripts/evaluate_dataset.py data/pusch_dataset.bin
```

Plot a single capture (IMSI shown in title for v4):

```bash
python3 scripts/plot_capture.py data/pusch_dataset.bin 0
```

Render all captures as a timelapse video:

```bash
python3 scripts/generate_capture_video.py data/pusch_dataset.bin
```

Retroactive labeling for v3 datasets (requires `label_map.json` from a
prior `label_monitor.py` run):

```bash
python3 scripts/post_label.py data/pusch_dataset.bin data/label_map.json \
    --npz --summary
```

## Dataset format

### v4 (this fork) — 144-byte capture header

File header (64 bytes):
- `magic`, `version`, `max_captures`, `num_captures`

Per-capture header (144 bytes):
- `record_bytes`, `capture_idx`
- timing: `frame`, `timestamp_ns` (CLOCK_MONOTONIC)
- slot, RNTI, modulation, layers
- PUSCH allocation: `start_symbol`, `num_symbols`, `rb_size`, `rb_start`, `bwp_start`
- DMRS config: `ul_dmrs_symb_pos`, `scid`, `ul_dmrs_scrambling_id`, `data_scrambling_id`,
  `transform_precoding`, `dmrs_config_type`, `num_dmrs_cdm_grps_no_data`, `dmrs_ports`
- PHY context: `ofdm_symbol_size`, `first_carrier_offset`, `nb_re_per_sym`, `output_shift`, `nvar`
- `valid_re[14]`
- `iq_bytes`
- **`imsi[16]`** ← new in v4

Payload:
- IQ only, interleaved `int16` real/imag samples

The Python reader is backward compatible with v1/v2/v3 datasets.

### Upstream reference

See [C2A2-at-Florida-Atlantic-University/nr_pusch_capture](https://github.com/C2A2-at-Florida-Atlantic-University/nr_pusch_capture)
for the base plugin without OAI-specific labeling.
