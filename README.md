# NR PUSCH Capture Plugin

A receiver-level plugin for the Sionna Research Kit that captures raw 5G NR PUSCH data into a binary dataset for offline analysis. It hooks into the OAI gNB receiver pipeline to collect frequency-domain IQ samples, channel estimates, unscrambled LLRs, and full slot metadata for each PUSCH transmission.

## What it captures

Each slot capture contains:

| Data | Format | Description |
|------|--------|-------------|
| **IQ samples** | int16 (real, imag) per subcarrier per OFDM symbol | Raw frequency-domain received signal at allocated PUSCH subcarriers |
| **Channel estimates** | int16 (real, imag) per subcarrier per OFDM symbol | DMRS-interpolated channel estimates from OAI |
| **LLRs** | int16 per bit | Unscrambled soft LLR output from the default OAI receiver |
| **Metadata** | Per-capture header | Frame, slot, RNTI, modulation order, PRB allocation, DMRS config, scrambling IDs, noise variance, per-symbol valid RE counts |

## Where it hooks in

The plugin uses the OAI **receiver plugin interface** (same as the neural receiver). It is called twice per PUSCH slot by `nr_ulsch_demodulation.c`:

```
UE transmits PUSCH
       |
       v
gNB FFT --> rxdataF (freq-domain IQ per antenna)
       |
       v
nr_ulsch_demodulation.c calls receiver_compute_llr()
       |
       +---> [Phase 1] Plugin captures rxF + channel estimates
       |     Returns -1: "run default receiver, then call me again"
       |
       v
OAI default inner_rx (channel estimation, MMSE equalization, demapping, unscrambling)
       |
       +---> [Phase 2] Plugin captures unscrambled LLRs
       |     Writes complete binary record to dataset file
       |
       v
LDPC decoding continues normally
```

The plugin is **non-invasive**: it does not modify the receiver processing. OAI runs its full default pipeline; the plugin only reads and records data.

## Prerequisites

- Sionna Research Kit installed and operational
- Docker images built (gNB, UE, core network)
- A working 5G setup (rfsim or hardware)

## File structure

```
plugins/nr_pusch_capture/
    INSTRUCTIONS.md          # This file
    CMakeLists.txt           # Build configuration
    capture_config.txt       # Number of captures (edit before running)
    src/
        nr_pusch_capture.c   # Plugin C source
    scripts/
        read_dataset.py      # Python: read binary dataset into numpy arrays
        plot_capture.py      # Python: multi-panel analysis plot of a single capture
        export_to_mat.py     # Python: export captures to .mat for MATLAB
        evaluate_pusch_capture.m  # MATLAB: nrPUSCHDecode evaluation
    data/                    # Output directory (created automatically)
        pusch_dataset.bin    # Binary dataset (generated at runtime)
```

## Step 1: Configure the capture count

Edit `capture_config.txt` to set how many PUSCH slots to capture. The plugin reads this file at startup and stops automatically after N captures, becoming zero-overhead passthrough.

```bash
echo 500 > plugins/nr_pusch_capture/capture_config.txt
```

## Step 2: Rebuild the gNB Docker image

The plugin source is compiled into the gNB during the Docker build. The `plugins/` directory is copied into the build context and CMake picks up our `CMakeLists.txt` via `add_subdirectory(nr_pusch_capture)` in the top-level `plugins/CMakeLists.txt`.

The gNB Dockerfile copies `libreceiver*.so` (a glob), so our `libreceiver_pusch_capture.so` is included automatically.

```bash
cd /path/to/sionna-rk
make build-gnb
```

This may take a while on the first build, but Docker layer caching means subsequent builds (after plugin-only changes) are faster.

## Step 3: Enable the plugin

Edit the `.env` file for your configuration. For RF simulator:

```bash
# config/rfsim/.env
GNB_EXTRA_OPTIONS="--loader.receiver.shlibversion _pusch_capture"
```

For B200 hardware:

```bash
# config/b200/.env
GNB_EXTRA_OPTIONS="--loader.receiver.shlibversion _pusch_capture"
```

**Note**: Only one receiver plugin can be active at a time. This replaces any existing `--loader.receiver.shlibversion` setting (e.g., `_neural_rx`, `_capture`).

## Step 4: Prepare the output directory

The `plugins/` directory is bind-mounted into the gNB container at `/opt/oai-gnb/plugins`. The captured data is written to `plugins/nr_pusch_capture/data/pusch_dataset.bin` on the host.

```bash
mkdir -p plugins/nr_pusch_capture/data
chmod 777 plugins/nr_pusch_capture/data
```

## Step 5: Start the system

```bash
./scripts/start_system.sh rfsim
```

Check the gNB logs to confirm the plugin loaded:

```bash
docker logs oai-gnb 2>&1 | grep pusch_capture
```

Expected output:

```
[nr_pusch_capture] Initializing — will capture 500 slots
[nr_pusch_capture] Output: plugins/nr_pusch_capture/data/pusch_dataset.bin
```

## Step 6: Generate uplink traffic

The plugin only captures when there is PUSCH data scheduled. For rfsim, exec into the UE container and run iperf3:

```bash
docker exec oai-nr-ue iperf3 -c 192.168.72.135 -u -b 10M -t 60
```

Progress appears in the gNB logs:

```
[nr_pusch_capture] Captured 50 / 500 slots
[nr_pusch_capture] Captured 100 / 500 slots
...
[nr_pusch_capture] Dataset complete (500 captures). Plugin now passthrough.
```

Without active UL traffic, only periodic scheduling (e.g., BSR, SR) will trigger captures, which may take longer to reach N.

## Step 7: Analyze the dataset

### Python: read the dataset

```python
import sys
sys.path.insert(0, "plugins/nr_pusch_capture/scripts")
from read_dataset import PUSCHDataset

ds = PUSCHDataset("plugins/nr_pusch_capture/data/pusch_dataset.bin")
print(ds)           # PUSCHDataset(pusch_dataset.bin: 500/500 captures)
ds.summary()        # Print table of all captures

cap = ds[0]
iq    = cap["iq"]     # complex64 [num_symbols, nb_re_per_sym]
chest = cap["chest"]  # complex64 [num_symbols, nb_re_per_sym]
llr   = cap["llr"]    # int16 [total_llrs]
meta  = cap["meta"]   # dict with all metadata fields
```

### Python: plot a single capture

Generates a 12-panel analysis figure with IQ waveform, resource grid, constellation, channel estimates, LLR distribution, per-symbol power, and DMRS analysis (power spectrogram, constellation, phase error).

```bash
python3 plugins/nr_pusch_capture/scripts/plot_capture.py \
    plugins/nr_pusch_capture/data/pusch_dataset.bin 0
```

Output: `plugins/nr_pusch_capture/data/capture_0000.png`

### MATLAB: evaluate with 5G Toolbox

Export captures to `.mat` format, then use `nrPUSCHDecode` for standard-compliant evaluation.

```bash
# Export single capture
python3 plugins/nr_pusch_capture/scripts/export_to_mat.py \
    plugins/nr_pusch_capture/data/pusch_dataset.bin 0

# Export all captures
python3 plugins/nr_pusch_capture/scripts/export_to_mat.py \
    plugins/nr_pusch_capture/data/pusch_dataset.bin all
```

In MATLAB (requires 5G Toolbox):

```matlab
addpath('plugins/nr_pusch_capture/scripts');
results = evaluate_pusch_capture('plugins/nr_pusch_capture/data/mat/capture_0000.mat');
```

The MATLAB script runs two evaluation paths:
- **Path A (OAI-H)**: Uses OAI's channel estimates, MMSE equalization, then `nrPUSCHDecode`
- **Path B (ML-H)**: Uses MATLAB's `nrChannelEstimate` from DMRS, then `nrPUSCHDecode`

It reports sign agreement, correlation, and EVM between OAI and MATLAB LLR outputs.

## Binary dataset format

The dataset is a single binary file with the following structure:

```
[File Header — 64 bytes]
    uint32  magic           0x50555343 ("PUSC")
    uint32  version         1
    uint32  max_captures    configured N
    uint32  num_captures    captures written (updated live)
    uint8   reserved[48]

[Capture Record — variable length, repeated num_captures times]
    [Header — 128 bytes]
        uint32  record_bytes        total bytes of this record
        uint32  capture_idx         0-based index
        int64   frame               radio frame number
        int64   timestamp_ns        CLOCK_MONOTONIC nanoseconds
        int32   slot
        uint16  rnti
        uint8   qam_mod_order       2=QPSK, 4=16QAM, 6=64QAM, 8=256QAM
        uint8   num_layers
        int32   start_symbol
        int32   num_symbols
        int32   rb_size             number of allocated PRBs
        int32   rb_start
        int32   bwp_start
        uint32  ul_dmrs_symb_pos    DMRS symbol bitmask
        int32   scid
        int32   ul_dmrs_scrambling_id
        int32   data_scrambling_id
        int32   ofdm_symbol_size    FFT size
        int32   first_carrier_offset
        int32   nb_re_per_sym       subcarriers per symbol
        int32   output_shift        log2_maxh
        uint32  nvar                noise variance estimate
        int16   valid_re[14]        per-symbol valid RE counts
        int32   iq_bytes            IQ payload size
        int32   chest_bytes         channel estimate payload size
        int32   llr_bytes           LLR payload size

    [IQ Data — iq_bytes]
        int16 pairs (real, imag) x num_symbols x nb_re_per_sym

    [Channel Estimates — chest_bytes]
        int16 pairs (real, imag) x num_symbols x nb_re_per_sym

    [LLR Data — llr_bytes]
        int16 x total_llrs (per-symbol, using valid_re counts)
```

All values are little-endian. Structs are `__attribute__((packed))` in C.

## Recapturing data

To run a new capture session:

1. Delete or rename the existing dataset:
   ```bash
   rm plugins/nr_pusch_capture/data/pusch_dataset.bin
   ```
2. Optionally update `capture_config.txt` with a new count
3. Restart the gNB:
   ```bash
   ./scripts/stop_system.sh
   ./scripts/start_system.sh rfsim
   ```

## Troubleshooting

**Plugin not loading**: Check `docker logs oai-gnb` for `[LOADER]` messages. Ensure `GNB_EXTRA_OPTIONS` is set correctly in the `.env` file and the gNB image was rebuilt after adding the plugin.

**No captures appearing**: The plugin only captures when PUSCH is scheduled. Generate uplink traffic with iperf3. Without traffic, only periodic UL control messages (BSR/SR) trigger captures at ~1 per 100 ms.

**Permission denied on output file**: The gNB runs as root inside the container. Ensure the `data/` directory has write permissions: `chmod 777 plugins/nr_pusch_capture/data`.

**Only QPSK captures**: The default OAI scheduler starts with low MCS. To force higher modulation, add MCS options:
```bash
GNB_EXTRA_OPTIONS="--loader.receiver.shlibversion _pusch_capture --MACRLCs.[0].ul_max_mcs 28"
```