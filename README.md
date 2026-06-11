# NR PUSCH Capture Plugin

A receiver-side plugin for the Sionna Research Kit that stores only the PUSCH IQ needed to study DMRS comb visibility.

The plugin now has a single purpose:
- capture frequency-domain PUSCH IQ
- keep only captures whose DMRS RE comb is clearly visible
- skip duplicate IQ payloads

Legacy channel-estimate, LLR, MATLAB export, and receiver-comparison tooling were removed to keep the repo centered on IQ capture quality.

## What is stored

Each accepted capture contains:
- IQ samples for the allocated PUSCH REs
- slot and allocation metadata
- DMRS configuration metadata needed to evaluate the expected comb layout
- per-symbol valid RE counts

Captures are accepted only if all of the following are true:
- the allocation window contains DMRS symbols
- the DMRS configuration exposes a supportable active-vs-quiet comb layout
- the received IQ shows a strong comb with active-bin mean power at least `1.50x` the quiet-bin mean power
- the IQ payload is not a duplicate of a previously accepted capture

## Repo layout

```
plugins/nr_pusch_capture/
    CMakeLists.txt
    README.md
    capture_config.txt
    src/
        nr_pusch_capture.c
    scripts/
        read_dataset.py
        evaluate_dataset.py
        plot_capture.py
        generate_capture_video.py
    tests/
        test_read_dataset.py
        test_evaluate_dataset.py
        test_generate_capture_video.py
    data/
        pusch_dataset.bin
```

## Configure capture count

```bash
echo 500 > plugins/nr_pusch_capture/capture_config.txt
```

The plugin stops after `N` accepted captures and then becomes passthrough.

## Build and enable the plugin

Rebuild the gNB image after source changes:

```bash
cd /home/caai/Workspace/sionna-rk
make build-gnb
```

Enable the plugin in `config/rfsim/.env` or the config you are using:

```bash
GNB_EXTRA_OPTIONS="--loader.receiver.shlibversion _pusch_capture"
```

Only one receiver plugin can be active at a time.

## Run in simulation

Start the system:

```bash
cd /home/caai/Workspace/sionna-rk
./scripts/start_system.sh rfsim
```

Confirm the plugin loaded:

```bash
docker logs oai-gnb 2>&1 | grep pusch_capture
```

Generate uplink traffic so PUSCH is scheduled:

```bash
docker exec oai-nr-ue iperf3 -c 192.168.72.135 -u -b 10M -t 60
```

The dataset is written to `plugins/nr_pusch_capture/data/pusch_dataset.bin`.

## Read and inspect captures

Print a dataset summary:

```bash
cd /home/caai/Workspace/sionna-rk
./env/bin/python plugins/nr_pusch_capture/scripts/read_dataset.py \
  plugins/nr_pusch_capture/data/pusch_dataset.bin
```

Evaluate DMRS presence, comb visibility, and duplicate payloads:

```bash
cd /home/caai/Workspace/sionna-rk
./env/bin/python plugins/nr_pusch_capture/scripts/evaluate_dataset.py \
  plugins/nr_pusch_capture/data/pusch_dataset.bin \
  --reference-start 1 --reference-end 23 --one-based
```

Render one capture as a PNG focused on the DMRS comb:

```bash
cd /home/caai/Workspace/sionna-rk
./env/bin/python plugins/nr_pusch_capture/scripts/plot_capture.py \
  plugins/nr_pusch_capture/data/pusch_dataset.bin 0
```

Render all captures into a video at `2 fps` (`0.5 s` per frame):

```bash
cd /home/caai/Workspace/sionna-rk
./env/bin/python plugins/nr_pusch_capture/scripts/generate_capture_video.py \
  plugins/nr_pusch_capture/data/pusch_dataset.bin
```

## Dataset format

New captures use format version `3`.

File header:
- `magic`
- `version`
- `max_captures`
- `num_captures`

Per-capture header:
- `record_bytes`
- `capture_idx`
- timing and slot identifiers
- PUSCH allocation metadata
- DMRS configuration metadata
- PHY context (`ofdm_symbol_size`, `first_carrier_offset`, `nb_re_per_sym`, `output_shift`, `nvar`)
- `valid_re[14]`
- `iq_bytes`

Payload:
- IQ only, stored as interleaved `int16` real/imag samples

The Python reader remains backward compatible with legacy v1/v2 datasets.
