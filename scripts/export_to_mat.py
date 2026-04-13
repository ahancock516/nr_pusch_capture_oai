#!/usr/bin/env python3
"""
Export PUSCH captures to .mat files for MATLAB nrPUSCHDecode evaluation.

Each capture is exported as a .mat file containing:
  - rxGrid:   received resource grid [subcarriers x OFDM_symbols] complex double
  - chEst:    channel estimate grid  [subcarriers x OFDM_symbols] complex double
  - llr:      unscrambled LLR output [N x 1] int16
  - meta:     struct with all PUSCH allocation and PHY metadata

Usage:
    python export_to_mat.py [dataset_path] [capture_index | "all"]
    python export_to_mat.py                                          # export capture 0
    python export_to_mat.py path/to/pusch_dataset.bin 42             # single capture
    python export_to_mat.py path/to/pusch_dataset.bin all            # export all captures
"""

import sys
import os
import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.dirname(__file__))
from read_dataset import PUSCHDataset


def export_capture(ds, idx, out_dir):
    """Export a single capture to a .mat file."""
    cap = ds[idx]
    meta = cap["meta"]
    iq    = cap["iq"]      # [num_symbols, nb_re_per_sym] complex64
    chest = cap["chest"]   # [num_symbols, nb_re_per_sym] complex64
    llr   = cap["llr"]     # [total_llrs] int16

    num_symbols   = meta["num_symbols"]
    nb_re_per_sym = meta["nb_re_per_sym"]
    rb_size       = meta["rb_size"]
    start_symbol  = meta["start_symbol"]

    # MATLAB nrPUSCHDecode expects the resource grid as:
    #   [K x N] where K = subcarriers, N = OFDM symbols in the slot
    # We place our captured data into the correct symbol positions
    # within a full 14-symbol slot grid.
    num_slot_symbols = 14
    K = nb_re_per_sym  # number of allocated subcarriers

    # Build grids with NaN for unoccupied symbols (MATLAB convention)
    rx_grid  = np.full((K, num_slot_symbols), np.nan + 1j * np.nan,
                       dtype=np.complex128)
    ch_grid  = np.full((K, num_slot_symbols), np.nan + 1j * np.nan,
                       dtype=np.complex128)

    for s in range(num_symbols):
        sym_idx = start_symbol + s
        # Transpose: iq is [symbols, subcarriers] → grid is [subcarriers, symbols]
        rx_grid[:, sym_idx] = iq[s, :].astype(np.complex128)
        ch_grid[:, sym_idx] = chest[s, :].astype(np.complex128)

    # DMRS symbol mask as boolean array for the slot
    dmrs_mask = meta["ul_dmrs_symb_pos"]
    dmrs_symbols = np.zeros(num_slot_symbols, dtype=np.float64)
    for s in range(num_slot_symbols):
        dmrs_symbols[s] = 1.0 if (dmrs_mask >> s) & 1 else 0.0

    # Per-symbol valid RE counts (padded to 14)
    valid_re = np.zeros(num_slot_symbols, dtype=np.float64)
    for s in range(num_symbols):
        valid_re[start_symbol + s] = float(meta["valid_re"][s])

    # Build metadata struct for MATLAB
    mat_meta = {
        "capture_idx":            float(meta["capture_idx"]),
        "frame":                  float(meta["frame"]),
        "slot":                   float(meta["slot"]),
        "rnti":                   float(meta["rnti"]),
        "qam_mod_order":          float(meta["qam_mod_order"]),
        "num_layers":             float(meta["num_layers"]),
        "start_symbol":           float(start_symbol),
        "num_symbols":            float(num_symbols),
        "rb_size":                float(rb_size),
        "rb_start":               float(meta["rb_start"]),
        "bwp_start":              float(meta["bwp_start"]),
        "ul_dmrs_symb_pos":       float(meta["ul_dmrs_symb_pos"]),
        "dmrs_symbols":           dmrs_symbols,
        "scid":                   float(meta["scid"]),
        "ul_dmrs_scrambling_id":  float(meta["ul_dmrs_scrambling_id"]),
        "data_scrambling_id":     float(meta["data_scrambling_id"]),
        "ofdm_symbol_size":       float(meta["ofdm_symbol_size"]),
        "first_carrier_offset":   float(meta["first_carrier_offset"]),
        "nb_re_per_sym":          float(nb_re_per_sym),
        "output_shift":           float(meta["output_shift"]),
        "nvar":                   float(meta["nvar"]),
        "valid_re":               valid_re,
    }

    # Modulation string for MATLAB
    mod_table = {2: "QPSK", 4: "16QAM", 6: "64QAM", 8: "256QAM"}
    mat_meta["modulation"] = mod_table.get(meta["qam_mod_order"], "QPSK")

    out_path = os.path.join(out_dir, f"capture_{idx:04d}.mat")
    sio.savemat(out_path, {
        "rxGrid":  rx_grid,
        "chEst":   ch_grid,
        "llr":     llr.astype(np.float64).reshape(-1, 1),
        "meta":    mat_meta,
    }, do_compression=True)

    print(f"  [{idx:4d}] frame={meta['frame']}, slot={meta['slot']}, "
          f"RNTI={meta['rnti']}, {mat_meta['modulation']}, "
          f"{rb_size} PRBs → {out_path}")
    return out_path


if __name__ == "__main__":
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else \
        "plugins/nr_pusch_capture/data/pusch_dataset.bin"
    which = sys.argv[2] if len(sys.argv) > 2 else "0"

    ds = PUSCHDataset(dataset_path)
    print(f"Dataset: {ds}")

    out_dir = os.path.join(os.path.dirname(dataset_path) or ".", "mat")
    os.makedirs(out_dir, exist_ok=True)

    if which.lower() == "all":
        print(f"Exporting all {len(ds)} captures to {out_dir}/")
        for i in range(len(ds)):
            export_capture(ds, i, out_dir)
        print(f"\nDone. Exported {len(ds)} .mat files.")
    else:
        idx = int(which)
        if idx >= len(ds):
            print(f"Error: capture index {idx} out of range [0, {len(ds)})")
            sys.exit(1)
        export_capture(ds, idx, out_dir)
