#!/usr/bin/env python3
"""
post_label.py — Attach IMSI labels to a PUSCH capture dataset.

Joins pusch_dataset.bin (from the nr_pusch_capture plugin) against
label_map.json (from label_monitor.py) using RNTI and CLOCK_MONOTONIC
timestamp to identify which device produced each capture.

Each capture is matched to a session by:
  1. rnti must equal session.rnti
  2. capture.timestamp_ns must fall within [session.t_start_mono_ns,
     session.t_end_mono_ns]  (open upper bound for still-active sessions)

Outputs
-------
  <dataset>.labels.json   — {capture_idx: imsi | null}  (always written)
  --npz                   — numpy .npz with iq, chest, label arrays
                            (only for captures sharing the same shape —
                            see note below)
  --summary               — per-device capture count table

Shape note
----------
  PUSCH allocations vary in rb_size, num_symbols, and qam_mod_order across
  the dataset.  The --npz export groups captures by shape and writes a
  separate array per group.  For fixed-shape training data, run the
  experiment with a single MCS target or filter in post-processing.

Usage
-----
  python3 post_label.py data/pusch_dataset.bin data/label_map.json
  python3 post_label.py data/pusch_dataset.bin data/label_map.json --npz --summary
  python3 post_label.py data/pusch_dataset.bin data/label_map.json \\
      --output data/my_labels.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from read_dataset import PUSCHDataset


# ── label resolution ─────────────────────────────────────────────────────────

def load_label_map(path: Path):
    raw = json.loads(path.read_text())
    sessions = raw["sessions"]
    offset   = raw["mono_to_wall_offset_ns"]
    return sessions, offset


def resolve(rnti: int, ts_mono_ns: int, sessions: list) -> Optional[str]:
    """
    Return the IMSI for a capture with the given RNTI and monotonic timestamp,
    or None if no session matches.
    """
    for s in sessions:
        if s["rnti"] != rnti:
            continue
        t_start = s.get("t_start_mono_ns")
        t_end   = s.get("t_end_mono_ns")   # None = session still active
        if t_start is None:
            continue
        if ts_mono_ns < t_start:
            continue
        if t_end is not None and ts_mono_ns > t_end:
            continue
        return s["imsi"]
    return None


# ── summary ───────────────────────────────────────────────────────────────────

def print_summary(labels: dict, sessions: list):
    counts: dict[str, int] = defaultdict(int)
    for imsi in labels.values():
        counts[imsi if imsi is not None else "(unlabeled)"] += 1

    print(f"\n{'IMSI':<25}  {'Captures':>8}")
    print("-" * 36)
    for key, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"{key:<25}  {n:>8}")
    print("-" * 36)
    print(f"{'Total':<25}  {len(labels):>8}")

    if sessions:
        print(f"\nLabel map sessions: {len(sessions)}")
        for s in sessions:
            t_start = s.get("t_start_mono_ns", "?")
            t_end   = s.get("t_end_mono_ns", "active")
            print(
                f"  RNTI=0x{s['rnti']:04x}  IMSI={s['imsi']:<25} "
                f"cu_ue_id={s.get('cu_ue_id','?')}  "
                f"status={s.get('status','?')}"
            )


# ── npz export ────────────────────────────────────────────────────────────────

def export_npz(ds: PUSCHDataset, labels: dict, out_path: Path):
    """
    Export labeled captures grouped by IQ shape to numpy .npz.

    Each shape group is stored as:
      iq_<rows>x<cols>       — float32 complex interleaved [N, rows, cols, 2]
      chest_<rows>x<cols>    — float32 complex interleaved [N, rows, cols, 2]
      label_<rows>x<cols>    — int32 class index (-1 = unlabeled) [N]
      idx_<rows>x<cols>      — uint32 capture_idx [N]
      ts_<rows>x<cols>       — int64 timestamp_ns [N]

    label_names              — str array mapping class index to IMSI
    """
    unique_labels = sorted({v for v in labels.values() if v is not None})
    label_to_int  = {imsi: i for i, imsi in enumerate(unique_labels)}
    print(f"\nClasses ({len(unique_labels)}): {unique_labels}")

    # Group by (num_symbols, nb_re_per_sym)
    groups: dict[tuple, list] = defaultdict(list)
    for cap in ds:
        meta  = cap["meta"]
        shape = (meta["num_symbols"], meta["nb_re_per_sym"])
        imsi  = labels.get(meta["capture_idx"])
        groups[shape].append((cap, label_to_int.get(imsi, -1)))

    arrays = {"label_names": np.array(unique_labels)}

    for (nsym, nre), items in sorted(groups.items()):
        key = f"{nsym}x{nre}"
        n   = len(items)
        iq_arr    = np.empty((n, nsym, nre, 2), dtype=np.float32)
        chest_arr = np.empty((n, nsym, nre, 2), dtype=np.float32)
        lbl_arr   = np.empty(n, dtype=np.int32)
        idx_arr   = np.empty(n, dtype=np.uint32)
        ts_arr    = np.empty(n, dtype=np.int64)

        for i, (cap, lbl) in enumerate(items):
            iq_arr[i, :, :, 0]    = cap["iq"].real
            iq_arr[i, :, :, 1]    = cap["iq"].imag
            chest_arr[i, :, :, 0] = cap["chest"].real
            chest_arr[i, :, :, 1] = cap["chest"].imag
            lbl_arr[i]            = lbl
            idx_arr[i]            = cap["meta"]["capture_idx"]
            ts_arr[i]             = cap["meta"]["timestamp_ns"]

        arrays[f"iq_{key}"]    = iq_arr
        arrays[f"chest_{key}"] = chest_arr
        arrays[f"label_{key}"] = lbl_arr
        arrays[f"idx_{key}"]   = idx_arr
        arrays[f"ts_{key}"]    = ts_arr
        print(f"  Shape {key}: {n} captures")

    np.savez_compressed(out_path, **arrays)
    print(f"Saved {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Attach IMSI labels to a PUSCH capture dataset."
    )
    ap.add_argument("dataset",
                    help="Path to pusch_dataset.bin")
    ap.add_argument("label_map",
                    help="Path to label_map.json produced by label_monitor.py")
    ap.add_argument("--output", default=None,
                    help="Output labels JSON (default: <dataset>.labels.json)")
    ap.add_argument("--npz", action="store_true",
                    help="Also export a labeled .npz dataset grouped by IQ shape")
    ap.add_argument("--summary", action="store_true",
                    help="Print per-device capture counts")
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    label_path   = Path(args.label_map)
    output_path  = (
        Path(args.output) if args.output
        else dataset_path.with_suffix(".labels.json")
    )

    ds, _         = PUSCHDataset(dataset_path), None
    sessions, _   = load_label_map(label_path)

    print(f"Dataset   : {ds}")
    print(f"Label map : {len(sessions)} session(s)")

    labels: dict[int, Optional[str]] = {}
    for cap in ds:
        meta = cap["meta"]
        imsi = resolve(meta["rnti"], meta["timestamp_ns"], sessions)
        labels[meta["capture_idx"]] = imsi

    labeled   = sum(1 for v in labels.values() if v is not None)
    unlabeled = len(labels) - labeled
    print(f"Labeled   : {labeled} / {len(labels)}")
    if unlabeled:
        print(
            f"Unlabeled : {unlabeled}  "
            f"(captures outside any session window — check label_map.json)"
        )

    output_path.write_text(json.dumps(labels, indent=2))
    print(f"Written   : {output_path}")

    if args.summary:
        print_summary(labels, sessions)

    if args.npz:
        npz_path = dataset_path.with_suffix(".labeled.npz")
        export_npz(ds, labels, npz_path)


if __name__ == "__main__":
    main()
