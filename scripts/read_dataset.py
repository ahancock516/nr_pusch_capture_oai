#!/usr/bin/env python3
"""
Reader for PUSCH capture binary datasets produced by nr_pusch_capture plugin.

Usage:
    from read_dataset import PUSCHDataset

    ds = PUSCHDataset("plugins/nr_pusch_capture/data/pusch_dataset.bin")
    print(ds)                    # summary
    cap = ds[0]                  # first capture (dict)
    iq   = cap["iq"]             # complex64 array [num_symbols, nb_re_per_sym]
    chest = cap["chest"]         # complex64 array [num_symbols, nb_re_per_sym]
    llr  = cap["llr"]            # int16 array, flat
    meta = cap["meta"]           # dict of scalar metadata fields

    # Iterate all captures
    for cap in ds:
        ...
"""

import struct
import numpy as np
from pathlib import Path

FILE_HEADER_BYTES = 64
CAPTURE_HEADER_BYTES = 128
PUSCH_FILE_MAGIC = 0x50555343
MAX_SYMBOLS_PER_SLOT = 14

# Struct format for file header (little-endian, packed)
_FILE_HDR_FMT = "<III I 48s"
_FILE_HDR_SIZE = struct.calcsize(_FILE_HDR_FMT)

# Struct format for capture header (little-endian, packed)
# Fields: record_bytes, capture_idx, frame(i64), timestamp_ns(i64),
#         slot(i32), rnti(u16), qam_mod_order(u8), num_layers(u8),
#         start_symbol, num_symbols, rb_size, rb_start, bwp_start,
#         ul_dmrs_symb_pos(u32), scid, ul_dmrs_scrambling_id, data_scrambling_id,
#         ofdm_symbol_size, first_carrier_offset, nb_re_per_sym,
#         output_shift, nvar(u32),
#         valid_re[14] (14 x i16),
#         iq_bytes, chest_bytes, llr_bytes
_CAP_HDR_FMT = (
    "<"        # little-endian
    "I I"      # record_bytes, capture_idx
    "q q"      # frame, timestamp_ns
    "i H B B"  # slot, rnti, qam_mod_order, num_layers
    "i i i i i" # start_symbol, num_symbols, rb_size, rb_start, bwp_start
    "I i i i"  # ul_dmrs_symb_pos, scid, ul_dmrs_scrambling_id, data_scrambling_id
    "i i i"    # ofdm_symbol_size, first_carrier_offset, nb_re_per_sym
    "i I"      # output_shift, nvar
    f"{MAX_SYMBOLS_PER_SLOT}h" # valid_re[14]
    "i i i"    # iq_bytes, chest_bytes, llr_bytes
)
_CAP_HDR_SIZE = struct.calcsize(_CAP_HDR_FMT)

_META_FIELDS = [
    "record_bytes", "capture_idx",
    "frame", "timestamp_ns",
    "slot", "rnti", "qam_mod_order", "num_layers",
    "start_symbol", "num_symbols", "rb_size", "rb_start", "bwp_start",
    "ul_dmrs_symb_pos", "scid", "ul_dmrs_scrambling_id", "data_scrambling_id",
    "ofdm_symbol_size", "first_carrier_offset", "nb_re_per_sym",
    "output_shift", "nvar",
]


def _parse_capture_header(buf):
    """Parse a capture header from bytes, return metadata dict."""
    vals = struct.unpack(_CAP_HDR_FMT, buf)

    meta = {}
    idx = 0
    for name in _META_FIELDS:
        meta[name] = vals[idx]
        idx += 1

    # valid_re array (14 entries)
    meta["valid_re"] = np.array(vals[idx:idx + MAX_SYMBOLS_PER_SLOT],
                                dtype=np.int16)
    idx += MAX_SYMBOLS_PER_SLOT

    meta["iq_bytes"] = vals[idx]
    meta["chest_bytes"] = vals[idx + 1]
    meta["llr_bytes"] = vals[idx + 2]
    return meta


class PUSCHDataset:
    """Reader for PUSCH capture binary datasets."""

    def __init__(self, path):
        self.path = Path(path)
        self._data = self.path.read_bytes()

        # Parse file header
        hdr = struct.unpack(_FILE_HDR_FMT, self._data[:_FILE_HDR_SIZE])
        magic, version, max_captures, num_captures, _ = hdr
        if magic != PUSCH_FILE_MAGIC:
            raise ValueError(
                f"Bad magic: 0x{magic:08X} (expected 0x{PUSCH_FILE_MAGIC:08X})"
            )
        if version != 1:
            raise ValueError(f"Unsupported format version: {version}")

        self.version = version
        self.max_captures = max_captures
        self.num_captures = num_captures

        # Build index of capture offsets
        self._offsets = []
        pos = FILE_HEADER_BYTES
        for _ in range(num_captures):
            if pos + CAPTURE_HEADER_BYTES > len(self._data):
                break
            record_bytes = struct.unpack_from("<I", self._data, pos)[0]
            self._offsets.append(pos)
            pos += record_bytes
        self.num_captures = len(self._offsets)

    def __len__(self):
        return self.num_captures

    def __repr__(self):
        return (f"PUSCHDataset({self.path.name}: "
                f"{self.num_captures}/{self.max_captures} captures)")

    def __getitem__(self, idx):
        if idx < 0:
            idx += self.num_captures
        if idx < 0 or idx >= self.num_captures:
            raise IndexError(f"Capture index {idx} out of range "
                             f"[0, {self.num_captures})")
        return self._read_capture(self._offsets[idx])

    def __iter__(self):
        for i in range(self.num_captures):
            yield self[i]

    def _read_capture(self, offset):
        """Read a single capture record at the given file offset."""
        hdr_buf = self._data[offset:offset + CAPTURE_HEADER_BYTES]
        meta = _parse_capture_header(hdr_buf)

        num_symbols = meta["num_symbols"]
        nb_re_per_sym = meta["nb_re_per_sym"]
        total_re = num_symbols * nb_re_per_sym

        pos = offset + CAPTURE_HEADER_BYTES

        # IQ samples: int16 pairs → complex64
        iq_raw = np.frombuffer(
            self._data, dtype=np.int16, count=total_re * 2, offset=pos
        ).reshape(num_symbols, nb_re_per_sym, 2).copy()
        iq = (iq_raw[..., 0] + 1j * iq_raw[..., 1]).astype(np.complex64)
        pos += meta["iq_bytes"]

        # Channel estimates: int16 pairs → complex64
        chest_raw = np.frombuffer(
            self._data, dtype=np.int16, count=total_re * 2, offset=pos
        ).reshape(num_symbols, nb_re_per_sym, 2).copy()
        chest = (chest_raw[..., 0] + 1j * chest_raw[..., 1]).astype(np.complex64)
        pos += meta["chest_bytes"]

        # LLRs: flat int16 array
        llr_count = meta["llr_bytes"] // 2
        llr = np.frombuffer(
            self._data, dtype=np.int16, count=llr_count, offset=pos
        ).copy()

        return {
            "meta": meta,
            "iq": iq,
            "chest": chest,
            "llr": llr,
        }

    def summary(self):
        """Print a summary of all captures in the dataset."""
        print(repr(self))
        print(f"{'idx':>5} {'frame':>8} {'slot':>4} {'RNTI':>6} "
              f"{'mod':>4} {'PRBs':>5} {'syms':>4} {'REs':>6}")
        print("-" * 50)
        for i, cap in enumerate(self):
            m = cap["meta"]
            mod_names = {2: "QPSK", 4: "16Q", 6: "64Q", 8: "256Q"}
            mod = mod_names.get(m["qam_mod_order"], f"Q{m['qam_mod_order']}")
            total_re = m["num_symbols"] * m["nb_re_per_sym"]
            print(f"{i:5d} {m['frame']:8d} {m['slot']:4d} {m['rnti']:6d} "
                  f"{mod:>4s} {m['rb_size']:5d} {m['num_symbols']:4d} "
                  f"{total_re:6d}")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else \
        "plugins/nr_pusch_capture/data/pusch_dataset.bin"
    ds = PUSCHDataset(path)
    ds.summary()

    if len(ds) > 0:
        cap = ds[0]
        print(f"\nFirst capture details:")
        print(f"  IQ shape:    {cap['iq'].shape}")
        print(f"  ChEst shape: {cap['chest'].shape}")
        print(f"  LLR count:   {len(cap['llr'])}")
        print(f"  IQ range:    [{cap['iq'].real.min():.0f}, "
              f"{cap['iq'].real.max():.0f}]")
