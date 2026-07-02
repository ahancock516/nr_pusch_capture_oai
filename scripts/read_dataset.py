#!/usr/bin/env python3
"""
Reader for PUSCH capture datasets produced by nr_pusch_capture.

Format support:
  - v1/v2: legacy datasets that may also include channel estimates and LLRs
  - v3:    simplified IQ-only datasets with DMRS metadata and no extra payloads
  - v4:    v3 + IMSI label
  - v5:    v4 + gNB's own UL channel estimate (chest), same shape as IQ
"""

import struct
from pathlib import Path

import numpy as np

FILE_HEADER_BYTES = 64
CAPTURE_HEADER_BYTES_V1 = 128
CAPTURE_HEADER_BYTES_V2 = 136
CAPTURE_HEADER_BYTES_V3 = 128
CAPTURE_HEADER_BYTES_V4 = 144
CAPTURE_HEADER_BYTES_V5 = 148
CAPTURE_HEADER_BYTES = CAPTURE_HEADER_BYTES_V3
PUSCH_FILE_MAGIC = 0x50555343
MAX_SYMBOLS_PER_SLOT = 14

_FILE_HDR_FMT = "<III I 48s"
_FILE_HDR_SIZE = struct.calcsize(_FILE_HDR_FMT)

_CAP_HDR_FMT_V1 = (
    "<"
    "I I"
    "q q"
    "i H B B"
    "i i i i i"
    "I i i i"
    "i i i"
    "i I"
    f"{MAX_SYMBOLS_PER_SLOT}h"
    "i i i"
)
_META_FIELDS_V1 = [
    "record_bytes", "capture_idx",
    "frame", "timestamp_ns",
    "slot", "rnti", "qam_mod_order", "num_layers",
    "start_symbol", "num_symbols", "rb_size", "rb_start", "bwp_start",
    "ul_dmrs_symb_pos", "scid", "ul_dmrs_scrambling_id", "data_scrambling_id",
    "ofdm_symbol_size", "first_carrier_offset", "nb_re_per_sym",
    "output_shift", "nvar",
]

_CAP_HDR_FMT_V2 = (
    "<"
    "I I"
    "q q"
    "i H B B"
    "i i i i i"
    "I i i i"
    "B B B B H H"
    "i i i"
    "i I"
    f"{MAX_SYMBOLS_PER_SLOT}h"
    "i i i"
)
_META_FIELDS_V2 = [
    "record_bytes", "capture_idx",
    "frame", "timestamp_ns",
    "slot", "rnti", "qam_mod_order", "num_layers",
    "start_symbol", "num_symbols", "rb_size", "rb_start", "bwp_start",
    "ul_dmrs_symb_pos", "scid", "ul_dmrs_scrambling_id", "data_scrambling_id",
    "transform_precoding", "dmrs_config_type", "num_dmrs_cdm_grps_no_data",
    "reserved0", "dmrs_ports", "reserved1",
    "ofdm_symbol_size", "first_carrier_offset", "nb_re_per_sym",
    "output_shift", "nvar",
]

_CAP_HDR_FMT_V3 = (
    "<"
    "I I"
    "q q"
    "i H B B"
    "i i i i i"
    "I i i i"
    "B B B B H H"
    "i i i"
    "i I"
    f"{MAX_SYMBOLS_PER_SLOT}h"
    "i"
)
_META_FIELDS_V3 = list(_META_FIELDS_V2)

_CAP_HDR_FMT_V4 = _CAP_HDR_FMT_V3 + "16s"
_META_FIELDS_V4 = list(_META_FIELDS_V3)

_CAP_HDR_FMT_V5 = _CAP_HDR_FMT_V4 + "i"
_META_FIELDS_V5 = list(_META_FIELDS_V4)


def _parse_capture_header(buf, fmt, field_names, payload_fields):
    vals = struct.unpack(fmt, buf)
    meta = {}
    idx = 0
    for name in field_names:
        meta[name] = vals[idx]
        idx += 1
    meta["valid_re"] = np.array(vals[idx:idx + MAX_SYMBOLS_PER_SLOT], dtype=np.int16)
    idx += MAX_SYMBOLS_PER_SLOT
    for name in payload_fields:
        meta[name] = vals[idx]
        idx += 1
    meta.setdefault("iq_bytes", 0)
    meta.setdefault("chest_bytes", 0)
    meta.setdefault("llr_bytes", 0)
    meta.pop("reserved0", None)
    meta.pop("reserved1", None)
    # v4: decode raw imsi bytes to string
    if "imsi" in meta and isinstance(meta["imsi"], bytes):
        meta["imsi"] = meta["imsi"].rstrip(b"\x00").decode("ascii", errors="replace")
    return meta


def _empty_complex_grid():
    return np.empty((0, 0), dtype=np.complex64)


def _read_complex_grid(data, offset, payload_bytes, num_symbols, nb_re_per_sym, label):
    if payload_bytes == 0:
        return _empty_complex_grid()
    expected_bytes = num_symbols * nb_re_per_sym * 2 * np.dtype(np.int16).itemsize
    if payload_bytes != expected_bytes:
        raise ValueError(
            f"{label} payload size mismatch: got {payload_bytes}, expected {expected_bytes}"
        )
    raw = np.frombuffer(
        data,
        dtype=np.int16,
        count=payload_bytes // np.dtype(np.int16).itemsize,
        offset=offset,
    ).reshape(num_symbols, nb_re_per_sym, 2).copy()
    return (raw[..., 0] + 1j * raw[..., 1]).astype(np.complex64)


def _read_int16_vector(data, offset, payload_bytes):
    if payload_bytes == 0:
        return np.empty(0, dtype=np.int16)
    return np.frombuffer(
        data,
        dtype=np.int16,
        count=payload_bytes // np.dtype(np.int16).itemsize,
        offset=offset,
    ).copy()


class PUSCHDataset:
    """Reader for PUSCH capture binary datasets."""

    def __init__(self, path):
        self.path = Path(path)
        self._data = self.path.read_bytes()

        hdr = struct.unpack(_FILE_HDR_FMT, self._data[:_FILE_HDR_SIZE])
        magic, version, max_captures, num_captures, _ = hdr
        if magic != PUSCH_FILE_MAGIC:
            raise ValueError(
                f"Bad magic: 0x{magic:08X} (expected 0x{PUSCH_FILE_MAGIC:08X})"
            )
        if version not in (1, 2, 3, 4, 5):
            raise ValueError(f"Unsupported format version: {version}")

        self.version = version
        self.max_captures = max_captures
        self.num_captures = num_captures
        if version == 1:
            self.capture_header_bytes = CAPTURE_HEADER_BYTES_V1
            self._capture_header_fmt = _CAP_HDR_FMT_V1
            self._meta_fields = _META_FIELDS_V1
            self._payload_fields = ("iq_bytes", "chest_bytes", "llr_bytes")
        elif version == 2:
            self.capture_header_bytes = CAPTURE_HEADER_BYTES_V2
            self._capture_header_fmt = _CAP_HDR_FMT_V2
            self._meta_fields = _META_FIELDS_V2
            self._payload_fields = ("iq_bytes", "chest_bytes", "llr_bytes")
        elif version == 3:
            self.capture_header_bytes = CAPTURE_HEADER_BYTES_V3
            self._capture_header_fmt = _CAP_HDR_FMT_V3
            self._meta_fields = _META_FIELDS_V3
            self._payload_fields = ("iq_bytes",)
        elif version == 4:
            self.capture_header_bytes = CAPTURE_HEADER_BYTES_V4
            self._capture_header_fmt = _CAP_HDR_FMT_V4
            self._meta_fields = _META_FIELDS_V4
            self._payload_fields = ("iq_bytes", "imsi")
        else:  # v5
            self.capture_header_bytes = CAPTURE_HEADER_BYTES_V5
            self._capture_header_fmt = _CAP_HDR_FMT_V5
            self._meta_fields = _META_FIELDS_V5
            self._payload_fields = ("iq_bytes", "imsi", "chest_bytes")

        self._offsets = []
        pos = FILE_HEADER_BYTES
        for _ in range(num_captures):
            if pos + self.capture_header_bytes > len(self._data):
                break
            record_bytes = struct.unpack_from("<I", self._data, pos)[0]
            if record_bytes < self.capture_header_bytes:
                break
            self._offsets.append(pos)
            pos += record_bytes
        self.num_captures = len(self._offsets)

    def __len__(self):
        return self.num_captures

    def __repr__(self):
        return (
            f"PUSCHDataset({self.path.name}: "
            f"{self.num_captures}/{self.max_captures} captures, v{self.version})"
        )

    def __getitem__(self, idx):
        if idx < 0:
            idx += self.num_captures
        if idx < 0 or idx >= self.num_captures:
            raise IndexError(f"Capture index {idx} out of range [0, {self.num_captures})")
        return self._read_capture(self._offsets[idx])

    def __iter__(self):
        for i in range(self.num_captures):
            yield self[i]

    def _read_capture(self, offset):
        hdr_buf = self._data[offset:offset + self.capture_header_bytes]
        meta = _parse_capture_header(
            hdr_buf,
            self._capture_header_fmt,
            self._meta_fields,
            self._payload_fields,
        )
        record_end = offset + int(meta["record_bytes"])
        if record_end > len(self._data):
            raise ValueError(
                f"Capture at offset {offset} exceeds file size: {record_end} > {len(self._data)}"
            )

        num_symbols = meta["num_symbols"]
        nb_re_per_sym = meta["nb_re_per_sym"]
        pos = offset + self.capture_header_bytes

        iq = _read_complex_grid(
            self._data,
            pos,
            meta["iq_bytes"],
            num_symbols,
            nb_re_per_sym,
            "IQ",
        )
        pos += meta["iq_bytes"]

        chest = _read_complex_grid(
            self._data,
            pos,
            meta["chest_bytes"],
            num_symbols,
            nb_re_per_sym,
            "ChEst",
        )
        pos += meta["chest_bytes"]

        llr = _read_int16_vector(self._data, pos, meta["llr_bytes"])
        return {"meta": meta, "iq": iq, "chest": chest, "llr": llr}

    def summary(self):
        print(repr(self))
        has_imsi = self.version >= 4
        hdr = f"{'idx':>5} {'frame':>8} {'slot':>4} {'RNTI':>6} {'mod':>4} {'PRBs':>5} {'syms':>4} {'REs':>6}"
        if has_imsi:
            hdr += "  IMSI"
        print(hdr)
        print("-" * (50 + (18 if has_imsi else 0)))
        for i, cap in enumerate(self):
            m = cap["meta"]
            mod_names = {2: "QPSK", 4: "16Q", 6: "64Q", 8: "256Q"}
            mod = mod_names.get(m["qam_mod_order"], f"Q{m['qam_mod_order']}")
            total_re = m["num_symbols"] * m["nb_re_per_sym"]
            row = (
                f"{i:5d} {m['frame']:8d} {m['slot']:4d} {m['rnti']:6d} "
                f"{mod:>4s} {m['rb_size']:5d} {m['num_symbols']:4d} {total_re:6d}"
            )
            if has_imsi:
                row += f"  {m.get('imsi') or '(unlabeled)'}"
            print(row)


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "plugins/nr_pusch_capture/data/pusch_dataset.bin"
    ds = PUSCHDataset(path)
    ds.summary()

    if len(ds) > 0:
        cap = ds[0]
        print("\nFirst capture details:")
        print(f"  IQ shape:       {cap['iq'].shape}")
        print(f"  ChEst shape:    {cap['chest'].shape}  (v1-4: legacy/unused, v5+: gNB UL estimate)")
        print(f"  Legacy LLRs:    {len(cap['llr'])}")
        if cap['iq'].size:
            print(f"  IQ range:       [{cap['iq'].real.min():.0f}, {cap['iq'].real.max():.0f}]")
