#!/usr/bin/env python3
"""
Plot a single IQ-only PUSCH capture with emphasis on DMRS comb visibility.

Usage:
    python plot_capture.py [dataset_path] [capture_index] [output_path]
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from evaluate_dataset import dmrs_comb_presence_stats, dmrs_mod12_profile, dmrs_symbol_indices
from read_dataset import PUSCHDataset

DEFAULT_DATASET = "plugins/nr_pusch_capture/data/pusch_dataset.bin"
MOD_NAMES = {2: "QPSK", 4: "16-QAM", 6: "64-QAM", 8: "256-QAM"}


def _dmrs_symbol_mask(meta):
    num_symbols = int(meta["num_symbols"])
    start_symbol = int(meta["start_symbol"])
    dmrs_mask = int(meta["ul_dmrs_symb_pos"])
    return np.array(
        [((dmrs_mask >> (start_symbol + s)) & 0x1) == 1 for s in range(num_symbols)],
        dtype=bool,
    )


def _format_bins(values):
    if not values:
        return "-"
    return ", ".join(str(int(v)) for v in values)


def plot_capture(cap, output_path="capture_analysis.png", verbose=True):
    meta = cap["meta"]
    iq = cap["iq"]
    num_symbols = int(meta["num_symbols"])
    nb_re_per_sym = int(meta["nb_re_per_sym"])
    start_symbol = int(meta["start_symbol"])
    mod_name = MOD_NAMES.get(int(meta["qam_mod_order"]), f"Qm={meta['qam_mod_order']}")
    is_dmrs = _dmrs_symbol_mask(meta)
    dmrs_symbols = dmrs_symbol_indices(meta)
    dmrs_profile = dmrs_mod12_profile(cap)
    comb_stats = dmrs_comb_presence_stats(cap)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    fig.suptitle(
        (
            f"PUSCH Capture #{meta['capture_idx']} | Frame {meta['frame']} Slot {meta['slot']} | "
            f"RNTI {meta['rnti']} | {mod_name} | {meta['rb_size']} PRBs"
        ),
        fontsize=14,
        fontweight="bold",
    )

    ax_td = axes[0, 0]
    iq_flat = iq.reshape(-1)
    sample_idx = np.arange(iq_flat.size)
    ax_td.plot(sample_idx, iq_flat.real, color="steelblue", linewidth=0.7, label="I")
    ax_td.plot(sample_idx, iq_flat.imag, color="coral", linewidth=0.7, label="Q")
    for s in range(1, num_symbols):
        ax_td.axvline(s * nb_re_per_sym, color="gray", linewidth=0.5, alpha=0.5)
    ax_td.set_title("Flattened IQ Waveform")
    ax_td.set_xlabel("Sample index")
    ax_td.set_ylabel("Amplitude")
    ax_td.grid(True, alpha=0.2)
    ax_td.legend(loc="upper right", fontsize=8)

    ax_rg = axes[0, 1]
    power_db = 10.0 * np.log10(np.maximum(np.abs(iq) ** 2, 1e-12))
    image = ax_rg.imshow(power_db.T, aspect="auto", origin="lower", cmap="viridis")
    fig.colorbar(image, ax=ax_rg, pad=0.02, label="Power [dB]")
    for rel_symbol in np.where(is_dmrs)[0]:
        ax_rg.axvline(rel_symbol, color="red", linewidth=1.2, alpha=0.7)
    ax_rg.set_title("Resource Grid Power")
    ax_rg.set_xlabel("Relative OFDM symbol")
    ax_rg.set_ylabel("Allocated subcarrier")
    ax_rg.set_xticks(range(num_symbols))
    ax_rg.set_xticklabels([str(start_symbol + s) for s in range(num_symbols)], fontsize=8)

    ax_comb = axes[1, 0]
    if dmrs_profile is None:
        ax_comb.text(0.5, 0.5, "No DMRS profile available", ha="center", va="center")
        ax_comb.set_axis_off()
    else:
        bins = np.arange(len(dmrs_profile))
        colors = ["lightgray"] * len(dmrs_profile)
        active_bins = []
        quiet_bins = []
        if comb_stats is not None:
            active_bins = comb_stats.get("active_bins") or []
            quiet_bins = comb_stats.get("quiet_bins") or []
        for idx in quiet_bins:
            colors[idx] = "silver"
        for idx in active_bins:
            colors[idx] = "tab:red"
        ax_comb.bar(bins, dmrs_profile, color=colors, edgecolor="black", linewidth=0.4)
        ax_comb.set_xticks(bins)
        ax_comb.set_xlabel("RE index mod 12")
        ax_comb.set_ylabel("Normalized DMRS power")
        if comb_stats is None:
            comb_title = "DMRS Comb Profile"
        elif not comb_stats["supported"]:
            comb_title = "DMRS Comb Profile (unsupported layout)"
        else:
            ratio = comb_stats["power_ratio"]
            ratio_text = "inf" if ratio == float("inf") else f"{ratio:.2f}"
            verdict = "visible" if comb_stats["present"] else "weak"
            comb_title = f"DMRS Comb Profile ({verdict}, ratio={ratio_text})"
        ax_comb.set_title(comb_title)
        ax_comb.grid(True, axis="y", alpha=0.2)

    ax_pow = axes[1, 1]
    symbol_power_db = 10.0 * np.log10(np.maximum(np.mean(np.abs(iq) ** 2, axis=1), 1e-12))
    rel_symbols = np.arange(num_symbols)
    colors = ["tab:red" if is_dmrs[s] else "steelblue" for s in rel_symbols]
    ax_pow.bar(rel_symbols, symbol_power_db, color=colors, alpha=0.85)
    ax_pow.set_title("Per-Symbol Power")
    ax_pow.set_xlabel("Relative OFDM symbol")
    ax_pow.set_ylabel("Average power [dB]")
    ax_pow.set_xticks(rel_symbols)
    ax_pow.set_xticklabels([str(start_symbol + s) for s in rel_symbols], fontsize=8)
    ax_pow.grid(True, axis="y", alpha=0.2)

    info_lines = [
        f"DMRS symbols: {_format_bins(dmrs_symbols)}",
        f"DMRS type: {meta.get('dmrs_config_type', '-')}",
        f"CDM groups: {meta.get('num_dmrs_cdm_grps_no_data', '-')}",
        f"DMRS ports: 0x{int(meta.get('dmrs_ports', 0)):x}" if "dmrs_ports" in meta else "DMRS ports: -",
        f"Transform precoding: {meta.get('transform_precoding', '-')}",
        f"Active mod12 bins: {_format_bins((comb_stats or {}).get('active_bins', []))}",
        f"Quiet mod12 bins: {_format_bins((comb_stats or {}).get('quiet_bins', []))}",
    ]
    if comb_stats is not None and comb_stats.get("power_ratio") is not None:
        ratio = comb_stats["power_ratio"]
        ratio_text = "inf" if ratio == float("inf") else f"{ratio:.2f}"
        info_lines.append(f"Comb ratio: {ratio_text}")
    ax_pow.text(
        1.02,
        0.98,
        "\n".join(info_lines),
        transform=ax_pow.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9},
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    if verbose:
        print(f"Saved {output_path}")


if __name__ == "__main__":
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATASET
    capture_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    output_path = sys.argv[3] if len(sys.argv) > 3 else f"capture_{capture_index:04d}.png"

    dataset = PUSCHDataset(dataset_path)
    plot_capture(dataset[capture_index], output_path)
