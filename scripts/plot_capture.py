#!/usr/bin/env python3
"""
Plot a single PUSCH capture: IQ time-domain, frequency-domain resource grid,
constellation diagram, channel estimate heatmap, LLR distribution, and
power delay profile.

Usage:
    python plot_capture.py [dataset_path] [capture_index]
    python plot_capture.py                                   # defaults: dataset idx 0
    python plot_capture.py path/to/pusch_dataset.bin 42      # specific capture
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(__file__))
from read_dataset import PUSCHDataset

MOD_NAMES = {2: "QPSK", 4: "16-QAM", 6: "64-QAM", 8: "256-QAM"}


def plot_capture(cap, output_path="capture_analysis.png"):
    """Generate a multi-panel analysis figure for a single PUSCH capture."""
    meta = cap["meta"]
    iq    = cap["iq"]       # [num_symbols, nb_re_per_sym] complex64
    chest = cap["chest"]    # [num_symbols, nb_re_per_sym] complex64
    llr   = cap["llr"]      # [total_llrs] int16

    num_symbols   = meta["num_symbols"]
    nb_re_per_sym = meta["nb_re_per_sym"]
    rb_size       = meta["rb_size"]
    mod_order     = meta["qam_mod_order"]
    mod_name      = MOD_NAMES.get(mod_order, f"QAM-{2**mod_order}")
    dmrs_mask     = meta["ul_dmrs_symb_pos"]
    start_symbol  = meta["start_symbol"]

    # Identify DMRS vs data symbols
    is_dmrs = np.array([(dmrs_mask >> (start_symbol + s)) & 1
                        for s in range(num_symbols)], dtype=bool)

    # Equalized symbols: IQ / channel estimate (avoid div-by-zero)
    chest_safe = np.where(np.abs(chest) > 0, chest, 1.0)
    eq = iq / chest_safe

    # ---- Figure layout: 4 rows x 3 cols ----
    fig = plt.figure(figsize=(20, 21), facecolor="white")
    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.30)

    title = (f"PUSCH Capture #{meta['capture_idx']}  —  "
             f"Frame {meta['frame']}, Slot {meta['slot']}, "
             f"RNTI {meta['rnti']}, {mod_name}, "
             f"{rb_size} PRBs, {num_symbols} symbols")
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    # =====================================================================
    # (0,0) IQ time-domain waveform — real & imag across all symbols
    # =====================================================================
    ax_td = fig.add_subplot(gs[0, 0])
    iq_flat = iq.flatten()
    sample_idx = np.arange(len(iq_flat))

    ax_td.plot(sample_idx, iq_flat.real, linewidth=0.5, alpha=0.8,
               color="steelblue", label="I (real)")
    ax_td.plot(sample_idx, iq_flat.imag, linewidth=0.5, alpha=0.8,
               color="coral", label="Q (imag)")

    # Mark symbol boundaries
    for s in range(1, num_symbols):
        ax_td.axvline(s * nb_re_per_sym, color="gray", linewidth=0.3,
                      alpha=0.5)

    ax_td.set_xlabel("Sample index (subcarrier)")
    ax_td.set_ylabel("Amplitude")
    ax_td.set_title("IQ Time-Domain Waveform")
    ax_td.legend(fontsize=8, loc="upper right")
    ax_td.grid(True, alpha=0.2)

    # =====================================================================
    # (0,1) Frequency-domain resource grid — power heatmap
    # =====================================================================
    ax_rg = fig.add_subplot(gs[0, 1])
    power_db = 20 * np.log10(np.maximum(np.abs(iq), 1e-6))

    im = ax_rg.imshow(power_db.T, aspect="auto", origin="lower",
                       cmap="viridis", interpolation="nearest")
    fig.colorbar(im, ax=ax_rg, pad=0.02, label="Power [dB]")

    # Mark DMRS symbols
    for s in range(num_symbols):
        if is_dmrs[s]:
            ax_rg.axvline(s, color="red", linewidth=1.5, alpha=0.6)

    ax_rg.set_xlabel("OFDM Symbol")
    ax_rg.set_ylabel("Subcarrier")
    ax_rg.set_title("Received Resource Grid |Y(k,l)|")
    ax_rg.set_xticks(range(num_symbols))
    ax_rg.set_xticklabels([f"{'D' if is_dmrs[s] else ''}{start_symbol+s}"
                           for s in range(num_symbols)], fontsize=7)

    # =====================================================================
    # (0,2) Constellation diagram — equalized symbols (data only)
    # =====================================================================
    ax_const = fig.add_subplot(gs[0, 2])

    # Separate DMRS and data symbols for plotting
    data_syms = eq[~is_dmrs].flatten()
    dmrs_syms = eq[is_dmrs].flatten() if is_dmrs.any() else np.array([])

    # Normalize for constellation display
    if len(data_syms) > 0:
        rms = np.sqrt(np.mean(np.abs(data_syms)**2))
        if rms > 0:
            data_norm = data_syms / rms
            if len(dmrs_syms) > 0:
                dmrs_norm = dmrs_syms / rms
        else:
            data_norm = data_syms
            dmrs_norm = dmrs_syms
    else:
        data_norm = data_syms
        dmrs_norm = dmrs_syms

    if len(data_norm) > 0:
        ax_const.scatter(data_norm.real, data_norm.imag, s=2, alpha=0.3,
                         c="steelblue", label=f"Data ({len(data_norm)})")
    if len(dmrs_norm) > 0:
        ax_const.scatter(dmrs_norm.real, dmrs_norm.imag, s=6, alpha=0.5,
                         c="red", marker="x", label=f"DMRS ({len(dmrs_norm)})")

    # Draw reference constellation points
    _draw_reference_constellation(ax_const, mod_order)

    ax_const.set_xlabel("In-phase (I)")
    ax_const.set_ylabel("Quadrature (Q)")
    ax_const.set_title(f"Constellation ({mod_name})")
    ax_const.set_aspect("equal")
    ax_const.legend(fontsize=7, loc="upper right")
    ax_const.grid(True, alpha=0.2)
    lim = max(np.abs(data_norm).max() * 1.3, 2.0) if len(data_norm) > 0 else 2.0
    ax_const.set_xlim(-lim, lim)
    ax_const.set_ylim(-lim, lim)

    # =====================================================================
    # (1,0) Channel estimate heatmap — magnitude
    # =====================================================================
    ax_ch = fig.add_subplot(gs[1, 0])
    ch_mag_db = 20 * np.log10(np.maximum(np.abs(chest), 1e-6))

    im2 = ax_ch.imshow(ch_mag_db.T, aspect="auto", origin="lower",
                        cmap="magma", interpolation="nearest")
    fig.colorbar(im2, ax=ax_ch, pad=0.02, label="|H| [dB]")
    ax_ch.set_xlabel("OFDM Symbol")
    ax_ch.set_ylabel("Subcarrier")
    ax_ch.set_title("Channel Estimate |H(k,l)|")
    ax_ch.set_xticks(range(num_symbols))
    ax_ch.set_xticklabels([f"{start_symbol+s}" for s in range(num_symbols)],
                          fontsize=7)

    # =====================================================================
    # (1,1) Channel frequency response — per-symbol overlay
    # =====================================================================
    ax_cfr = fig.add_subplot(gs[1, 1])
    sc_idx = np.arange(nb_re_per_sym)

    for s in range(num_symbols):
        color = "red" if is_dmrs[s] else "steelblue"
        alpha = 0.8 if is_dmrs[s] else 0.25
        lw = 1.2 if is_dmrs[s] else 0.5
        label = "DMRS" if (is_dmrs[s] and s == np.where(is_dmrs)[0][0]) else \
                ("Data" if (not is_dmrs[s] and s == np.where(~is_dmrs)[0][0]) else None)
        ax_cfr.plot(sc_idx, ch_mag_db[s], color=color, alpha=alpha,
                    linewidth=lw, label=label)

    ax_cfr.set_xlabel("Subcarrier index")
    ax_cfr.set_ylabel("|H(k)| [dB]")
    ax_cfr.set_title("Channel Frequency Response")
    ax_cfr.legend(fontsize=8)
    ax_cfr.grid(True, alpha=0.2)

    # =====================================================================
    # (1,2) Channel phase response — per-symbol overlay
    # =====================================================================
    ax_phase = fig.add_subplot(gs[1, 2])

    for s in range(num_symbols):
        color = "red" if is_dmrs[s] else "steelblue"
        alpha = 0.8 if is_dmrs[s] else 0.25
        lw = 1.2 if is_dmrs[s] else 0.5
        ax_phase.plot(sc_idx, np.angle(chest[s], deg=True),
                      color=color, alpha=alpha, linewidth=lw)

    ax_phase.set_xlabel("Subcarrier index")
    ax_phase.set_ylabel("Phase [deg]")
    ax_phase.set_title("Channel Phase Response")
    ax_phase.grid(True, alpha=0.2)

    # =====================================================================
    # (2,0) LLR distribution histogram
    # =====================================================================
    ax_llr = fig.add_subplot(gs[2, 0])

    if len(llr) > 0:
        bins = np.linspace(llr.min() - 0.5, llr.max() + 0.5,
                           min(200, llr.max() - llr.min() + 1))
        ax_llr.hist(llr, bins=bins, color="steelblue", alpha=0.8,
                    edgecolor="none", density=True)
        ax_llr.axvline(0, color="red", linewidth=1, linestyle="--", alpha=0.6)
        ax_llr.set_xlabel("LLR value")
        ax_llr.set_ylabel("Density")

    ax_llr.set_title(f"LLR Distribution ({len(llr)} values)")
    ax_llr.grid(True, alpha=0.2)

    # =====================================================================
    # (2,1) Per-symbol received power
    # =====================================================================
    ax_pow = fig.add_subplot(gs[2, 1])
    sym_power = 10 * np.log10(np.maximum(np.mean(np.abs(iq)**2, axis=1), 1e-12))
    sym_idx = np.arange(num_symbols)
    colors = ["red" if is_dmrs[s] else "steelblue" for s in range(num_symbols)]

    ax_pow.bar(sym_idx, sym_power, color=colors, alpha=0.8, edgecolor="white",
               linewidth=0.5)
    ax_pow.set_xlabel("OFDM Symbol")
    ax_pow.set_ylabel("Avg Power [dB]")
    ax_pow.set_title("Per-Symbol Received Power")
    ax_pow.set_xticks(sym_idx)
    ax_pow.set_xticklabels([f"{start_symbol+s}" for s in range(num_symbols)],
                           fontsize=7)
    ax_pow.grid(True, alpha=0.2, axis="y")

    # Add legend
    from matplotlib.patches import Patch
    ax_pow.legend(handles=[Patch(color="steelblue", label="Data"),
                           Patch(color="red", label="DMRS")],
                  fontsize=8, loc="upper right")

    # =====================================================================
    # (2,2) Metadata info panel
    # =====================================================================
    ax_info = fig.add_subplot(gs[2, 2])
    ax_info.axis("off")

    # Compute EVM on equalized data symbols
    evm_pct = _compute_evm(data_norm, mod_order) if len(data_norm) > 0 else 0.0
    snr_est = _estimate_snr(eq[~is_dmrs].flatten(), mod_order) \
              if (~is_dmrs).any() else 0.0

    dmrs_positions = [start_symbol + s for s in range(num_symbols) if is_dmrs[s]]

    info_lines = [
        ("Capture Index",       f"{meta['capture_idx']}"),
        ("Frame / Slot",        f"{meta['frame']} / {meta['slot']}"),
        ("RNTI",                f"{meta['rnti']}"),
        ("Modulation",          f"{mod_name} (Qm={mod_order})"),
        ("Layers",              f"{meta['num_layers']}"),
        ("PRBs (start, size)",  f"{meta['rb_start']}, {rb_size}"),
        ("BWP Start",           f"{meta['bwp_start']}"),
        ("Symbols",             f"{start_symbol}..{start_symbol+num_symbols-1} "
                                f"({num_symbols} total)"),
        ("DMRS Symbols",        f"{dmrs_positions}"),
        ("Total REs",           f"{num_symbols * nb_re_per_sym}"),
        ("Total LLRs",          f"{len(llr)}"),
        ("FFT Size",            f"{meta['ofdm_symbol_size']}"),
        ("Noise Var (nvar)",    f"{meta['nvar']}"),
        ("EVM (data)",          f"{evm_pct:.1f} %"),
        ("Est. SNR (data)",     f"{snr_est:.1f} dB"),
    ]

    text = "\n".join(f"  {k:<22s}{v}" for k, v in info_lines)
    ax_info.text(0.05, 0.95, text, transform=ax_info.transAxes,
                 fontsize=10, verticalalignment="top",
                 fontfamily="monospace",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                           edgecolor="gray", alpha=0.9))
    ax_info.set_title("Capture Metadata", fontweight="bold")

    # =====================================================================
    # Row 4: DMRS analysis
    # =====================================================================
    dmrs_indices = np.where(is_dmrs)[0]
    n_dmrs = len(dmrs_indices)

    # Noise estimate for MMSE equalization
    noise_est = _estimate_noise_var(iq, chest, is_dmrs, mod_order)

    # (3,0) DMRS received power spectrogram
    ax_dmrs_pow = fig.add_subplot(gs[3, 0])
    if n_dmrs > 0:
        dmrs_power = np.array([
            20 * np.log10(np.maximum(np.abs(iq[s]), 1e-6))
            for s in dmrs_indices
        ])  # [n_dmrs, nb_re_per_sym]
        im3 = ax_dmrs_pow.imshow(dmrs_power.T, aspect="auto", origin="lower",
                                  cmap="viridis", interpolation="nearest")
        fig.colorbar(im3, ax=ax_dmrs_pow, pad=0.02, label="Power [dB]")
        ax_dmrs_pow.set_xticks(range(n_dmrs))
        ax_dmrs_pow.set_xticklabels(
            [f"Sym {start_symbol + s}" for s in dmrs_indices], fontsize=8)
    ax_dmrs_pow.set_xlabel("DMRS Symbol")
    ax_dmrs_pow.set_ylabel("Subcarrier")
    ax_dmrs_pow.set_title(r"DMRS Received Power $|Y_{DMRS}(k)|$ [dB]")

    # (3,1) DMRS constellation — equalized received vs expected reference
    ax_dmrs_const = fig.add_subplot(gs[3, 1])
    if n_dmrs > 0:
        dmrs_iq = iq[is_dmrs].flatten()
        dmrs_ch = chest[is_dmrs].flatten()
        dmrs_ch_safe = np.where(np.abs(dmrs_ch) > 0, dmrs_ch, 1.0)
        eq_dmrs = np.conj(dmrs_ch_safe) * dmrs_iq / (
            np.abs(dmrs_ch_safe)**2 + noise_est)

        # NR DMRS Type 1 uses QPSK-like reference symbols on even subcarriers
        # Approximate: generate unit-magnitude BPSK/QPSK reference points
        # Plot equalized DMRS colored per symbol
        cmap_dmrs = plt.cm.tab10(np.linspace(0, 1, max(n_dmrs, 1)))
        re_per_dmrs = nb_re_per_sym
        for di, s in enumerate(dmrs_indices):
            eq_s = eq_dmrs[di * re_per_dmrs:(di + 1) * re_per_dmrs]
            ax_dmrs_const.scatter(eq_s.real, eq_s.imag, s=10, alpha=0.5,
                                  color=cmap_dmrs[di],
                                  label=f"Sym {start_symbol + s}")
        # Reference QPSK points (DMRS are QPSK-modulated in NR)
        ref = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
        ax_dmrs_const.scatter(ref.real, ref.imag, s=80, c="none",
                              edgecolors="lime", linewidths=1.5, zorder=5,
                              label="Ref QPSK")
        ax_dmrs_const.legend(fontsize=7, loc="upper right")
        ax_dmrs_const.set_aspect("equal")
        lim_d = max(np.abs(eq_dmrs).max() * 1.3, 1.5)
        ax_dmrs_const.set_xlim(-lim_d, lim_d)
        ax_dmrs_const.set_ylim(-lim_d, lim_d)
    ax_dmrs_const.set_xlabel("In-phase (I)")
    ax_dmrs_const.set_ylabel("Quadrature (Q)")
    ax_dmrs_const.set_title("DMRS Constellation (Eq vs Ref)")
    ax_dmrs_const.grid(True, alpha=0.2)

    # (3,2) DMRS phase error per symbol across subcarriers
    ax_dmrs_phase = fig.add_subplot(gs[3, 2])
    if n_dmrs > 0:
        cmap_dmrs = plt.cm.tab10(np.linspace(0, 1, max(n_dmrs, 1)))
        for di, s in enumerate(dmrs_indices):
            eq_s = eq_dmrs[di * re_per_dmrs:(di + 1) * re_per_dmrs]
            # Phase error: angle of equalized DMRS relative to nearest
            # QPSK reference point
            ref = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
            nearest_idx = np.argmin(
                np.abs(eq_s[:, None] - ref[None, :]), axis=1)
            nearest_ref = ref[nearest_idx]
            phase_err = np.angle(eq_s / nearest_ref, deg=True)

            ax_dmrs_phase.plot(np.arange(re_per_dmrs), phase_err,
                               linewidth=0.8, alpha=0.8,
                               color=cmap_dmrs[di],
                               label=f"Sym {start_symbol + s}")
        ax_dmrs_phase.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax_dmrs_phase.legend(fontsize=7, loc="upper right")
    ax_dmrs_phase.set_xlabel("Subcarrier")
    ax_dmrs_phase.set_ylabel("Phase Error [deg]")
    ax_dmrs_phase.set_title("DMRS Phase Error (Eq vs Nearest Ref)")
    ax_dmrs_phase.grid(True, alpha=0.2)

    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output_path}")


def _estimate_noise_var(iq, chest, is_dmrs, mod_order):
    """Estimate noise variance from data symbols via decision-directed method."""
    data_iq = iq[~is_dmrs].flatten()
    data_ch = chest[~is_dmrs].flatten()
    data_ch_safe = np.where(np.abs(data_ch) > 0, data_ch, 1.0)
    eq = np.conj(data_ch_safe) * data_iq / (np.abs(data_ch_safe)**2 + 1e-6)
    if len(eq) == 0:
        return 1e-4
    rms = np.sqrt(np.mean(np.abs(eq)**2))
    if rms == 0:
        return 1e-4
    eq_n = eq / rms
    ref = _get_constellation(mod_order)
    if ref is None:
        return 1e-4
    nearest = ref[np.argmin(np.abs(eq_n[:, None] - ref[None, :]), axis=1)]
    noise = eq_n - nearest
    return float(np.mean(np.abs(noise)**2))


def _get_constellation(mod_order):
    """Return normalized constellation points for a given modulation order."""
    if mod_order == 2:
        return np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    elif mod_order == 4:
        vals = np.array([-3, -1, 1, 3])
        grid = (vals[:, None] + 1j * vals[None, :]).flatten()
        return grid / np.sqrt(np.mean(np.abs(grid)**2))
    elif mod_order == 6:
        vals = np.arange(-7, 8, 2)
        grid = (vals[:, None] + 1j * vals[None, :]).flatten()
        return grid / np.sqrt(np.mean(np.abs(grid)**2))
    elif mod_order == 8:
        vals = np.arange(-15, 16, 2)
        grid = (vals[:, None] + 1j * vals[None, :]).flatten()
        return grid / np.sqrt(np.mean(np.abs(grid)**2))
    return None


def _draw_reference_constellation(ax, mod_order):
    """Overlay ideal constellation grid points (normalized)."""
    if mod_order == 2:  # QPSK
        pts = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    elif mod_order == 4:  # 16-QAM
        vals = np.array([-3, -1, 1, 3])
        grid = (vals[:, None] + 1j * vals[None, :]).flatten()
        pts = grid / np.sqrt(np.mean(np.abs(grid)**2))
    elif mod_order == 6:  # 64-QAM
        vals = np.arange(-7, 8, 2)
        grid = (vals[:, None] + 1j * vals[None, :]).flatten()
        pts = grid / np.sqrt(np.mean(np.abs(grid)**2))
    elif mod_order == 8:  # 256-QAM
        vals = np.arange(-15, 16, 2)
        grid = (vals[:, None] + 1j * vals[None, :]).flatten()
        pts = grid / np.sqrt(np.mean(np.abs(grid)**2))
    else:
        return

    ax.scatter(pts.real, pts.imag, s=30, c="none", edgecolors="lime",
               linewidths=0.8, alpha=0.6, zorder=1, label="Ideal")


def _compute_evm(eq_norm, mod_order):
    """Compute EVM % by snapping to nearest ideal constellation point."""
    if mod_order == 2:
        ref = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    elif mod_order == 4:
        vals = np.array([-3, -1, 1, 3])
        grid = (vals[:, None] + 1j * vals[None, :]).flatten()
        ref = grid / np.sqrt(np.mean(np.abs(grid)**2))
    elif mod_order == 6:
        vals = np.arange(-7, 8, 2)
        grid = (vals[:, None] + 1j * vals[None, :]).flatten()
        ref = grid / np.sqrt(np.mean(np.abs(grid)**2))
    else:
        return 0.0

    # Nearest-neighbor mapping
    dists = np.abs(eq_norm[:, None] - ref[None, :])
    nearest = ref[np.argmin(dists, axis=1)]
    error = eq_norm - nearest
    evm = np.sqrt(np.mean(np.abs(error)**2) / np.mean(np.abs(nearest)**2))
    return evm * 100.0


def _estimate_snr(eq_symbols, mod_order):
    """Rough SNR estimate from equalized symbols via EVM."""
    if len(eq_symbols) == 0:
        return 0.0
    rms = np.sqrt(np.mean(np.abs(eq_symbols)**2))
    if rms == 0:
        return 0.0
    eq_norm = eq_symbols / rms
    evm_pct = _compute_evm(eq_norm, mod_order)
    if evm_pct <= 0:
        return 60.0  # cap
    evm_linear = evm_pct / 100.0
    snr_db = -20 * np.log10(evm_linear)
    return snr_db


if __name__ == "__main__":
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else \
        "plugins/nr_pusch_capture/data/pusch_dataset.bin"
    capture_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    ds = PUSCHDataset(dataset_path)
    print(f"Dataset: {ds}")

    if capture_idx >= len(ds):
        print(f"Error: capture index {capture_idx} out of range [0, {len(ds)})")
        sys.exit(1)

    cap = ds[capture_idx]
    m = cap["meta"]
    print(f"Plotting capture {capture_idx}: frame={m['frame']}, "
          f"slot={m['slot']}, RNTI={m['rnti']}, "
          f"{MOD_NAMES.get(m['qam_mod_order'], '?')}, "
          f"{m['rb_size']} PRBs, {m['num_symbols']} symbols")

    out_dir = os.path.dirname(dataset_path) or "."
    out_path = os.path.join(out_dir, f"capture_{capture_idx:04d}.png")
    plot_capture(cap, out_path)
