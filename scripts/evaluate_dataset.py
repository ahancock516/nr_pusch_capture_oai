#!/usr/bin/env python3
"""
Evaluate a PUSCH capture dataset for three issues:
  1. Captures with missing DMRS in the recorded allocation window.
  2. Exact duplicate signal payloads across captures.
  3. Whether later captures preserve the DMRS comb pattern seen in a
     reference capture range.
"""

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from read_dataset import PUSCHDataset

DEFAULT_DATASET = "plugins/nr_pusch_capture/data/pusch_dataset.bin"
MOD12_SIZE = 12
DMRS_COMB_ACTIVE_RE = 6
DMRS_COMB_POWER_RATIO_THRESHOLD = 1.50
NFAPI_NR_DMRS_TYPE1 = 0
NFAPI_NR_DMRS_TYPE2 = 1
TYPE1_PORT_DELTAS = [0, 0, 1, 1, 0, 0, 1, 1]
TYPE2_PORT_DELTAS = [0, 0, 2, 2, 4, 4, 0, 0, 2, 2, 4, 4]
TYPE2_GROUP_BINS = ([0, 1, 6, 7], [2, 3, 8, 9], [4, 5, 10, 11])


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate DMRS presence, duplicate captures, and reference DMRS "
            "pattern consistency in a PUSCH dataset."
        )
    )
    parser.add_argument("dataset_path", nargs="?", default=DEFAULT_DATASET,
                        help=f"Path to pusch_dataset.bin (default: {DEFAULT_DATASET})")
    parser.add_argument("--json", dest="json_path",
                        help="Optional path to write the evaluation result as JSON.")
    parser.add_argument("--reference-start", type=int,
                        help="Reference capture start index (inclusive).")
    parser.add_argument("--reference-end", type=int,
                        help="Reference capture end index (inclusive).")
    parser.add_argument("--one-based", action="store_true",
                        help="Interpret reference indices as one-based instead of zero-based.")
    return parser.parse_args(argv)


def dmrs_symbol_indices(meta):
    start_symbol = int(meta["start_symbol"])
    num_symbols = int(meta["num_symbols"])
    dmrs_mask = int(meta["ul_dmrs_symb_pos"])
    return [
        start_symbol + rel_symbol
        for rel_symbol in range(num_symbols)
        if (dmrs_mask >> (start_symbol + rel_symbol)) & 0x1
    ]


def average_power(values):
    if values.size == 0:
        return None
    return float(np.mean(np.abs(values) ** 2))


def summarize_values(values):
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {"min": float(np.min(arr)), "mean": float(np.mean(arr)), "max": float(np.max(arr))}


def dmrs_power_stats(cap):
    meta = cap["meta"]
    iq = cap["iq"]
    start_symbol = int(meta["start_symbol"])
    num_symbols = int(meta["num_symbols"])
    dmrs_mask = int(meta["ul_dmrs_symb_pos"])

    dmrs_rel = [rel_symbol for rel_symbol in range(num_symbols)
                if (dmrs_mask >> (start_symbol + rel_symbol)) & 0x1]
    data_rel = [rel_symbol for rel_symbol in range(num_symbols)
                if not ((dmrs_mask >> (start_symbol + rel_symbol)) & 0x1)]

    dmrs_power = average_power(iq[dmrs_rel, :]) if dmrs_rel else None
    data_power = average_power(iq[data_rel, :]) if data_rel else None

    return {
        "dmrs_symbols": [start_symbol + rel_symbol for rel_symbol in dmrs_rel],
        "avg_dmrs_power": dmrs_power,
        "avg_data_power": data_power,
    }


def dmrs_mod12_profile(cap):
    meta = cap["meta"]
    iq = cap["iq"]
    start_symbol = int(meta["start_symbol"])
    dmrs_symbols = dmrs_symbol_indices(meta)
    if not dmrs_symbols:
        return None

    dmrs_rel = [symbol - start_symbol for symbol in dmrs_symbols]
    dmrs_iq = iq[dmrs_rel, :]
    if dmrs_iq.size == 0:
        return None

    power = np.abs(dmrs_iq) ** 2
    profile = np.zeros(MOD12_SIZE, dtype=np.float64)
    for offset in range(MOD12_SIZE):
        offset_samples = power[:, offset::MOD12_SIZE]
        if offset_samples.size:
            profile[offset] = float(np.mean(offset_samples))

    mean_power = float(np.mean(profile))
    if mean_power == 0.0:
        return None
    return [float(value) for value in (profile / mean_power)]


def select_active_bins(profile, count=DMRS_COMB_ACTIVE_RE):
    arr = np.asarray(profile, dtype=np.float64)
    order = np.argsort(-arr, kind="stable")[:count]
    return sorted(int(index) for index in order)


def profile_contrast(profile, active_bins):
    arr = np.asarray(profile, dtype=np.float64)
    active_set = set(active_bins)
    inactive_bins = [index for index in range(arr.size) if index not in active_set]
    return float(np.mean(arr[active_bins]) - np.mean(arr[inactive_bins]))


def profile_correlation(profile, reference_profile):
    arr = np.asarray(profile, dtype=np.float64)
    ref = np.asarray(reference_profile, dtype=np.float64)
    if arr.size != ref.size:
        raise ValueError("Profiles must have the same length")
    if np.std(arr) == 0.0 or np.std(ref) == 0.0:
        return None
    return float(np.corrcoef(arr, ref)[0, 1])


def has_dmrs_config(meta):
    return all(name in meta for name in ("dmrs_config_type", "num_dmrs_cdm_grps_no_data", "dmrs_ports"))


def dmrs_reserved_mod12(meta):
    if not has_dmrs_config(meta):
        return None
    config_type = int(meta["dmrs_config_type"])
    num_groups = int(meta["num_dmrs_cdm_grps_no_data"])
    reserved = set()
    if config_type == NFAPI_NR_DMRS_TYPE1:
        if num_groups not in (1, 2):
            return None
        for group in range(num_groups):
            reserved.update(range(group, MOD12_SIZE, 2))
    elif config_type == NFAPI_NR_DMRS_TYPE2:
        if num_groups not in (1, 2, 3):
            return None
        for group in range(num_groups):
            reserved.update(TYPE2_GROUP_BINS[group])
    else:
        return None
    return sorted(reserved)


def dmrs_active_mod12(meta):
    if not has_dmrs_config(meta):
        return None
    config_type = int(meta["dmrs_config_type"])
    dmrs_ports = int(meta["dmrs_ports"])
    if dmrs_ports == 0:
        return None

    deltas = TYPE1_PORT_DELTAS if config_type == NFAPI_NR_DMRS_TYPE1 else TYPE2_PORT_DELTAS
    active = set()
    for port, delta in enumerate(deltas):
        if not (dmrs_ports & (1 << port)):
            continue
        if config_type == NFAPI_NR_DMRS_TYPE1:
            active.update(range(delta, MOD12_SIZE, 2))
        elif config_type == NFAPI_NR_DMRS_TYPE2:
            active.update(delta + offset for offset in (0, 1, 6, 7))
        else:
            return None
    active = {bin_index for bin_index in active if 0 <= bin_index < MOD12_SIZE}
    reserved = dmrs_reserved_mod12(meta)
    if reserved is None or not active:
        return None
    if any(bin_index not in reserved for bin_index in active):
        return None
    return sorted(active)


def dmrs_quiet_mod12(meta):
    reserved = dmrs_reserved_mod12(meta)
    active = dmrs_active_mod12(meta)
    if reserved is None or active is None:
        return None
    active_set = set(active)
    return [bin_index for bin_index in reserved if bin_index not in active_set]


def dmrs_comb_presence_stats(cap):
    profile = dmrs_mod12_profile(cap)
    if profile is None:
        return None
    meta = cap["meta"]
    active_bins = dmrs_active_mod12(meta)
    quiet_bins = dmrs_quiet_mod12(meta)
    if active_bins is None:
        return None
    profile_arr = np.asarray(profile, dtype=np.float64)
    active_mean = float(np.mean(profile_arr[active_bins]))
    if not quiet_bins:
        return {
            "supported": False,
            "present": None,
            "active_bins": active_bins,
            "quiet_bins": [],
            "power_ratio": None,
            "active_mean": active_mean,
            "quiet_mean": None,
        }
    quiet_mean = float(np.mean(profile_arr[quiet_bins]))
    power_ratio = float("inf") if quiet_mean == 0.0 else active_mean / quiet_mean
    return {
        "supported": True,
        "present": power_ratio >= DMRS_COMB_POWER_RATIO_THRESHOLD,
        "active_bins": active_bins,
        "quiet_bins": quiet_bins,
        "power_ratio": power_ratio,
        "active_mean": active_mean,
        "quiet_mean": quiet_mean,
    }


def payload_hash(dataset, idx, record_bytes):
    offset = dataset._offsets[idx]
    start = offset + dataset.capture_header_bytes
    end = offset + int(record_bytes)
    payload = dataset._data[start:end]
    return hashlib.sha256(payload).hexdigest()


def summarize_capture(idx, cap):
    meta = cap["meta"]
    dmrs_stats = dmrs_power_stats(cap)
    summary = {
        "index": idx,
        "capture_idx": int(meta["capture_idx"]),
        "frame": int(meta["frame"]),
        "slot": int(meta["slot"]),
        "rnti": int(meta["rnti"]),
        "start_symbol": int(meta["start_symbol"]),
        "num_symbols": int(meta["num_symbols"]),
        "rb_start": int(meta["rb_start"]),
        "rb_size": int(meta["rb_size"]),
        "qam_mod_order": int(meta["qam_mod_order"]),
        "ul_dmrs_symb_pos": int(meta["ul_dmrs_symb_pos"]),
        "dmrs_symbols": dmrs_symbol_indices(meta),
        "avg_dmrs_power": dmrs_stats["avg_dmrs_power"],
        "avg_data_power": dmrs_stats["avg_data_power"],
    }
    for optional_name in ("transform_precoding", "dmrs_config_type", "num_dmrs_cdm_grps_no_data", "dmrs_ports"):
        if optional_name in meta:
            summary[optional_name] = int(meta[optional_name])
    return summary


def format_index_list(indices, limit=24):
    if len(indices) <= limit:
        return str(indices)
    head = indices[: limit // 2]
    tail = indices[-(limit // 2):]
    return f"{head} ... {tail}"


def format_profile(profile):
    return "[" + ", ".join(f"{value:.3f}" for value in profile) + "]"


def normalize_reference_range(start, end, capture_count, one_based=False):
    if start is None or end is None:
        return None
    if one_based:
        start -= 1
        end -= 1
    if start < 0 or end < 0 or start > end or end >= capture_count:
        raise ValueError(f"Invalid reference range: start={start}, end={end}, captures={capture_count}")
    return list(range(start, end + 1))


def analyze_reference_pattern(pattern_profiles, reference_indices):
    reference_set = set(reference_indices)
    reference_profiles = [pattern_profiles[index] for index in reference_indices if pattern_profiles[index] is not None]
    if not reference_profiles:
        return None

    reference_mean_profile = np.mean(np.asarray(reference_profiles, dtype=np.float64), axis=0)
    expected_active_mod12 = select_active_bins(reference_mean_profile)
    expected_inactive_mod12 = [index for index in range(MOD12_SIZE) if index not in expected_active_mod12]
    reference_comb_strengths = [profile_contrast(profile, expected_active_mod12) for profile in reference_profiles]
    visible_comb_threshold = 0.5 * float(np.mean(reference_comb_strengths))

    candidate_exact_pattern = []
    candidate_complement_pattern = []
    candidate_weak_pattern = []
    candidate_other_pattern = []
    candidate_correlations = []

    for index, profile in enumerate(pattern_profiles):
        if index in reference_set:
            continue
        if profile is None:
            candidate_weak_pattern.append(index)
            continue
        detected_active_mod12 = select_active_bins(profile)
        detected_comb_strength = profile_contrast(profile, detected_active_mod12)
        correlation = profile_correlation(profile, reference_mean_profile)
        if correlation is not None:
            candidate_correlations.append(correlation)
        if detected_comb_strength < visible_comb_threshold:
            candidate_weak_pattern.append(index)
        elif detected_active_mod12 == expected_active_mod12:
            candidate_exact_pattern.append(index)
        elif detected_active_mod12 == expected_inactive_mod12:
            candidate_complement_pattern.append(index)
        else:
            candidate_other_pattern.append(index)

    candidate_count = len(pattern_profiles) - len(reference_indices)
    return {
        "reference_mean_profile": [float(value) for value in reference_mean_profile],
        "expected_active_mod12": expected_active_mod12,
        "expected_inactive_mod12": expected_inactive_mod12,
        "reference_comb_strength_summary": summarize_values(reference_comb_strengths),
        "visible_comb_threshold": visible_comb_threshold,
        "candidate_exact_pattern_count": len(candidate_exact_pattern),
        "candidate_exact_pattern_indices": candidate_exact_pattern,
        "candidate_complement_pattern_count": len(candidate_complement_pattern),
        "candidate_complement_pattern_indices": candidate_complement_pattern,
        "candidate_weak_pattern_count": len(candidate_weak_pattern),
        "candidate_weak_pattern_indices": candidate_weak_pattern,
        "candidate_other_pattern_count": len(candidate_other_pattern),
        "candidate_other_pattern_indices": candidate_other_pattern,
        "candidate_reference_correlation_summary": summarize_values(candidate_correlations),
        "verdict_expected_pattern_present": len(candidate_exact_pattern) == candidate_count,
    }


def analyze_reference_group(summaries, payload_hashes, reference_indices, one_based=False, pattern_profiles=None):
    reference_set = set(reference_indices)
    reference = [summaries[i] for i in reference_indices]
    candidate = [item for item in summaries if item["index"] not in reference_set]

    reference_patterns = Counter(tuple(item["dmrs_symbols"]) for item in reference)
    expected_dmrs_symbols = list(reference_patterns.most_common(1)[0][0]) if reference_patterns else []
    candidate_missing_dmrs = [item["index"] for item in candidate if not item["dmrs_symbols"]]
    candidate_zero_dmrs_power = [item["index"] for item in candidate if item["avg_dmrs_power"] in (None, 0.0)]
    candidate_symbol_mismatches = [item["index"] for item in candidate if item["dmrs_symbols"] != expected_dmrs_symbols]
    reference_hashes = {payload_hashes[i] for i in reference_indices}
    candidate_duplicates_of_reference = [item["index"] for item in candidate if payload_hashes[item["index"]] in reference_hashes]

    result = {
        "reference_range": {
            "start_index": reference_indices[0],
            "end_index": reference_indices[-1],
            "display_start": reference_indices[0] + 1 if one_based else reference_indices[0],
            "display_end": reference_indices[-1] + 1 if one_based else reference_indices[-1],
            "one_based": one_based,
        },
        "reference_count": len(reference),
        "candidate_count": len(candidate),
        "expected_dmrs_symbols": expected_dmrs_symbols,
        "reference_dmrs_patterns": {str(list(pattern)): count for pattern, count in reference_patterns.items()},
        "reference_dmrs_power_summary": summarize_values([item["avg_dmrs_power"] for item in reference if item["avg_dmrs_power"] is not None]),
        "candidate_dmrs_power_summary": summarize_values([item["avg_dmrs_power"] for item in candidate if item["avg_dmrs_power"] is not None]),
        "reference_data_power_summary": summarize_values([item["avg_data_power"] for item in reference if item["avg_data_power"] is not None]),
        "candidate_data_power_summary": summarize_values([item["avg_data_power"] for item in candidate if item["avg_data_power"] is not None]),
        "candidate_missing_dmrs_count": len(candidate_missing_dmrs),
        "candidate_missing_dmrs_indices": candidate_missing_dmrs,
        "candidate_zero_dmrs_power_count": len(candidate_zero_dmrs_power),
        "candidate_zero_dmrs_power_indices": candidate_zero_dmrs_power,
        "candidate_symbol_mismatch_count": len(candidate_symbol_mismatches),
        "candidate_symbol_mismatch_indices": candidate_symbol_mismatches,
        "candidate_duplicates_of_reference_count": len(candidate_duplicates_of_reference),
        "candidate_duplicates_of_reference_indices": candidate_duplicates_of_reference,
        "verdict_dmrs_present": len(candidate_missing_dmrs) == 0 and len(candidate_zero_dmrs_power) == 0 and len(candidate_symbol_mismatches) == 0,
    }
    if pattern_profiles is not None:
        result["pattern_analysis"] = analyze_reference_pattern(pattern_profiles, reference_indices)
    return result


def analyze_dataset(dataset, reference_indices=None, one_based=False):
    missing_dmrs = []
    zero_dmrs_power = []
    payload_groups = defaultdict(list)
    dmrs_powers = []
    summaries = []
    payload_hashes = []
    pattern_profiles = []
    comb_mismatches = []
    comb_supported_count = 0

    for idx, cap in enumerate(dataset):
        meta = cap["meta"]
        summary = summarize_capture(idx, cap)
        summaries.append(summary)
        pattern_profiles.append(dmrs_mod12_profile(cap))

        comb_stats = dmrs_comb_presence_stats(cap)
        if comb_stats is not None:
            summary["dmrs_expected_active_mod12"] = comb_stats["active_bins"]
            summary["dmrs_expected_quiet_mod12"] = comb_stats["quiet_bins"]
            summary["dmrs_comb_supported"] = comb_stats["supported"]
            summary["dmrs_comb_present"] = comb_stats["present"]
            summary["dmrs_comb_power_ratio"] = comb_stats["power_ratio"]
            if comb_stats["supported"]:
                comb_supported_count += 1
                if not comb_stats["present"]:
                    comb_mismatches.append(summary)

        if not summary["dmrs_symbols"]:
            missing_dmrs.append(summary)
        elif summary["avg_dmrs_power"] in (None, 0.0):
            zero_dmrs_power.append(summary)
        else:
            dmrs_powers.append(summary["avg_dmrs_power"])

        h = payload_hash(dataset, idx, meta["record_bytes"])
        payload_hashes.append(h)
        payload_groups[h].append(summary)

    duplicate_payload_groups = [{
        "hash": payload_hash_value,
        "indices": [entry["index"] for entry in entries],
        "count": len(entries),
        "example": entries[0],
    } for payload_hash_value, entries in payload_groups.items() if len(entries) > 1]
    duplicate_payload_groups.sort(key=lambda item: (-item["count"], item["indices"][0]))

    consecutive_duplicate_pairs = []
    for group in duplicate_payload_groups:
        for left, right in zip(group["indices"], group["indices"][1:]):
            if right == left + 1:
                consecutive_duplicate_pairs.append([left, right])

    result = {
        "dataset": str(dataset.path),
        "capture_count": len(dataset),
        "dmrs_missing_count": len(missing_dmrs),
        "dmrs_missing": missing_dmrs,
        "dmrs_zero_power_count": len(zero_dmrs_power),
        "dmrs_zero_power": zero_dmrs_power,
        "dmrs_power_summary": summarize_values(dmrs_powers),
        "dmrs_comb_check_supported_count": comb_supported_count,
        "dmrs_comb_mismatch_count": len(comb_mismatches),
        "dmrs_comb_mismatches": comb_mismatches,
        "duplicate_payload_group_count": len(duplicate_payload_groups),
        "duplicate_payload_groups": duplicate_payload_groups,
        "consecutive_duplicate_pairs": consecutive_duplicate_pairs,
    }
    if reference_indices is not None:
        result["reference_analysis"] = analyze_reference_group(
            summaries, payload_hashes, reference_indices, one_based=one_based, pattern_profiles=pattern_profiles
        )
    return result


def print_power_summary(label, summary):
    if not summary:
        return
    print(f"  {label}: min={summary['min']:.3f} mean={summary['mean']:.3f} max={summary['max']:.3f}")


def print_report(result):
    print(f"Dataset: {result['dataset']}")
    print(f"Captures: {result['capture_count']}")
    print()
    print("DMRS checks")
    print(f"  Missing DMRS mask in capture window: {result['dmrs_missing_count']}")
    if result["dmrs_missing"]:
        for item in result["dmrs_missing"]:
            print("    " +
                  f"idx={item['index']} frame={item['frame']} slot={item['slot']} "
                  f"symbols={item['start_symbol']}..{item['start_symbol'] + item['num_symbols'] - 1} "
                  f"mask=0x{item['ul_dmrs_symb_pos']:x}")
    print(f"  Zero DMRS power on DMRS-labeled symbols: {result['dmrs_zero_power_count']}")
    if result["dmrs_zero_power"]:
        for item in result["dmrs_zero_power"]:
            print("    " +
                  f"idx={item['index']} frame={item['frame']} slot={item['slot']} "
                  f"dmrs_symbols={item['dmrs_symbols']} avg_dmrs_power={item['avg_dmrs_power']}")
    print_power_summary("DMRS power summary", result["dmrs_power_summary"])
    print(f"  Config-aware DMRS comb checks supported: {result['dmrs_comb_check_supported_count']}")
    print(f"  Config-aware DMRS comb mismatches: {result['dmrs_comb_mismatch_count']}")
    if result["dmrs_comb_mismatches"]:
        for item in result["dmrs_comb_mismatches"][:10]:
            ratio = item.get("dmrs_comb_power_ratio")
            ratio_text = "inf" if ratio == float("inf") else f"{ratio:.3f}"
            print("    " +
                  f"idx={item['index']} type={item.get('dmrs_config_type')} "
                  f"ports=0x{item.get('dmrs_ports', 0):x} "
                  f"active={item.get('dmrs_expected_active_mod12')} "
                  f"quiet={item.get('dmrs_expected_quiet_mod12')} ratio={ratio_text}")

    print()
    print("Duplicate checks")
    print(f"  Duplicate payload groups: {result['duplicate_payload_group_count']}")
    if result["duplicate_payload_groups"]:
        for group in result["duplicate_payload_groups"]:
            example = group["example"]
            print("    " +
                  f"count={group['count']} indices={format_index_list(group['indices'])} "
                  f"frame={example['frame']} slot={example['slot']} rnti={example['rnti']}")
    print("  Consecutive duplicate pairs: " + format_index_list(result["consecutive_duplicate_pairs"], limit=12))

    reference = result.get("reference_analysis")
    if reference:
        print()
        print("Reference comparison")
        ref_range = reference["reference_range"]
        print("  Reference captures: " +
              f"{ref_range['display_start']}..{ref_range['display_end']} "
              f"({'one-based' if ref_range['one_based'] else 'zero-based'})")
        print(f"  Reference count: {reference['reference_count']}")
        print(f"  Candidate count: {reference['candidate_count']}")
        print(f"  Expected DMRS symbols from reference: {reference['expected_dmrs_symbols']}")
        print(f"  Candidate missing DMRS: {reference['candidate_missing_dmrs_count']}")
        print(f"  Candidate zero DMRS power: {reference['candidate_zero_dmrs_power_count']}")
        print(f"  Candidate DMRS symbol mismatches: {reference['candidate_symbol_mismatch_count']}")
        print(f"  Candidate duplicates of reference payloads: {reference['candidate_duplicates_of_reference_count']}")
        print_power_summary("Reference DMRS power", reference['reference_dmrs_power_summary'])
        print_power_summary("Candidate DMRS power", reference['candidate_dmrs_power_summary'])
        print_power_summary("Reference data power", reference['reference_data_power_summary'])
        print_power_summary("Candidate data power", reference['candidate_data_power_summary'])
        print(f"  Verdict: DMRS present in all candidate captures = {reference['verdict_dmrs_present']}")
        pattern = reference.get('pattern_analysis')
        if pattern:
            print()
            print("Reference pattern comparison")
            print(f"  Reference mean mod-12 profile: {format_profile(pattern['reference_mean_profile'])}")
            print(f"  Expected active offsets mod 12: {pattern['expected_active_mod12']}")
            print(f"  Expected inactive offsets mod 12: {pattern['expected_inactive_mod12']}")
            print(f"  Visible comb threshold: {pattern['visible_comb_threshold']:.3f}")
            print(f"  Candidate exact reference-pattern matches: {pattern['candidate_exact_pattern_count']}")
            if pattern['candidate_exact_pattern_indices']:
                print("    indices=" + format_index_list(pattern['candidate_exact_pattern_indices']))
            print(f"  Candidate complementary-pattern matches: {pattern['candidate_complement_pattern_count']}")
            if pattern['candidate_complement_pattern_indices']:
                print("    indices=" + format_index_list(pattern['candidate_complement_pattern_indices']))
            print(f"  Candidate weak/no visible comb: {pattern['candidate_weak_pattern_count']}")
            if pattern['candidate_weak_pattern_indices']:
                print("    indices=" + format_index_list(pattern['candidate_weak_pattern_indices']))
            print(f"  Candidate other visible-pattern mismatches: {pattern['candidate_other_pattern_count']}")
            if pattern['candidate_other_pattern_indices']:
                print("    indices=" + format_index_list(pattern['candidate_other_pattern_indices']))
            print_power_summary("Reference comb strength", pattern['reference_comb_strength_summary'])
            print_power_summary("Candidate/reference correlation", pattern['candidate_reference_correlation_summary'])
            print("  Verdict: expected reference pattern present in all candidate captures = " +
                  f"{pattern['verdict_expected_pattern_present']}")


def main(argv=None):
    args = parse_args(argv)
    dataset = PUSCHDataset(args.dataset_path)
    reference_indices = normalize_reference_range(args.reference_start, args.reference_end, len(dataset), one_based=args.one_based)
    result = analyze_dataset(dataset, reference_indices=reference_indices, one_based=args.one_based)
    print_report(result)
    if args.json_path:
        with open(args.json_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote JSON report to {args.json_path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
