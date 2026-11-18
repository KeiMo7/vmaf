"""
Shard merging and result aggregation.

Merge strategy
--------------
Each shard JSON contains per-frame metric values for its owned frame range
only (warm-up/look-ahead context frames were filtered out by the CLI's emit
window). Merging is therefore a pure concatenation ordered by frameNum, plus
a recomputation of pooled statistics over the full frame range.

Pooled statistics replicate libvmaf's `vmaf_feature_score_pooled` exactly
(see libvmaf/src/libvmaf.c): mean, min, max and harmonic mean
`n / sum(1/(s+1)) - 1`, all in float64, honoring the same subsample rule
(only frames where `frameNum % subsample == 0`). No weighting or shard-level
partial pooling is involved, which is what makes sharded and whole-file
results identical.
"""

import json
from collections import OrderedDict

from .state import atomic_write_json, read_json

POOL_METHODS = ("min", "max", "mean", "harmonic_mean")


def _pool(values):
    """Same formulas and float64 accumulation order as libvmaf."""
    n = len(values)
    if n == 0:
        return {name: None for name in POOL_METHODS}
    total = 0.0
    inv_sum = 0.0
    minimum = values[0]
    maximum = values[0]
    for s in values:
        total += s
        inv_sum += 1.0 / (s + 1.0)
        if s < minimum:
            minimum = s
        if s > maximum:
            maximum = s
    return {
        "min": minimum,
        "max": maximum,
        "mean": total / n,
        "harmonic_mean": n / inv_sum - 1.0,
    }


def load_shard_frames(path):
    doc = read_json(path, default=None)
    if not isinstance(doc, dict):
        raise ValueError("unreadable shard output: %s" % path)
    return doc.get("frames", []), doc.get("version", "")


def merge_shards(shard_paths, subsample=1):
    """
    Concatenate shard outputs into a whole-clip frame table and recompute
    pooled metrics for every metric column.

    Returns (merged_doc, missing_frame_nums).
    """
    all_frames = []
    version = ""
    for path in shard_paths:
        frames, version = load_shard_frames(path)
        all_frames.extend(frames)

    all_frames.sort(key=lambda fr: fr.get("frameNum", -1))

    # Detect duplicate frame ownership (would indicate a planning bug).
    seen = set()
    duplicates = []
    for fr in all_frames:
        num = fr.get("frameNum")
        if num in seen:
            duplicates.append(num)
        seen.add(num)
    if duplicates:
        raise ValueError("duplicate frame numbers after merge: %s" %
                         sorted(set(duplicates)))

    if all_frames:
        first = all_frames[0]["frameNum"]
        last = all_frames[-1]["frameNum"]
        present = {fr["frameNum"] for fr in all_frames}
        missing = [i for i in range(first, last + 1) if i not in present]
    else:
        missing = []

    metric_names = set()
    for fr in all_frames:
        metric_names.update(fr.get("metrics", {}).keys())

    columns = OrderedDict((name, []) for name in sorted(metric_names))
    for fr in all_frames:
        if subsample > 1 and (fr["frameNum"] % subsample):
            continue
        for name in metric_names:
            value = fr.get("metrics", {}).get(name)
            if value is not None:
                columns[name].append(value)

    pooled = OrderedDict()
    for name, values in columns.items():
        pooled[name] = _pool(values)

    merged = {
        "version": version,
        "frames": all_frames,
        "pooled_metrics": pooled,
        "n_frames": len(all_frames),
        "subsample": subsample,
    }
    return merged, missing


def write_merged_result(result_path, job, merged, shard_count):
    payload = {
        "status": "done",
        "job_id": job.id,
        "content": job.content,
        "version": job.version,
        "n_frames": merged["n_frames"],
        "shard_count": shard_count,
        "pooled_metrics": merged["pooled_metrics"],
    }
    atomic_write_json(result_path, payload)
    return payload


def aggregate_results(store, jobs):
    """
    Build per-pair and per-version summary statistics from completed jobs.

    The summary is a plain dict so callers can render JSON or a table.
    Job ids / versions are the only identifiers included — never paths.
    """
    per_pair = []
    by_version = {}

    for job in jobs:
        state = store.job_state(job.id)
        result = read_json(state.result_path, default=None)
        entry = {
            "job_id": job.id,
            "content": job.content,
            "version": job.version,
            "status": "done" if result else state.get_status()["status"],
        }
        if result:
            pooled = result.get("pooled_metrics", {})
            vmaf_pooled = pooled.get("vmaf", pooled.get("VMAF", {}))
            entry["vmaf_mean"] = vmaf_pooled.get("mean")
            entry["vmaf_min"] = vmaf_pooled.get("min")
            entry["vmaf_harmonic_mean"] = vmaf_pooled.get("harmonic_mean")
            entry["n_frames"] = result.get("n_frames")
            bucket = by_version.setdefault(job.version, [])
            bucket.append(entry)
        per_pair.append(entry)

    version_summary = {}
    for version, entries in sorted(by_version.items()):
        scored = [e for e in entries if e.get("vmaf_mean") is not None]
        if not scored:
            continue
        means = [e["vmaf_mean"] for e in scored]
        version_summary[version] = {
            "job_count": len(entries),
            "vmaf_mean_avg": sum(means) / len(means),
            "vmaf_mean_min": min(means),
            "vmaf_mean_max": max(means),
            "worst_job_id": min(scored, key=lambda e: e["vmaf_mean"])["job_id"],
        }

    return {"per_pair": per_pair, "by_version": version_summary}
