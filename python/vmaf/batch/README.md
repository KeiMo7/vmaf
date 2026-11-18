# Parallel, resumable batch quality analysis

`vmaf.batch` runs many reference/distorted video pairs through the existing
`vmaf` CLI in parallel, with hard resource limits, time-sharded single-pair
parallelism, failure isolation and resumable checkpoints.

The single-file `vmaf` command line, its output format and every metric value
are unchanged: batch mode is a pure orchestration layer on top of it.

## Concepts

* **Job / pair** — one reference + one distorted video, identified by an
  opaque `id`. Paths live in the manifest and per-job state only, never in the
  shared log.
* **Shard** — a contiguous frame range of one pair. A pair is split into
  shards so several workers can process one long video concurrently.
* **Warm-up / look-ahead** — temporal features (integer motion) depend on
  neighboring frames. Each shard *feeds* `[start-2, end+1)` but only *emits*
  its owned `[start, end)` range, so merged per-frame scores are identical to
  a whole-file run (verified frame by frame in the test suite).
* **State directory** — atomic, idempotent checkpoint store. Re-running with
  the same state directory resumes without recomputing finished work.

## Manifest

```json
{
  "jobs": [
    {
      "id": "clipA-v1",
      "reference": "/data/src/clipA.y4m",
      "distorted": "/data/out/clipA-v1.y4m",
      "version": "v1",
      "content": "clipA"
    }
  ]
}
```

Raw `.yuv` jobs also take `width`, `height`, `pixel_format` and `bitdepth`.

## Running

```
python -m vmaf.batch.cli run \
  --manifest jobs.json \
  --state-dir ./batch-state \
  --model /path/to/local/vmaf_v0.6.1.json \
  --shard-frames 60 \
  --max-parallel-jobs 4 \
  --threads-per-shard 2 \
  --cpu-limit 16 --mem-limit-mb 32768 --gpu-limit 0 \
  --mem-per-shard-mb 2048 \
  --report report.json
```

No model is downloaded: `--model` always points at a local file.

* Omit `--shard-frames` to process each pair as a single shard.
* Limits default to the machine's usable CPU count (cgroup-aware), RAM
  (cgroup-aware) and detected GPU count; every flag overrides detection.
* `report` and `retry` are separate subcommands. `retry` re-queues only jobs
  marked `failed`; `done` and `skipped` jobs are left untouched.

## Resource enforcement

A shard is admitted only when **all three** dimensions fit the budget: CPU
threads, estimated RAM and GPU slots. On an OOM kill the scheduler halves the
job's thread count and raises its memory reservation, lowering effective
parallelism until the workload fits — it never aborts the batch. Observed peak
RSS feeds back into subsequent reservations.

## Failure handling

* A decoder/input error quarantines just that pair as `skipped` with a reason;
  every other pair continues.
* Other failures retry up to `--max-attempts`, then the pair is `failed`.
* A missing shard on resume is localized and recomputed alone.
* Each job's captured stderr stays in its private `error.log`; the shared log
  references job ids and redacts path-like tokens via the logging filter.

## State layout

```
state/
  manifest.json
  jobs/<id>/
    job.json            # immutable spec (the only place paths are stored)
    status.json         # atomically replaced status + per-shard outcomes
    shards.plan.json
    shards/shardNNNN.json[.meta.json]
    result.json         # completion marker, written after a verified merge
    error.log
  report.json
```

All writes use temp-file + fsync + rename, so an interrupted process never
leaves a half-written record.

## Equivalence guarantee

`vmaf_write_output_range` emits only the shard's owned frames while warm-up
and look-ahead frames still flow through the context. The Python merge
concatenates per-frame outputs and recomputes pooled statistics with the same
float64 formulas and accumulation order as `vmaf_feature_score_pooled`.
`batch_scheduler_test.py` asserts that sharded merge and whole-file output
match on every frame and every metric.
