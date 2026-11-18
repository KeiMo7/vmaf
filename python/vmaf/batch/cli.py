"""
Command-line entry point for batch quality analysis.

Subcommands
-----------
  run       Run (or resume) a batch described by a manifest JSON.
  report    Print the aggregated report from an existing state directory.
  retry     Re-queue failed jobs from a previous run (skips done/skipped).

The single-file `vmaf` CLI is untouched; this module only orchestrates it.
"""

import argparse
import json
import sys

from .logging_setup import configure_logging
from .manifest import load_manifest
from .resources import ResourceBudget
from .scheduler import BatchScheduler, SchedulerConfig
from .state import CheckpointStore, STATUS_DONE, STATUS_FAILED, STATUS_SKIPPED


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="python -m vmaf.batch.cli",
        description="Parallel, resumable batch VMAF analysis")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-file", default=None,
                        help="shared (path-sanitized) log file")

    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run or resume a batch")
    run.add_argument("--manifest", required=True)
    run.add_argument("--state-dir", required=True)
    run.add_argument("--model", required=True,
                     help="local model path; no network access is performed")
    run.add_argument("--vmaf-bin", default=None)
    run.add_argument("--shard-frames", type=int, default=0,
                     help="frames per time shard; 0 disables sharding")
    run.add_argument("--max-parallel-jobs", type=int, default=0)
    run.add_argument("--threads-per-shard", type=int, default=0)
    run.add_argument("--cpu-limit", type=int, default=0)
    run.add_argument("--mem-limit-mb", type=int, default=0)
    run.add_argument("--gpu-limit", type=int, default=0)
    run.add_argument("--mem-per-shard-mb", type=int, default=0)
    run.add_argument("--max-attempts", type=int, default=3)
    run.add_argument("--use-gpu", action="store_true")
    run.add_argument("--timeout-s", type=int, default=0)
    run.add_argument("--subsample", type=int, default=1)
    run.add_argument("--report", default=None,
                     help="write the aggregated JSON report here")

    retry = sub.add_parser("retry",
                           help="retry failed jobs from a state directory")
    retry.add_argument("--manifest", required=True)
    retry.add_argument("--state-dir", required=True)
    retry.add_argument("--model", required=True)
    retry.add_argument("--vmaf-bin", default=None)
    retry.add_argument("--shard-frames", type=int, default=0)
    retry.add_argument("--max-parallel-jobs", type=int, default=0)
    retry.add_argument("--threads-per-shard", type=int, default=0)
    retry.add_argument("--cpu-limit", type=int, default=0)
    retry.add_argument("--mem-limit-mb", type=int, default=0)
    retry.add_argument("--gpu-limit", type=int, default=0)
    retry.add_argument("--mem-per-shard-mb", type=int, default=0)
    retry.add_argument("--max-attempts", type=int, default=3)
    retry.add_argument("--use-gpu", action="store_true")
    retry.add_argument("--subsample", type=int, default=1)

    report = sub.add_parser("report", help="print an aggregated report")
    report.add_argument("--state-dir", required=True)
    report.add_argument("--manifest", required=True)
    return parser


def _budget_from_args(args):
    return ResourceBudget.autodetect(
        cpu_limit=args.cpu_limit,
        mem_limit_mb=args.mem_limit_mb,
        gpu_limit=args.gpu_limit,
        mem_per_shard_mb=args.mem_per_shard_mb)


def _config_from_args(args):
    return SchedulerConfig(
        state_dir=args.state_dir,
        model_path=args.model,
        vmaf_bin=args.vmaf_bin,
        shard_frames=args.shard_frames,
        max_parallel_jobs=args.max_parallel_jobs,
        threads_per_shard=args.threads_per_shard,
        mem_per_shard_mb=args.mem_per_shard_mb,
        max_attempts=args.max_attempts,
        use_gpu=args.use_gpu,
        timeout_s=getattr(args, "timeout_s", 0),
        subsample=getattr(args, "subsample", 1))


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    log = configure_logging(args.verbose, args.log_file)

    jobs = load_manifest(args.manifest)
    if args.command == "report":
        store = CheckpointStore(args.state_dir)
        from .merge import aggregate_results
        print(json.dumps(aggregate_results(store, jobs), indent=2,
                         sort_keys=True))
        return 0

    if args.command == "retry":
        # Drop terminal failures' FAILED marker so the scheduler re-plans
        # them; done/skipped jobs remain untouched by the resume logic.
        store = CheckpointStore(args.state_dir)
        for job in jobs:
            state = store.job_state(job.id)
            if state.get_status().get("status") == STATUS_FAILED:
                state.set_status("pending", reason="retry requested")

    budget = _budget_from_args(args)
    config = _config_from_args(args)
    scheduler = BatchScheduler(jobs, config, budget=budget)
    summary = scheduler.run()
    log.info("batch finished: %s", summary)

    report_path = getattr(args, "report", None) or \
        (args.state_dir.rstrip("/") + "/report.json")
    report = scheduler.write_report(report_path)
    print(json.dumps({"summary": summary,
                      "report_path": report_path,
                      "version_summary": report["by_version"]}, indent=2,
                     sort_keys=True))

    # Non-zero exit only when nothing usable completed? No: failures are
    # isolated; surface the counts but return 0 if the scheduler itself ran.
    failed = summary.get(STATUS_FAILED, 0)
    return 2 if failed and summary.get(STATUS_DONE, 0) == 0 else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
