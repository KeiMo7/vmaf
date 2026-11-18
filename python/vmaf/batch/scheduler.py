"""
Parallel batch scheduler.

Design
------
* A fixed worker pool drains a shared pending queue of (job, shard) tasks.
* Before every subprocess launch the shard must be admitted by a
  three-dimensional ResourcePool (CPU threads, estimated RAM, GPU slots).
  A shard that does not fit parks and retries later instead of failing.
* Memory pressure / OOM kills trigger graceful degradation: the shard's
  thread count is halved and its memory estimate doubled, shrinking the
  effective parallelism until the workload fits. The batch never aborts on
  resource exhaustion.
* Decoder-class failures quarantine the whole job as `skipped` with a
  reason; other jobs are unaffected.
* Completed shards and merged results on disk are trusted after validation,
  so interrupting and re-running the same state directory resumes without
  recomputing anything (idempotent).
* The scheduler log references jobs by opaque id only — never paths.
"""

import json
import logging
import os
import queue
import threading
import time

from .manifest import count_y4m_frames, y4m_header
from .merge import aggregate_results, merge_shards, write_merged_result
from .resources import (ResourceBudget, ResourcePool,
                        available_system_memory_bytes)
from .runner import (EXIT_DECODER, EXIT_OOM, ShardResult, find_vmaf_binary,
                     run_shard, validate_shard_output)
from .sharding import find_missing_shards, plan_shards
from .state import (CheckpointStore, STATUS_DONE, STATUS_FAILED,
                    STATUS_PENDING, STATUS_RUNNING, STATUS_SKIPPED,
                    atomic_write_json, read_json)

LOG = logging.getLogger("vmaf.batch")


class SchedulerConfig:
    def __init__(self, state_dir, model_path, vmaf_bin=None,
                 shard_frames=0, max_parallel_jobs=0, threads_per_shard=0,
                 mem_per_shard_mb=0, max_attempts=3, use_gpu=False,
                 timeout_s=0, subsample=1):
        self.state_dir = state_dir
        self.model_path = model_path
        self.vmaf_bin = vmaf_bin
        self.shard_frames = shard_frames          # 0 = one shard per job
        self.max_parallel_jobs = max_parallel_jobs
        self.threads_per_shard = threads_per_shard
        self.mem_per_shard_mb = mem_per_shard_mb
        self.max_attempts = max_attempts
        self.use_gpu = use_gpu
        self.timeout_s = timeout_s
        self.subsample = subsample


class BatchScheduler:
    def __init__(self, jobs, config: SchedulerConfig,
                 budget: ResourceBudget = None):
        self.jobs = {job.id: job for job in jobs}
        self.config = config
        self.store = CheckpointStore(config.state_dir)
        self.budget = budget or ResourceBudget.autodetect(
            mem_per_shard_mb=config.mem_per_shard_mb)
        self.pool = ResourcePool(self.budget)
        self.vmaf_bin = find_vmaf_binary(config.vmaf_bin)
        self._lock = threading.Lock()
        # Per-job adaptive resource profile, adjusted on OOM.
        self._profiles = {}
        # Jobs that must not be scheduled (decoder failure / manual skip).
        self._quarantine = set()
        self._queue = queue.PriorityQueue()
        self._seq = 0

    # ------------------------------------------------------------------ #
    # Planning & preflight
    # ------------------------------------------------------------------ #
    def prepare(self):
        """Resolve frame counts, shard plans and decoder availability."""
        self.store.write_manifest_snapshot({
            "budget": self.budget.as_dict(),
            "config": {k: v for k, v in vars(self.config).items()
                       if k != "vmaf_bin"},
            "job_ids": [j.id for j in self.jobs.values()],
        })
        plans = {}
        for job in self.jobs.values():
            state = self.store.job_state(job.id)
            self.store.write_job_spec(job)

            # Respect terminal states from a previous run. A skipped job stays
            # quarantined without re-probing its (still-broken) decoder; the
            # `retry` CLI command resets FAILED jobs explicitly, and completed
            # jobs are short-circuited later via their result marker.
            prior = state.get_status()
            if prior.get("status") == STATUS_SKIPPED:
                self._quarantine.add(job.id)
                LOG.info("job %s remains skipped from prior run: %s",
                         job.id, prior.get("reason", ""))
                continue

            problems = job.validate()
            if problems:
                reason = "; ".join(problems)
                state.set_status(STATUS_SKIPPED, reason=reason)
                self._quarantine.add(job.id)
                LOG.warning("job %s skipped: %s", job.id, reason)
                continue

            total_frames = self._frame_count(job)
            if total_frames <= 0:
                state.set_status(STATUS_SKIPPED,
                                 reason="could not determine frame count")
                self._quarantine.add(job.id)
                continue

            shard_frames = self.config.shard_frames or total_frames
            shards = plan_shards(total_frames, shard_frames)
            plans[job.id] = shards
            atomic_write_json(os.path.join(state.dir, "shards.plan.json"),
                              {"total_frames": total_frames, "shards": shards})

            status = state.get_status()
            if status.get("status") != STATUS_RUNNING:
                state.set_status(STATUS_PENDING, total_frames=total_frames,
                                 shard_count=len(shards))
            self._profiles[job.id] = self._initial_profile()
        return plans

    def _frame_count(self, job):
        try:
            if job.reference.lower().endswith(".y4m"):
                return count_y4m_frames(job.reference)
        except (OSError, ValueError) as exc:
            LOG.warning("job %s frame probe failed: %s", job.id, exc)
        return int(job.options.get("frame_count", 0) or 0)

    def _initial_profile(self):
        threads = self.config.threads_per_shard or max(
            1, self.budget.cpu_total // max(1, self._job_count_estimate()))
        threads = min(threads, self.budget.cpu_total)
        mem = self.budget.mem_per_shard_bytes
        return {"threads": max(1, threads), "mem_bytes": mem, "gpus": 0}

    def _job_count_estimate(self):
        if self.config.max_parallel_jobs:
            return self.config.max_parallel_jobs
        return max(1, len(self.jobs))

    # ------------------------------------------------------------------ #
    # Queueing, including gap-localized shard recomputation
    # ------------------------------------------------------------------ #
    def enqueue_work(self, plans):
        for job_id, shards in plans.items():
            if job_id in self._quarantine:
                continue
            state = self.store.job_state(job_id)
            if state.is_complete():
                LOG.info("job %s already complete; skipping resume", job_id)
                continue

            present = self._validated_shard_ids(state, shards)
            missing = find_missing_shards(shards, present)
            if not missing:
                self._finalize_job(job_id, shards)
                continue
            for shard in missing:
                self._enqueue(job_id, shard)

    def _validated_shard_ids(self, state, shards):
        ok_ids = set()
        for shard in shards:
            out = state.shard_output_path(shard["index"])
            if validate_shard_output(out, shard) is None:
                ok_ids.add(shard["index"])
        return ok_ids

    def _enqueue(self, job_id, shard):
        # Lower shard index first so logs read in frame order.
        self._seq += 1
        self._queue.put((shard["emit_start"], self._seq, job_id, shard))

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(self):
        plans = self.prepare()
        self.enqueue_work(plans)
        worker_count = self.config.max_parallel_jobs or min(
            self.budget.cpu_total, max(1, self._queue.qsize()))
        worker_count = max(1, worker_count)
        LOG.info("scheduler starting: budget=%s workers=%d shards_queued=%d",
                 self.budget.as_dict(), worker_count, self._queue.qsize())

        threads = []
        for _ in range(worker_count):
            t = threading.Thread(target=self._worker, args=(plans,),
                                 daemon=True)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        # Anything left unqueued because it got quarantined stays skipped;
        # merge everything that completed.
        for job_id, shards in plans.items():
            state = self.store.job_state(job_id)
            if not state.is_complete() and job_id not in self._quarantine:
                present = self._validated_shard_ids(state, shards)
                if len(present) == len(shards):
                    self._finalize_job(job_id, shards)
                else:
                    status = state.get_status()
                    if status.get("status") != STATUS_FAILED:
                        state.set_status(STATUS_FAILED,
                                         reason="incomplete after run")
        return self.store.summary()

    def _worker(self, plans):
        while True:
            try:
                _, _, job_id, shard = self._queue.get_nowait()
            except queue.Empty:
                return
            try:
                self._process_shard(job_id, shard, plans)
            except Exception as exc:  # absolute failure isolation
                LOG.exception("worker error on job %s shard %d: %s",
                              job_id, shard["index"], exc)
                state = self.store.job_state(job_id)
                state.record_shard(shard["index"], shard, False, str(exc))
            finally:
                self._queue.task_done()

    def _process_shard(self, job_id, shard, plans):
        job = self.jobs[job_id]
        state = self.store.job_state(job_id)

        # Idempotency re-check under the lock: another worker may have
        # finished and merged this job while we waited for budget.
        if state.is_complete():
            return

        attempts = int(state.get_status().get("attempts", 0))
        while True:
            profile = self._profiles[job_id]
            admitted = self._admit_with_backoff(job_id, profile)
            if not admitted:
                # Should not happen (blocking helper), but stay defensive.
                self._enqueue(job_id, shard)
                return

            attempts += 1
            state.record_attempt()
            result = run_shard(
                self.vmaf_bin, job, shard, self.config.model_path,
                state.shard_output_path(shard["index"]),
                state.error_log_path(),
                threads=profile["threads"],
                use_gpu=self.config.use_gpu and bool(profile["gpus"]),
                timeout_s=self.config.timeout_s)
            self.pool.release(profile["threads"], profile["mem_bytes"],
                              profile["gpus"])
            self._record_peak_memory(job_id, result.peak_rss_bytes)

            if result.outcome == "ok":
                state.record_shard(shard["index"], shard, True)
                LOG.info("job %s shard %d/%d done (%.1fs, %d MiB peak)",
                         job_id, shard["index"] + 1, len(plans[job_id]),
                         result.duration_s,
                         result.peak_rss_bytes // (1024 * 1024))
                self._maybe_finalize(job_id, plans[job_id])
                return

            if result.outcome == EXIT_DECODER:
                # Permanent for this pair; never retry, never block others.
                state.set_status(STATUS_SKIPPED,
                                 reason="decoder unavailable: %s"
                                        % result.detail)
                self._quarantine.add(job_id)
                LOG.warning("job %s skipped (decoder): %s", job_id,
                            result.detail)
                return

            if result.outcome == EXIT_OOM:
                degraded = self._degrade_profile(job_id)
                LOG.warning("job %s shard %d OOM; degraded profile to %s",
                            job_id, shard["index"], degraded)
                if attempts >= self.config.max_attempts:
                    break
                continue

            LOG.warning("job %s shard %d failed: %s", job_id,
                        shard["index"], result.detail)
            if attempts >= self.config.max_attempts:
                break

        state.record_shard(shard["index"], shard, False, result.detail)
        state.set_status(STATUS_FAILED,
                         reason="shard %d: %s" % (shard["index"],
                                                  result.detail))

    def _admit_with_backoff(self, job_id, profile):
        """Block until the 3D pool admits the shard, reacting to pressure."""
        waited = 0.0
        while True:
            with self._lock:
                if self.pool.can_admit(profile["threads"],
                                       profile["mem_bytes"],
                                       profile["gpus"]):
                    return self.pool.admit(profile["threads"],
                                           profile["mem_bytes"],
                                           profile["gpus"])
            # System-wide pressure backoff, independent of our reservations.
            avail = available_system_memory_bytes()
            if avail and avail < profile["mem_bytes"]:
                time.sleep(0.25)
                waited += 0.25
                if waited > 2.0 and self._degrade_profile(job_id):
                    profile = self._profiles[job_id]
                    waited = 0.0
                continue
            time.sleep(0.05)

    def _degrade_profile(self, job_id):
        profile = self._profiles[job_id]
        old_threads = profile["threads"]
        profile["threads"] = max(1, old_threads // 2)
        # Conservatively double the memory reservation; the observed OOM may
        # have been caused by under-estimation, not just parallelism.
        profile["mem_bytes"] = min(self.budget.mem_bytes_total,
                                   profile["mem_bytes"] * 2)
        changed = old_threads != profile["threads"]
        return changed or True

    def _record_peak_memory(self, job_id, peak_rss):
        if peak_rss <= 0:
            return
        profile = self._profiles[job_id]
        # Learn from real measurements so future admits use real sizes.
        profile["mem_bytes"] = max(profile["mem_bytes"],
                                   int(peak_rss * 1.25))

    def _maybe_finalize(self, job_id, shards):
        state = self.store.job_state(job_id)
        if self._validated_shard_ids(state, shards) == \
                {s["index"] for s in shards}:
            self._finalize_job(job_id, shards)

    def _finalize_job(self, job_id, shards):
        state = self.store.job_state(job_id)
        if state.is_complete():
            return
        paths = [state.shard_output_path(s["index"]) for s in shards]
        try:
            merged, missing = merge_shards(paths,
                                           subsample=self.config.subsample)
        except (OSError, ValueError) as exc:
            state.set_status(STATUS_FAILED, reason="merge failed: %s" % exc)
            return
        if missing:
            state.set_status(STATUS_FAILED,
                             reason="gap after merge: frames %s" % missing)
            return
        write_merged_result(state.result_path, self.jobs[job_id], merged,
                            len(shards))
        state.set_status(STATUS_DONE)
        LOG.info("job %s merged: %d frames from %d shards", job_id,
                 merged["n_frames"], len(shards))

    def build_report(self):
        return aggregate_results(self.store, list(self.jobs.values()))

    def write_report(self, path):
        report = self.build_report()
        atomic_write_json(path, report)
        return report
