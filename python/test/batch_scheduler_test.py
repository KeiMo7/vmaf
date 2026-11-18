"""
Integration tests for the batch parallel scheduler.

These exercise the real `vmaf` binary on synthetic local clips, covering:
  1. parallel batch execution with per-version aggregation,
  2. time sharding whose merged result equals whole-file processing,
  3. failure isolation: a decoder-broken pair is skipped, others finish,
  4. resumability: an interrupted run never recomputes finished shards,
  5. gap localization: a deleted shard is recomputed, nothing else,
  6. resource budget enforcement across CPU/memory/GPU dimensions,
  7. single-file CLI byte-compatibility when shard options are unused.

The binary is located via VMAF_BIN (CI sets this) or common build paths.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from vmaf.batch.logging_setup import redact
from vmaf.batch.manifest import count_y4m_frames, load_manifest
from vmaf.batch.merge import merge_shards
from vmaf.batch.resources import ResourceBudget, ResourcePool
from vmaf.batch.runner import (classify_failure, find_vmaf_binary,
                               validate_shard_output)
from vmaf.batch.scheduler import BatchScheduler, SchedulerConfig
from vmaf.batch.sharding import find_missing_shards, plan_shards
from vmaf.batch.state import (CheckpointStore, STATUS_DONE, STATUS_SKIPPED,
                              atomic_write_json, read_json)

from batch_videomaker import (make_broken_y4m, make_distorted,
                              make_reference)

REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = str(REPO_ROOT / "model" / "vmaf_v0.6.1.json")


def _locate_vmaf():
    env = os.environ.get("VMAF_BIN")
    candidates = [env] if env else []
    candidates += [
        "/tmp/vmaf-build/tools/vmaf",
        str(REPO_ROOT / "libvmaf" / "build" / "tools" / "vmaf"),
        str(REPO_ROOT / "libvmaf" / "builddir" / "tools" / "vmaf"),
    ]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return shutil.which("vmaf")


VMAF_BIN = _locate_vmaf()


@unittest.skipUnless(VMAF_BIN and os.path.isfile(MODEL_PATH),
                     "vmaf binary or local model unavailable")
class BatchTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="vmaf-batch-test-")
        self.work = Path(self.tmp)
        self.ref = make_reference(str(self.work / "ref.y4m"), frames=24)
        self.n_frames = count_y4m_frames(self.ref)
        self.assertEqual(self.n_frames, 24)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def make_dist(self, name, **kwargs):
        return make_distorted(str(self.work / name), frames=self.n_frames,
                              **kwargs)

    def write_manifest(self, jobs):
        path = self.work / "manifest.json"
        with open(path, "w") as f:
            json.dump({"jobs": jobs}, f)
        return str(path)

    def make_scheduler(self, manifest, state_dir, **kwargs):
        jobs = load_manifest(manifest)
        config = SchedulerConfig(
            state_dir=state_dir,
            model_path=MODEL_PATH,
            vmaf_bin=VMAF_BIN,
            max_attempts=kwargs.pop("max_attempts", 2),
            **kwargs)
        budget = ResourceBudget.autodetect(
            cpu_limit=kwargs.get("cpu_limit", 2),
            mem_limit_mb=kwargs.get("mem_limit_mb", 2048),
            gpu_limit=0,
            mem_per_shard_mb=kwargs.get("mem_per_shard_mb", 256))
        return BatchScheduler(jobs, config, budget=budget)

    def run_vmaf_cli(self, ref, dist, output, *extra_args):
        cmd = [VMAF_BIN, "--reference", ref, "--distorted", dist,
               "--json", "--output", output,
               "--model", "path=%s" % MODEL_PATH, "--threads", "1",
               "--quiet", *extra_args]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        return output


class ShardPlanningTest(unittest.TestCase):
    def test_windows_cover_all_frames_with_warmup_and_lookahead(self):
        plans = plan_shards(24, shard_frames=8)
        self.assertEqual([p["emit_start"] for p in plans], [0, 8, 16])
        # First shard: no negative feed start.
        self.assertEqual((plans[0]["feed_start"], plans[0]["feed_end"]),
                         (0, 9))   # 8 owned + 1 lookahead
        # Middle shard: 2 warmup + 8 owned + 1 lookahead
        self.assertEqual((plans[1]["feed_start"], plans[1]["feed_end"]),
                         (6, 17))
        # Last shard: lookahead clamped at total frames
        self.assertEqual((plans[2]["feed_start"], plans[2]["feed_end"]),
                         (14, 24))
        # Emit windows partition the full range exactly once
        emitted = []
        for p in plans:
            emitted.extend(range(p["emit_start"], p["emit_end"]))
        self.assertEqual(emitted, list(range(24)))

    def test_find_missing_shards_localizes_gap(self):
        plans = plan_shards(24, 8)
        missing = find_missing_shards(plans, present_shard_ids={0, 2})
        self.assertEqual([p["index"] for p in missing], [1])


class ShardMergeEquivalenceTest(BatchTestBase):
    """Acceptance: sharded merge == whole-file result, frame by frame."""

    def test_sharded_merge_matches_whole_file(self):
        dist = self.make_dist("dist.y4m", noise=10)

        # Whole-file baseline through the untouched single-file CLI path.
        full_path = str(self.work / "full.json")
        self.run_vmaf_cli(self.ref, dist, full_path)
        full = read_json(full_path)

        # Sharded run: 3 shards of 8 frames (warmup/lookahead handled by CLI).
        manifest = self.write_manifest([{
            "id": "pair-A", "reference": self.ref, "distorted": dist,
            "version": "v1", "content": "clipA"}])
        state_dir = str(self.work / "state")
        scheduler = self.make_scheduler(
            manifest, state_dir, shard_frames=8,
            threads_per_shard=1, max_parallel_jobs=2)
        summary = scheduler.run()
        self.assertEqual(summary[STATUS_DONE], 1)

        store = CheckpointStore(state_dir)
        result = read_json(store.job_state("pair-A").result_path)
        self.assertEqual(result["n_frames"], self.n_frames)
        self.assertEqual(result["shard_count"], 3)

        shard_paths = sorted(p for p in (
            Path(state_dir) / "jobs" / "pair-A" / "shards"
        ).glob("shard*.json") if not p.name.endswith(".meta.json"))
        merged, missing = merge_shards([str(p) for p in shard_paths])
        self.assertEqual(missing, [])

        # Frame-level exactness on every metric column.
        self.assertEqual(len(merged["frames"]), len(full["frames"]))
        for merged_frame, full_frame in zip(merged["frames"],
                                            full["frames"]):
            self.assertEqual(merged_frame["frameNum"],
                             full_frame["frameNum"])
            self.assertEqual(set(merged_frame["metrics"]),
                             set(full_frame["metrics"]))
            for name, value in full_frame["metrics"].items():
                self.assertEqual(merged_frame["metrics"][name], value,
                                 "metric %s diverged on frame %d" %
                                 (name, full_frame["frameNum"]))

        # Pooled stats must match the CLI's own pooled_metrics block. The CLI
        # serializes them to 6 decimals, so compare at that precision while
        # the merge itself accumulates full float64 values.
        for name, pools in full["pooled_metrics"].items():
            for method, expected in pools.items():
                if expected is None:
                    continue
                self.assertAlmostEqual(
                    merged["pooled_metrics"][name][method], expected,
                    places=5,
                    msg="pooled %s/%s diverged" % (name, method))


class ParallelBatchTest(BatchTestBase):
    def test_parallel_batch_and_version_aggregation(self):
        d1 = self.make_dist("d1.y4m", noise=6)
        d2 = self.make_dist("d2.y4m", noise=20, corrupt_after=12)
        d3 = self.make_dist("d3.y4m", noise=8)
        manifest = self.write_manifest([
            {"id": "c1-v1", "reference": self.ref, "distorted": d1,
             "version": "v1", "content": "clip1"},
            {"id": "c1-v2", "reference": self.ref, "distorted": d2,
             "version": "v2", "content": "clip1"},
            {"id": "c2-v1", "reference": self.ref, "distorted": d3,
             "version": "v1", "content": "clip2"},
        ])
        state_dir = str(self.work / "state")
        scheduler = self.make_scheduler(
            manifest, state_dir, shard_frames=6,
            threads_per_shard=1, max_parallel_jobs=3)
        summary = scheduler.run()
        self.assertEqual(summary[STATUS_DONE], 3)

        report = scheduler.write_report(str(self.work / "report.json"))
        self.assertEqual(len(report["per_pair"]), 3)
        self.assertEqual(set(report["by_version"]), {"v1", "v2"})
        # The visibly degraded version must rank worst on its shared content.
        v1 = [e for e in report["per_pair"] if e["job_id"] == "c1-v1"][0]
        v2 = [e for e in report["per_pair"] if e["job_id"] == "c1-v2"][0]
        self.assertLess(v2["vmaf_mean"], v1["vmaf_mean"])

        # Public report contains no filesystem paths.
        raw = json.dumps(report)
        self.assertNotIn(self.tmp, raw)


class FailureIsolationTest(BatchTestBase):
    def test_decoder_failure_skips_pair_only(self):
        good = self.make_dist("good.y4m", noise=10)
        broken = make_broken_y4m(str(self.work / "broken.y4m"))
        manifest = self.write_manifest([
            {"id": "good-pair", "reference": self.ref, "distorted": good},
            {"id": "broken-pair", "reference": self.ref,
             "distorted": broken},
        ])
        state_dir = str(self.work / "state")
        scheduler = self.make_scheduler(
            manifest, state_dir, threads_per_shard=1,
            max_parallel_jobs=2)
        summary = scheduler.run()
        self.assertEqual(summary[STATUS_DONE], 1)
        self.assertEqual(summary[STATUS_SKIPPED], 1)

        store = CheckpointStore(state_dir)
        broken_status = store.job_state("broken-pair").get_status()
        self.assertEqual(broken_status["status"], STATUS_SKIPPED)
        self.assertTrue(broken_status.get("reason"))
        # The healthy pair still produced a full result.
        good_result = read_json(store.job_state("good-pair").result_path)
        self.assertEqual(good_result["status"], STATUS_DONE)

    def test_classify_failure_maps_decoder_and_oom(self):
        self.assertEqual(
            classify_failure(1, "problem with distorted file: x"),
            "decoder_unavailable")
        self.assertEqual(classify_failure(-9, "killed"), "out_of_memory")
        self.assertEqual(classify_failure(1, "some other error"), "failed")
        self.assertEqual(classify_failure(0, ""), "ok")

    def test_skipped_pair_remains_quarantined_on_resume(self):
        good = self.make_dist("good.y4m", noise=10)
        broken = make_broken_y4m(str(self.work / "broken.y4m"))
        manifest = self.write_manifest([
            {"id": "good-pair", "reference": self.ref, "distorted": good},
            {"id": "broken-pair", "reference": self.ref,
             "distorted": broken},
        ])
        state_dir = str(self.work / "state")
        scheduler = self.make_scheduler(
            manifest, state_dir, threads_per_shard=1,
            max_parallel_jobs=2)
        first = scheduler.run()
        self.assertEqual(first[STATUS_SKIPPED], 1)

        store = CheckpointStore(state_dir)
        err_log = Path(store.job_state("broken-pair").error_log_path())
        baseline_size = err_log.stat().st_size if err_log.exists() else 0

        # Second run must honor the terminal SKIPPED state and never re-probe
        # the broken decoder (which would append another failure to its log).
        scheduler2 = self.make_scheduler(
            manifest, state_dir, threads_per_shard=1,
            max_parallel_jobs=2)
        second = scheduler2.run()
        self.assertEqual(second[STATUS_SKIPPED], 1)
        self.assertEqual(second[STATUS_DONE], 1)
        new_size = err_log.stat().st_size if err_log.exists() else 0
        self.assertEqual(new_size, baseline_size)


class ResumeTest(BatchTestBase):
    """Acceptance: resume never recomputes finished work."""

    def test_resume_skips_completed_shards_and_jobs(self):
        dist = self.make_dist("dist.y4m", noise=10)
        manifest = self.write_manifest([{
            "id": "pair-R", "reference": self.ref, "distorted": dist}])
        state_dir = str(self.work / "state")

        scheduler = self.make_scheduler(
            manifest, state_dir, shard_frames=6,
            threads_per_shard=1, max_parallel_jobs=2)
        summary = scheduler.run()
        self.assertEqual(summary[STATUS_DONE], 1)

        store = CheckpointStore(state_dir)
        state = store.job_state("pair-R")
        shard_files = sorted(
            p for p in
            (Path(state_dir) / "jobs" / "pair-R" / "shards")
            .glob("shard[0-9]*.json")
            if not p.name.endswith(".meta.json"))
        self.assertTrue(shard_files)
        mtimes = {p.name: p.stat().st_mtime_ns for p in shard_files}
        result_mtime = Path(state.result_path).stat().st_mtime_ns

        # Second run over the same state directory: everything must be
        # trusted as-is, with zero shard recomputation.
        scheduler2 = self.make_scheduler(
            manifest, state_dir, shard_frames=6,
            threads_per_shard=1, max_parallel_jobs=2)
        summary2 = scheduler2.run()
        self.assertEqual(summary2[STATUS_DONE], 1)
        for p in shard_files:
            self.assertEqual(p.stat().st_mtime_ns, mtimes[p.name],
                             "shard %s was recomputed on resume" % p.name)
        self.assertEqual(Path(state.result_path).stat().st_mtime_ns,
                         result_mtime)

    def test_missing_shard_gap_is_recomputed_alone(self):
        dist = self.make_dist("dist.y4m", noise=10)
        manifest = self.write_manifest([{
            "id": "pair-G", "reference": self.ref, "distorted": dist}])
        state_dir = str(self.work / "state")
        scheduler = self.make_scheduler(
            manifest, state_dir, shard_frames=6,
            threads_per_shard=1, max_parallel_jobs=2)
        scheduler.run()

        shard_dir = Path(state_dir) / "jobs" / "pair-G" / "shards"
        survivors = sorted(
            p for p in shard_dir.glob("shard[0-9]*.json")
            if not p.name.endswith(".meta.json"))
        # Per-file baseline mtimes, so each surviving shard is compared only
        # against its own previous state.
        baseline_mtimes = {p.name: p.stat().st_mtime_ns for p in survivors}
        victim = survivors[1]
        os.unlink(victim)
        meta = Path(str(victim) + ".meta.json")
        if meta.exists():
            os.unlink(meta)
        # Also remove the completion marker so re-finalization runs.
        result_path = Path(state_dir) / "jobs" / "pair-G" / "result.json"
        os.unlink(result_path)

        scheduler2 = self.make_scheduler(
            manifest, state_dir, shard_frames=6,
            threads_per_shard=1, max_parallel_jobs=2)
        summary = scheduler2.run()
        self.assertEqual(summary[STATUS_DONE], 1)
        # Only the deleted shard was rebuilt; every survivor keeps its mtime.
        self.assertTrue(victim.exists())
        for name, baseline in baseline_mtimes.items():
            p = shard_dir / name
            if p == victim:
                continue
            self.assertEqual(p.stat().st_mtime_ns, baseline,
                             "unexpected recomputation of %s" % name)


class ResourceBudgetTest(unittest.TestCase):
    def test_pool_requires_all_three_dimensions(self):
        budget = ResourceBudget(cpu_total=4, mem_bytes_total=1000,
                                gpu_total=1, mem_per_shard_bytes=250)
        pool = ResourcePool(budget)
        # Fits everything.
        self.assertTrue(pool.admit(2, 200, 1))
        # CPU exhausted -> rejected even though memory is free.
        self.assertFalse(pool.can_admit(3, 100, 0))
        # Memory exhausted -> rejected even though CPU frees up.
        self.assertFalse(pool.can_admit(1, 900, 0))
        # The single GPU slot is already taken -> a second GPU shard rejected.
        self.assertFalse(pool.can_admit(1, 100, 1))
        pool.release(2, 200, 1)
        self.assertTrue(pool.can_admit(1, 100, 1))

    def test_gpu_disabled_build_rejects_gpu_shards(self):
        budget = ResourceBudget(cpu_total=4, mem_bytes_total=1000,
                                gpu_total=0, mem_per_shard_bytes=250)
        pool = ResourcePool(budget)
        self.assertFalse(pool.can_admit(1, 100, 1))


class DegradationTest(unittest.TestCase):
    """Resource pressure must lower parallelism, never abort the batch."""

    def _scheduler(self):
        from vmaf.batch.manifest import Job
        jobs = [Job(id="j1", reference="/tmp/r.y4m", distorted="/tmp/d.y4m")]
        config = SchedulerConfig(
            state_dir=tempfile.mkdtemp(),
            model_path=MODEL_PATH, vmaf_bin=VMAF_BIN,
            threads_per_shard=8, mem_per_shard_mb=100)
        budget = ResourceBudget(cpu_total=16, mem_bytes_total=1000,
                                gpu_total=0, mem_per_shard_bytes=100)
        sched = BatchScheduler(jobs, config, budget=budget)
        # Seed an adaptive profile directly; degradation does not require a
        # full preflight or real input files.
        sched._profiles["j1"] = {"threads": 8, "mem_bytes": 100, "gpus": 0}
        return sched

    def test_oom_degradation_halves_threads_and_raises_memory(self):
        sched = self._scheduler()
        profile = sched._profiles["j1"]

        sched._degrade_profile("j1")
        self.assertEqual(profile["threads"], 4)
        self.assertEqual(profile["mem_bytes"], 200)

        # Repeated degradation bottoms out at a single thread instead of
        # failing, and memory is clamped to the total budget.
        for _ in range(10):
            sched._degrade_profile("j1")
        self.assertEqual(profile["threads"], 1)
        self.assertLessEqual(profile["mem_bytes"],
                             sched.budget.mem_bytes_total)

    def test_measured_peak_memory_updates_profile(self):
        sched = self._scheduler()
        profile = sched._profiles["j1"]
        huge_peak = profile["mem_bytes"] + 5000
        sched._record_peak_memory("j1", huge_peak)
        # 25% safety margin above the observed peak.
        self.assertGreaterEqual(profile["mem_bytes"], huge_peak)
        sched._record_peak_memory("j1", 1)  # smaller peak must not shrink it
        self.assertGreaterEqual(profile["mem_bytes"], huge_peak)


class LogSanitizerTest(unittest.TestCase):
    def test_redact_strips_paths_but_keeps_ids(self):
        self.assertNotIn("/home/wll/secret/clip.y4m",
                         redact("job pair-A read /home/wll/secret/clip.y4m"))
        self.assertIn("pair-A",
                      redact("job pair-A finished"))
        self.assertEqual(redact(42), 42)


class SingleFileCompatibilityTest(BatchTestBase):
    """Acceptance: without shard flags the CLI output is byte-identical."""

    def test_default_flags_emit_all_frames(self):
        dist = self.make_dist("dist.y4m", noise=10)
        out1 = str(self.work / "a.json")
        out2 = str(self.work / "b.json")
        self.run_vmaf_cli(self.ref, dist, out1)
        # Explicit zero-valued shard options must not change any byte, except
        # the timing-derived fps field which legitimately varies per run.
        self.run_vmaf_cli(self.ref, dist, out2,
                          "--frame_offset", "0",
                          "--frame_index_offset", "0",
                          "--emit_start", "0", "--emit_cnt", "0")

        def without_fps(path):
            doc = read_json(path)
            doc.pop("fps", None)
            return json.dumps(doc, sort_keys=True)

        self.assertEqual(without_fps(out1), without_fps(out2))

    def test_emit_window_filters_frames(self):
        dist = self.make_dist("dist.y4m", noise=10)
        out = str(self.work / "window.json")
        # Skip 4 input frames, feed 10 (global 4..13), emit only 6..9.
        self.run_vmaf_cli(self.ref, dist, out,
                          "--frame_offset", "4",
                          "--frame_index_offset", "4",
                          "--frame_cnt", "10",
                          "--emit_start", "6", "--emit_cnt", "4")
        doc = read_json(out)
        self.assertEqual([f["frameNum"] for f in doc["frames"]],
                         [6, 7, 8, 9])


if __name__ == "__main__":
    unittest.main(verbosity=2)
