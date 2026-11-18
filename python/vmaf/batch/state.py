"""
Idempotent checkpoint state for resumable batch runs.

Layout under the state directory:

    state/
      manifest.json                 # resolved job list + run settings
      jobs/<job_id>/
        job.json                    # immutable job spec (paths live here only)
        status.json                 # current status, atomically replaced
        shards/
          shard0000.json            # raw per-shard vmaf JSON output
          shard0000.meta.json       # shard plan + command outcome
        result.json                 # written once, after a verified merge
        error.log                   # captured stderr (private, per job)

Invariants
----------
* Every write goes through `atomic_write_json`: tmp file + fsync + rename,
  so an interrupted process can never leave a half-written record.
* A shard counts as done only when its output JSON parses and contains the
  exact frame range the plan assigned it. Re-running the batch re-validates
  existing shards instead of recomputing them.
* `result.json` is the single completion marker. Its existence (plus
  validation) means the job is finished; resume never touches it.
* Status transitions are content-addressed by shard index, so duplicate
  deliveries (e.g. two resume runs racing the same shard) cannot create
  duplicate results: a shard file has one deterministic filename.
"""

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

# Status vocabulary.
STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_SKIPPED = "skipped"

TERMINAL_STATUSES = (STATUS_DONE, STATUS_SKIPPED)


def atomic_write_json(path, payload):
    path = str(path)
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=directory)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return default


@dataclass
class JobState:
    store: "CheckpointStore"
    job_id: str

    @property
    def dir(self):
        return self.store.job_dir(self.job_id)

    @property
    def status_path(self):
        return os.path.join(self.dir, "status.json")

    @property
    def result_path(self):
        return os.path.join(self.dir, "result.json")

    def shard_dir(self):
        d = os.path.join(self.dir, "shards")
        os.makedirs(d, exist_ok=True)
        return d

    def shard_output_path(self, shard_index):
        return os.path.join(self.shard_dir(),
                            "shard%04d.json" % shard_index)

    def shard_meta_path(self, shard_index):
        return os.path.join(self.shard_dir(),
                            "shard%04d.meta.json" % shard_index)

    def error_log_path(self):
        return os.path.join(self.dir, "error.log")

    def get_status(self):
        data = read_json(self.status_path, default=None)
        if data is None:
            return {"status": STATUS_PENDING, "attempts": 0, "shards": {}}
        return data

    def set_status(self, status, **extra):
        current = self.get_status()
        current.update(extra)
        current["status"] = status
        atomic_write_json(self.status_path, current)
        return current

    def record_attempt(self):
        current = self.get_status()
        current["attempts"] = int(current.get("attempts", 0)) + 1
        current["status"] = STATUS_RUNNING
        atomic_write_json(self.status_path, current)
        return current["attempts"]

    def record_shard(self, shard_index, plan, ok, detail=""):
        current = self.get_status()
        shards = current.setdefault("shards", {})
        key = str(shard_index)
        # Deterministic overwrite: same key, no duplicate entries possible.
        shards[key] = {
            "ok": bool(ok),
            "emit_start": plan["emit_start"],
            "emit_end": plan["emit_end"],
            "detail": detail,
        }
        current["status"] = STATUS_RUNNING
        atomic_write_json(self.status_path, current)

    def is_complete(self):
        """Completion = validated merged result exists."""
        result = read_json(self.result_path, default=None)
        return isinstance(result, dict) and result.get("status") == STATUS_DONE


class CheckpointStore:
    def __init__(self, root):
        self.root = Path(root)
        self.jobs_root = self.root / "jobs"
        self.jobs_root.mkdir(parents=True, exist_ok=True)

    def job_dir(self, job_id):
        d = self.jobs_root / job_id
        d.mkdir(parents=True, exist_ok=True)
        return str(d)

    def job_state(self, job_id):
        return JobState(self, job_id)

    def write_manifest_snapshot(self, payload):
        atomic_write_json(self.root / "manifest.json", payload)

    def write_job_spec(self, job):
        state = self.job_state(job.id)
        atomic_write_json(os.path.join(state.dir, "job.json"), job.to_dict())

    def list_job_ids(self):
        if not self.jobs_root.exists():
            return []
        return sorted(p.name for p in self.jobs_root.iterdir() if p.is_dir())

    def summary(self):
        """Aggregate status counts across all known jobs."""
        counts = {STATUS_PENDING: 0, STATUS_RUNNING: 0, STATUS_DONE: 0,
                  STATUS_FAILED: 0, STATUS_SKIPPED: 0}
        for job_id in self.list_job_ids():
            state = self.job_state(job_id)
            if state.is_complete():
                counts[STATUS_DONE] += 1
                continue
            status = state.get_status().get("status", STATUS_PENDING)
            counts[status] = counts.get(status, 0) + 1
        return counts
