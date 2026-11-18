"""
Subprocess execution of the `vmaf` CLI for one shard.

This module is the only place that touches paths on disk. Its outputs are:
  * shard JSON (path inside the per-job state dir),
  * captured stderr in the per-job error log,
  * a structured ShardResult for the scheduler.

The shared scheduler log never receives file paths; only the job id, shard
index and exit classification are returned upward.
"""

import os
import shutil
import subprocess
import time
from dataclasses import dataclass

from .state import atomic_write_json, read_json

# Exit classification vocabulary.
EXIT_OK = "ok"
EXIT_DECODER = "decoder_unavailable"
EXIT_OOM = "out_of_memory"
EXIT_FAILED = "failed"

# Signals the kernel uses to kill a process on OOM / external abort.
_OOM_SIGNALS = {9: "SIGKILL", 15: "SIGTERM"}


@dataclass
class ShardResult:
    job_id: str
    shard_index: int
    outcome: str
    returncode: int
    duration_s: float
    peak_rss_bytes: int
    detail: str = ""


def find_vmaf_binary(override=None):
    if override:
        if not os.path.isfile(override) or not os.access(override, os.X_OK):
            raise FileNotFoundError("vmaf binary not executable: %s" % override)
        return override
    env_bin = os.environ.get("VMAF_BIN")
    if env_bin and os.path.isfile(env_bin) and os.access(env_bin, os.X_OK):
        return env_bin
    found = shutil.which("vmaf")
    if found:
        return found
    raise FileNotFoundError(
        "vmaf binary not found; set VMAF_BIN or pass --vmaf-bin")


def build_cli_command(vmaf_bin, job, shard, model_path, output_path,
                      threads, use_gpu=False):
    cmd = [
        vmaf_bin,
        "--reference", job.reference,
        "--distorted", job.distorted,
        "--json",
        "--output", output_path,
        "--model", "path=%s" % model_path,
        "--threads", str(threads),
        "--frame_offset", str(shard["feed_start"]),
        "--frame_cnt", str(shard["feed_cnt"]),
        "--frame_index_offset", str(shard["frame_index_offset"]),
        "--emit_start", str(shard["emit_start"]),
        "--emit_cnt", str(shard["emit_cnt"]),
        "--quiet",
    ]
    if use_gpu:
        # gpumask bit 0 selects the first GPU; the CLI default (~0) means CPU.
        cmd += ["--gpumask", "1"]

    # Raw YUV inputs need explicit geometry, mirroring the single-file CLI.
    if job.width and job.height:
        cmd += ["--width", str(job.width), "--height", str(job.height)]
        if job.pixel_format:
            cmd += ["--pixel_format", job.pixel_format]
        if job.bitdepth:
            cmd += ["--bitdepth", str(job.bitdepth)]

    for key, value in (job.options or {}).items():
        cmd += ["--%s" % key.lstrip("-"), str(value)]
    return cmd


def _peak_rss_of(pid):
    """Best-effort peak RSS (bytes) from /proc/<pid>/status."""
    try:
        with open("/proc/%d/status" % pid) as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def classify_failure(returncode, stderr_tail):
    """Map an exit code + stderr fragment to a retry/skip decision class."""
    if returncode == 0:
        return EXIT_OK
    text = stderr_tail.lower()
    if returncode < 0 and -returncode in (9, 15):
        return EXIT_OOM
    if "out of memory" in text or "cannot allocate memory" in text:
        return EXIT_OOM
    decoder_markers = (
        "problem with reference file",
        "problem with distorted file",
        "incomplete magic",
        "incorrect yuv input file version",
        "error parsing yuv4mpeg",
        "could not open file",
        "unsupported bitdepth",
        "pixel formats do not match",
    )
    if any(marker in text for marker in decoder_markers):
        return EXIT_DECODER
    return EXIT_FAILED


def run_shard(vmaf_bin, job, shard, model_path, output_path, error_log_path,
              threads, use_gpu=False, timeout_s=0):
    cmd = build_cli_command(vmaf_bin, job, shard, model_path, output_path,
                            threads, use_gpu=use_gpu)
    start = time.monotonic()
    peak_rss = 0
    try:
        with open(error_log_path, "ab") as errf:
            errf.write(("\n# shard %d attempt at %s\n" %
                        (shard["index"], time.strftime("%FT%T"))).encode())
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                    stderr=errf)
            while proc.poll() is None:
                peak_rss = max(peak_rss, _peak_rss_of(proc.pid))
                if timeout_s and (time.monotonic() - start) > timeout_s:
                    proc.kill()
                    proc.wait()
                    return ShardResult(job.id, shard["index"], EXIT_FAILED,
                                       proc.returncode,
                                       time.monotonic() - start, peak_rss,
                                       "timeout after %ds" % timeout_s)
                time.sleep(0.05)
            rc = proc.returncode
    except OSError as exc:  # e.g. binary vanished mid-run
        return ShardResult(job.id, shard["index"], EXIT_FAILED, -1,
                           time.monotonic() - start, peak_rss, str(exc))

    duration = time.monotonic() - start
    stderr_tail = ""
    try:
        with open(error_log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 4096))
            stderr_tail = f.read().decode("utf-8", "replace")
    except OSError:
        pass

    outcome = classify_failure(rc, stderr_tail)
    if rc == 0:
        # Validate the output before claiming success: a zero exit with a
        # truncated JSON must still count as failure.
        validated = validate_shard_output(output_path, shard)
        if validated is not None:
            outcome = EXIT_FAILED
            detail = validated
        else:
            detail = ""
    else:
        detail = "rc=%s %s" % (rc, _OOM_SIGNALS.get(-rc, "") if rc < 0 else "")

    meta = {
        "job_id": job.id,
        "shard_index": shard["index"],
        "plan": shard,
        "outcome": outcome,
        "returncode": rc,
        "duration_s": duration,
        "peak_rss_bytes": peak_rss,
        "detail": detail,
    }
    atomic_write_json(output_path + ".meta.json", meta)
    return ShardResult(job.id, shard["index"], outcome, rc, duration,
                       peak_rss, detail)


def validate_shard_output(path, shard):
    """Return None when the JSON covers exactly the shard's emit window."""
    doc = read_json(path, default=None)
    if not isinstance(doc, dict) or not isinstance(doc.get("frames"), list):
        return "shard output missing or unreadable"
    expected = list(range(shard["emit_start"], shard["emit_end"]))
    actual = [fr.get("frameNum") for fr in doc["frames"]]
    if actual != expected:
        return ("frame range mismatch: expected %s..%s, got %s" %
                (expected[0], expected[-1],
                 ("%s..%s" % (actual[0], actual[-1])) if actual else "empty"))
    return None
