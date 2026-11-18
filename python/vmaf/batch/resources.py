"""
Resource discovery and three-dimensional (CPU / memory / GPU) budget
accounting for the batch scheduler.

The scheduler only ever admits a shard when *all three* dimensions have
enough free budget. Every value can be overridden from the command line so
the same code path runs on machines without cgroup limits or GPUs.
"""

import json
import os
from dataclasses import dataclass

try:
    # Linux-only; the package degrades gracefully elsewhere.
    import resource
except ImportError:  # pragma: no cover - non-Linux platforms
    resource = None


def _read_int(path):
    try:
        with open(path) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def detect_cpu_count():
    """Usable CPU count, honoring cgroup quotas before sched_getaffinity."""
    quota = _read_int("/sys/fs/cgroup/cpu.max")
    # cgroup v2: "max 100000" or "50000 100000"
    if quota is not None:
        pass
    cpu_max = None
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:
            parts = f.read().strip().split()
        if parts[0] != "max":
            cpu_max = int(parts[0]) / max(int(parts[1]), 1)
    except (OSError, ValueError, IndexError):
        cpu_max = None

    cfs_quota = _read_int("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    cfs_period = _read_int("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if cpu_max is None and cfs_quota and cfs_period and cfs_quota > 0:
        cpu_max = cfs_quota / cfs_period

    affinity = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    host_cpus = affinity or 1
    if cpu_max is not None:
        return max(1, int(min(host_cpus, cpu_max)))
    return host_cpus


def detect_memory_bytes():
    """Usable RAM in bytes, honoring cgroup memory limits."""
    mem_limit = None
    for path in ("/sys/fs/cgroup/memory.max",
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        val = _read_int(path)
        if val is not None and val < (1 << 62):  # kernel sentinel for "max"
            mem_limit = val
            break

    host_mem = None
    try:
        meminfo = {}
        with open("/proc/meminfo") as f:
            for line in f:
                key, rest = line.split(":", 1)
                meminfo[key] = int(rest.strip().split()[0]) * 1024
        host_mem = meminfo.get("MemTotal")
    except (OSError, ValueError):
        host_mem = None

    if mem_limit and host_mem:
        return min(mem_limit, host_mem)
    return mem_limit or host_mem or 0


def detect_gpu_count():
    """
    Count usable GPUs without requiring any third-party SDK.

    The authoritative source is CUDA_PATH / the NVIDIA procfs layout; on
    machines without a driver this returns 0 and the scheduler runs CPU-only.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible == "":  # explicitly hidden
        return 0
    if visible is not None:
        return len([p for p in visible.split(",") if p.strip()])

    count = 0
    while os.path.exists("/dev/nvidia%d" % count):
        count += 1
    return count


@dataclass
class ResourceBudget:
    """Hard limits for a whole batch run."""
    cpu_total: int
    mem_bytes_total: int
    gpu_total: int
    # Fraction of each dimension a single shard may reserve at most.
    mem_per_shard_bytes: int = 0
    gpu_mem_bytes_total: int = 0

    @classmethod
    def autodetect(cls, cpu_limit=0, mem_limit_mb=0, gpu_limit=0,
                   mem_per_shard_mb=0):
        cpu_total = cpu_limit or detect_cpu_count()
        mem_total = (mem_limit_mb * 1024 * 1024) or detect_memory_bytes()
        gpu_total = gpu_limit if gpu_limit else detect_gpu_count()
        mem_per_shard = (mem_per_shard_mb * 1024 * 1024) or int(mem_total * 0.25)
        return cls(cpu_total=cpu_total,
                   mem_bytes_total=mem_total,
                   gpu_total=gpu_total,
                   mem_per_shard_bytes=mem_per_shard)

    def as_dict(self):
        return {
            "cpu_total": self.cpu_total,
            "mem_bytes_total": self.mem_bytes_total,
            "gpu_total": self.gpu_total,
            "mem_per_shard_bytes": self.mem_per_shard_bytes,
        }


class ResourcePool:
    """
    Reservation-based 3D resource pool.

    A shard declares (cpu_threads, mem_bytes, gpus). It is admitted only when
    every dimension fits. Reservations are released in `release()`, so a
    crashed worker never leaks budget as long as the scheduler observes its
    termination.
    """

    def __init__(self, budget: ResourceBudget):
        self.budget = budget
        self.cpu_used = 0
        self.mem_used = 0
        self.gpu_used = 0

    def can_admit(self, cpu_threads, mem_bytes, gpus):
        if cpu_threads <= 0:
            return False
        if self.cpu_used + cpu_threads > self.budget.cpu_total:
            return False
        if self.mem_used + mem_bytes > self.budget.mem_bytes_total:
            return False
        if gpus and self.budget.gpu_total == 0:
            return False
        if self.gpu_used + gpus > self.budget.gpu_total:
            return False
        return True

    def admit(self, cpu_threads, mem_bytes, gpus):
        if not self.can_admit(cpu_threads, mem_bytes, gpus):
            return False
        self.cpu_used += cpu_threads
        self.mem_used += mem_bytes
        self.gpu_used += gpus
        return True

    def release(self, cpu_threads, mem_bytes, gpus):
        self.cpu_used = max(0, self.cpu_used - cpu_threads)
        self.mem_used = max(0, self.mem_used - mem_bytes)
        self.gpu_used = max(0, self.gpu_used - gpus)

    def utilization(self):
        return {
            "cpu": (self.cpu_used, self.budget.cpu_total),
            "mem_bytes": (self.mem_used, self.budget.mem_bytes_total),
            "gpu": (self.gpu_used, self.budget.gpu_total),
        }

    def headroom_cpu(self):
        return max(0, self.budget.cpu_total - self.cpu_used)


def current_rss_bytes():
    """RSS of this process tree is not tracked here; callers measure workers
    via /proc/<pid>/status instead. This helper is used for self-sizing."""
    if resource is None:
        return 0
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def available_system_memory_bytes():
    """Instantaneous free+reclaimable memory, used for pressure backoff."""
    try:
        meminfo = {}
        with open("/proc/meminfo") as f:
            for line in f:
                key, rest = line.split(":", 1)
                meminfo[key] = int(rest.strip().split()[0]) * 1024
        return meminfo.get("MemAvailable", 0)
    except (OSError, ValueError):
        return 0


def dump_budget(path, budget: ResourceBudget):
    with open(path, "w") as f:
        json.dump(budget.as_dict(), f, indent=2, sort_keys=True)
