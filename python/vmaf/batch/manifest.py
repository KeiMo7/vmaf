"""
Batch manifest (job list) model.

A manifest is a JSON document:

{
  "jobs": [
    {"id": "clipA-v1",
     "reference": "/data/src/clipA.y4m",
     "distorted": "/data/out/clipA-v1.y4m",
     "version": "v1",
     "content": "clipA",
     "options": {"threads": 2}},
    ...
  ]
}

Paths live only in the manifest and in per-job private artifacts. They are
deliberately excluded from the shared scheduler log: the log references jobs
by their opaque `id` only.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

_SAFE_ID_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_id(raw):
    """Map arbitrary user strings to filesystem-safe job ids."""
    cleaned = _SAFE_ID_RE.sub("_", str(raw)).strip("._-")
    return cleaned or "job"


@dataclass
class Job:
    id: str
    reference: str
    distorted: str
    version: str = "default"
    content: str = "default"
    # Optional raw-yuv metadata (mirrors the vmaf CLI flags).
    width: int = 0
    height: int = 0
    pixel_format: str = ""
    bitdepth: int = 0
    # Per-job overrides forwarded to the vmaf CLI.
    options: dict = field(default_factory=dict)
    # Set when a pre-flight probe proves the job cannot run.
    skip_reason: str = ""

    def validate(self):
        problems = []
        if not self.reference:
            problems.append("missing reference path")
        if not self.distorted:
            problems.append("missing distorted path")
        if not self.skip_reason:
            if self.reference and not Path(self.reference).is_file():
                problems.append("reference file not found")
            if self.distorted and not Path(self.distorted).is_file():
                problems.append("distorted file not found")
        return problems

    def to_dict(self):
        return {
            "id": self.id,
            "reference": self.reference,
            "distorted": self.distorted,
            "version": self.version,
            "content": self.content,
            "width": self.width,
            "height": self.height,
            "pixel_format": self.pixel_format,
            "bitdepth": self.bitdepth,
            "options": self.options,
            "skip_reason": self.skip_reason,
        }


def load_manifest(path):
    with open(path) as f:
        doc = json.load(f)
    if not isinstance(doc, dict) or not isinstance(doc.get("jobs"), list):
        raise ValueError("manifest must be an object with a 'jobs' array")

    jobs = []
    seen = set()
    for idx, entry in enumerate(doc["jobs"]):
        if not isinstance(entry, dict):
            raise ValueError("job #%d is not an object" % idx)
        raw_id = entry.get("id") or "%s__%s" % (
            entry.get("content", "content"),
            entry.get("version", idx))
        job_id = sanitize_id(raw_id)
        if job_id in seen:
            raise ValueError("duplicate job id: %s" % job_id)
        seen.add(job_id)

        jobs.append(Job(
            id=job_id,
            reference=entry.get("reference", ""),
            distorted=entry.get("distorted", ""),
            version=str(entry.get("version", "default")),
            content=str(entry.get("content", "default")),
            width=int(entry.get("width", 0) or 0),
            height=int(entry.get("height", 0) or 0),
            pixel_format=str(entry.get("pixel_format", "") or ""),
            bitdepth=int(entry.get("bitdepth", 0) or 0),
            options=dict(entry.get("options", {}) or {}),
            skip_reason=str(entry.get("skip_reason", "") or ""),
        ))
    return jobs


def count_y4m_frames(path):
    """Count FRAME markers without decoding pixel data.

    A text-line scan is unsafe here: raw plane bytes can contain 0x0A, which
    readline() treats as a line break, hiding later frame markers. The
    b"FRAME" token only ever starts a frame header in y4m, so a streaming
    byte count is exact.
    """
    count = 0
    marker = b"FRAME"
    with open(path, "rb") as f:
        header = f.readline()
        if not header.startswith(b"YUV4MPEG2"):
            raise ValueError("not a y4m file")
        tail = b""
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            data = tail + chunk
            count += data.count(marker)
            # Keep a possible partial marker spanning the chunk boundary.
            tail = data[-(len(marker) - 1):]
    return count


def y4m_header(path):
    """Parse the y4m stream header for dimension/pixfmt/bitdepth metadata."""
    with open(path, "rb") as f:
        header = f.readline().decode("ascii", "replace").split()
    if not header or header[0] != "YUV4MPEG2":
        raise ValueError("not a y4m file")
    info = {"width": 0, "height": 0, "pixel_format": "", "bitdepth": 8}
    for token in header[1:]:
        if token.startswith("W"):
            info["width"] = int(token[1:])
        elif token.startswith("H"):
            info["height"] = int(token[1:])
        elif token.startswith("C"):
            info["pixel_format"] = token[1:]
        elif token.startswith("b"):
            # e.g. b10 -> 10-bit
            digits = token[1:].rstrip("p")
            if digits.isdigit():
                info["bitdepth"] = int(digits)
    return info
