"""
Defensive log sanitizer.

The scheduler's own messages only contain job ids, but subprocess output or
third-party libraries could embed absolute paths. This filter strips
Unix/Windows path-like tokens from every log record before it reaches a
shared handler, satisfying the "no content paths in public logs" rule.
"""

import logging
import re

# Matches /home/u/clip.y4m, ./x/y, C:\dir\file, /tmp/a-b/c.y4m ...
_PATH_RE = re.compile(
    r"(?:[A-Za-z]:)?(?:/|\\\\)[^\s'\"<>|]+?(?:\.[A-Za-z0-9]{1,8})?"
)


def redact(text):
    if not isinstance(text, str):
        return text
    return _PATH_RE.sub("<redacted-path>", text)


class PathRedactingFilter(logging.Filter):
    def filter(self, record):
        record.msg = redact(str(record.msg))
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: redact(str(v)) for k, v in record.args.items()}
            else:
                record.args = tuple(redact(str(a)) for a in record.args)
        return True


def configure_logging(verbose=False, log_file=None):
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler()
    handler.addFilter(PathRedactingFilter())
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger("vmaf.batch")
    root.handlers = [handler]
    root.setLevel(level)
    root.propagate = False

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.addFilter(PathRedactingFilter())
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(file_handler)
    return root
