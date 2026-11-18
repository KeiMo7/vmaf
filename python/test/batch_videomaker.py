"""
Synthetic y4m fixtures for batch integration tests.

Tiny 420p8 clips generated in pure Python — no ffmpeg or network needed.
Distorted clips are produced by deterministic per-frame pixel perturbation
so VMAF scores are stable across runs but differ measurably from the
reference.
"""

import os

Y4M_MAGIC = b"YUV4MPEG2"


def write_y4m(path, width, height, frame_count, pixel_fn, frame_tag=""):
    """
    pixel_fn(frame_index, plane) -> bytes for the raw plane.
    plane: 0 = Y (w*h), 1 = U (w/2*h/2), 2 = V (same).
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    header = ("YUV4MPEG2 W%d H%d F30:1 Ip A1:1 C420\n" %
              (width, height)).encode("ascii")
    with open(path, "wb") as f:
        f.write(header)
        for i in range(frame_count):
            f.write(b"FRAME" + (frame_tag.encode() if frame_tag else b"")
                    + b"\n")
            f.write(pixel_fn(i, 0))
            f.write(pixel_fn(i, 1))
            f.write(pixel_fn(i, 2))
    return path


def _gradient_plane(size, frame_index, scale=1, offset=0):
    return bytes(((x + frame_index * scale + offset) % 256
                  for x in range(size)))


def make_reference(path, width=64, height=48, frames=24):
    y_size = width * height
    c_size = (width // 2) * (height // 2)

    def pixel_fn(i, plane):
        if plane == 0:
            return _gradient_plane(y_size, i, scale=7)
        if plane == 1:
            return _gradient_plane(c_size, i, scale=3, offset=16)
        return _gradient_plane(c_size, i, scale=5, offset=128)

    return write_y4m(path, width, height, frames, pixel_fn)


def make_distorted(path, width=64, height=48, frames=24, noise=12,
                   corrupt_after=None):
    """
    noise: amplitude of deterministic quantization-like perturbation.
    corrupt_after: if set, frames from this index contain a stronger
                   distortion, simulating a visible quality drop.
    """
    y_size = width * height
    c_size = (width // 2) * (height // 2)

    def pixel_fn(i, plane):
        base_scale = (7, 3, 5)[plane]
        base_offset = (0, 16, 128)[plane]
        size = y_size if plane == 0 else c_size
        amp = noise * 3 if (corrupt_after is not None
                            and i >= corrupt_after) else noise
        return bytes(
            ((x + i * base_scale + base_offset
              + ((x * 13 + i * 7) % (2 * amp + 1)) - amp) % 256)
            for x in range(size))

    return write_y4m(path, width, height, frames, pixel_fn)


def make_broken_y4m(path):
    """A file whose magic is wrong, i.e. the decoder must reject it."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"NOTAVIDEO\n")
        f.write(os.urandom(256))
    return path
