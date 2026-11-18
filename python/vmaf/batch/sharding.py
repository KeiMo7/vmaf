"""
Time sharding for a single video pair.

Why a naive split is wrong
--------------------------
libvmaf's temporal feature extractors carry state across frames:

* integer motion compares frame i against frame i-1 (or i-2 in five-frame
  window mode);
* the motion flush post-processing also looks *ahead* one frame
  (motion2 uses the SAD at i+1) and keeps a two-tap moving average
  (motion3 = blend(motion2_i) + previous processed value);
* so a score at frame i can depend on frames [i-2, i+1].

A shard therefore processes a wider window than it emits:

    feed window  : [shard_start - WARMUP, shard_end + LOOKAHEAD)
    emit window  : [shard_start, shard_end)

Warm-up frames are fed into the VMAF context (never skipped, skipping would
drop the temporal history) with their true global frame indices via
`--frame_index_offset`, and the output writer is told to only emit the
shard's own frames via `--emit_start/--emit_cnt`. Because every emitted
frame sees exactly the same neighbor frames as in an unsharded run, the
merged per-frame scores are bit-identical to whole-file processing.
"""

# Max backward dependency, in frames, across the built-in extractors.
WARMUP_FRAMES = 2
# Max forward dependency (motion2's hi_idx = i + 1).
LOOKAHEAD_FRAMES = 1


def plan_shards(total_frames, shard_frames,
                warmup=WARMUP_FRAMES, lookahead=LOOKAHEAD_FRAMES):
    """
    Return a list of shard dicts covering [0, total_frames).

    Each dict contains:
      index            shard ordinal
      emit_start       first global frame this shard owns
      emit_end         exclusive end of owned frames
      emit_cnt         emit_end - emit_start
      feed_start       first frame fed into the context (>= 0)
      feed_cnt         number of frames fed (warm-up + owned + look-ahead)
      frame_index_offset  global index assigned to the first fed frame
    """
    if total_frames <= 0:
        return []
    if shard_frames <= 0:
        shard_frames = total_frames

    shards = []
    shard_index = 0
    for emit_start in range(0, total_frames, shard_frames):
        emit_end = min(emit_start + shard_frames, total_frames)
        feed_start = max(0, emit_start - warmup)
        feed_end = min(total_frames, emit_end + lookahead)
        shards.append({
            "index": shard_index,
            "emit_start": emit_start,
            "emit_end": emit_end,
            "emit_cnt": emit_end - emit_start,
            "feed_start": feed_start,
            "feed_end": feed_end,
            "feed_cnt": feed_end - feed_start,
            "frame_index_offset": feed_start,
        })
        shard_index += 1
    return shards


def find_missing_shards(shard_plans, present_shard_ids):
    """Gap localization: return planned shards whose result is absent."""
    return [s for s in shard_plans if s["index"] not in set(present_shard_ids)]


def shard_output_filename(job_id, shard_index):
    return "%s.shard%04d.json" % (job_id, shard_index)
