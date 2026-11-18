#ifndef __VMAF_CLI_PARSE_H__
#define __VMAF_CLI_PARSE_H__

#include <stdbool.h>
#include <stdint.h>

#include "libvmaf/libvmaf.h"
#include "libvmaf/model.h"
#include "libvmaf/feature.h"

#define CLI_SETTINGS_STATIC_ARRAY_LEN 32

typedef struct {
    const char *name;
    VmafFeatureDictionary *opts_dict;
    void *buf;
} CLIFeatureConfig;

typedef struct {
    const char *path;
    const char *version;
    VmafModelConfig cfg;
    struct {
        const char *name;
        VmafFeatureDictionary *opts_dict;
    } feature_overload[CLI_SETTINGS_STATIC_ARRAY_LEN];
    unsigned overload_cnt;
    void *buf;
} CLIModelConfig;

typedef struct {
    char *path_ref, *path_dist;
    unsigned frame_skip_ref;
    unsigned frame_skip_dist;
    unsigned frame_cnt;
    /*
     * Global frame index assigned to the first frame fed into the context.
     * Non-zero values are used by time-sharded batch runs: warm-up frames are
     * still fed into the context (so temporal feature extractors see the same
     * frame history as an unsharded run) but keep their true global indices.
     * Defaults to 0, i.e. the historical single-file behavior.
     */
    unsigned frame_index_offset;
    /*
     * Number of frames discarded from each input before the context is fed.
     * Unlike frame_skip_*, these position a time shard at its file offset;
     * the discarded frames never enter the context. Used together with
     * frame_index_offset so the first fed frame gets its true global index.
     */
    unsigned frame_offset;
    /*
     * Inclusive lower bound and count of frame indices written to the output
     * file. Frames outside the emit window are still processed (they act as
     * temporal context for neighboring shards) but never appear in the output.
     * emit_cnt == 0 means "emit every frame", preserving the old behavior.
     */
    unsigned emit_start;
    unsigned emit_cnt;
    /* the _ref and _dist sizes default to width and height */
    unsigned width, height;
    unsigned width_ref, height_ref, width_dist, height_dist;
    enum VmafPixelFormat pix_fmt;
    unsigned bitdepth;
    VmafColor color_ref, color_dist;
    bool use_yuv;
    char *output_path;
    enum VmafOutputFormat output_fmt;
    CLIModelConfig model_config[CLI_SETTINGS_STATIC_ARRAY_LEN];
    unsigned model_cnt;
    CLIFeatureConfig feature_cfg[CLI_SETTINGS_STATIC_ARRAY_LEN];
    unsigned feature_cnt;
    enum VmafLogLevel log_level;
    unsigned subsample;
    unsigned thread_cnt;
    bool no_prediction;
    bool quiet;
    bool common_bitdepth;
    unsigned cpumask;
    unsigned gpumask;
} CLISettings;

void cli_parse(const int argc, char *const *const argv,
               CLISettings *const settings);

void cli_free(CLISettings *settings);

#endif /* __VMAF_CLI_PARSE_H__ */
