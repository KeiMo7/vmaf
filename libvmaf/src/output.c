/**
 *
 *  Copyright 2016-2020 Netflix, Inc.
 *
 *     Licensed under the BSD+Patent License (the "License");
 *     you may not use this file except in compliance with the License.
 *     You may obtain a copy of the License at
 *
 *         https://opensource.org/licenses/BSDplusPatent
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License.
 *
 */

#include <errno.h>
#include <math.h>
#include <stdio.h>

#include "feature/alias.h"
#include "feature/feature_collector.h"

#include "libvmaf/libvmaf.h"

static unsigned max_capacity(VmafFeatureCollector *fc)
{
    unsigned capacity = 0;

    for (unsigned j = 0; j < fc->cnt; j++) {
        if (fc->feature_vector[j]->capacity > capacity)
            capacity = fc->feature_vector[j]->capacity;
    }

    return capacity;
}

/*
 * Emit window handling. emit_end == 0 means "no upper bound", which together
 * with emit_start == 0 reproduces the historical unsharded output exactly.
 */
static int frame_is_emitted(unsigned index, unsigned emit_start,
                            unsigned emit_end)
{
    if (index < emit_start)
        return 0;
    if (emit_end && index >= emit_end)
        return 0;
    return 1;
}

static unsigned pool_index_high(unsigned emit_end, unsigned pic_cnt)
{
    return emit_end ? emit_end - 1 : pic_cnt - 1;
}

/*
 * Find the highest frame index actually present in any feature vector.
 * Needed for sharded contexts, where occupied slots start at a global index
 * greater than zero so pic_cnt-1 cannot be used as the pool upper bound.
 */
static unsigned last_written_index(VmafFeatureCollector *fc)
{
    unsigned last = 0;
    for (unsigned j = 0; j < fc->cnt; j++) {
        FeatureVector *fv = fc->feature_vector[j];
        for (unsigned i = fv->capacity; i > 0; i--) {
            if (fv->score[i - 1].written) {
                if (i - 1 > last)
                    last = i - 1;
                break;
            }
        }
    }
    return last;
}

static unsigned first_written_index(VmafFeatureCollector *fc)
{
    unsigned first = 0;
    int found = 0;
    for (unsigned j = 0; j < fc->cnt; j++) {
        FeatureVector *fv = fc->feature_vector[j];
        for (unsigned i = 0; i < fv->capacity; i++) {
            if (fv->score[i].written) {
                if (!found || i < first)
                    first = i;
                found = 1;
                break;
            }
        }
    }
    return first;
}

static const char *pool_method_name[] = {
    [VMAF_POOL_METHOD_MIN] = "min",
    [VMAF_POOL_METHOD_MAX] = "max",
    [VMAF_POOL_METHOD_MEAN] = "mean",
    [VMAF_POOL_METHOD_HARMONIC_MEAN] = "harmonic_mean",
};

static int count_leading_zeros_d(double x)
{
    if(x < 0)
        x = fabs(x);

    int int_part = (int)x;
    double fractional_part = x - int_part;

    // Count leading zeroes in the fractional part
    int leading_zeros_count = 0;

    while (fractional_part < 1.0 && fractional_part != 0)
    {
        fractional_part *= 10; // Shift decimal point to the right
        leading_zeros_count++;
    }

    return leading_zeros_count;
}

int vmaf_write_output_xml(VmafContext *vmaf, VmafFeatureCollector *fc,
                          FILE *outfile, unsigned subsample, unsigned width,
                          unsigned height, double fps, unsigned pic_cnt,
                          unsigned emit_start, unsigned emit_end)
{
    if (!vmaf) return -EINVAL;
    if (!fc) return -EINVAL;
    if (!outfile) return -EINVAL;

    fprintf(outfile, "<VMAF version=\"%s\">\n", vmaf_version());
    fprintf(outfile, "  <params qualityWidth=\"%d\" qualityHeight=\"%d\" />\n",
            width, height);
    fprintf(outfile, "  <fyi fps=\"%.2f\" />\n", fps);

    unsigned n_frames = 0;
    int leading_zeros_count;
    fprintf(outfile, "  <frames>\n");
    for (unsigned i = 0 ; i < max_capacity(fc); i++) {
        if ((subsample > 1) && (i % subsample))
            continue;
        if (!frame_is_emitted(i, emit_start, emit_end))
            continue;

        unsigned cnt = 0;
        for (unsigned j = 0; j < fc->cnt; j++) {
            if (i > fc->feature_vector[j]->capacity)
                continue;
            if (fc->feature_vector[j]->score[i].written)
                cnt++;
        }
        if (!cnt) continue;

        fprintf(outfile, "    <frame frameNum=\"%d\" ", i);
        for (unsigned j = 0; j < fc->cnt; j++) {
            if (i > fc->feature_vector[j]->capacity)
                continue;
            if (!fc->feature_vector[j]->score[i].written)
                continue;
            leading_zeros_count = count_leading_zeros_d(fc->feature_vector[j]->score[i].value);
            if (leading_zeros_count <= 6)
                fprintf(outfile, "%s=\"%.6f\" ",
                    vmaf_feature_name_alias(fc->feature_vector[j]->name),
                    fc->feature_vector[j]->score[i].value);
            else
                fprintf(outfile, "%s=\"%.16f\" ",
                    vmaf_feature_name_alias(fc->feature_vector[j]->name),
                    fc->feature_vector[j]->score[i].value);
        }
        n_frames++;
        fprintf(outfile, "/>\n");
    }
    fprintf(outfile, "  </frames>\n");

    fprintf(outfile, "  <pooled_metrics>\n");
    const unsigned pool_lo = emit_start > first_written_index(fc)
                             ? emit_start : first_written_index(fc);
    const unsigned pool_hi = pool_index_high(emit_end, pic_cnt);
    const unsigned pool_hi_clamped = pool_hi < last_written_index(fc)
                                     ? pool_hi : last_written_index(fc);
    for (unsigned i = 0; i < fc->cnt; i++) {
        const char *feature_name = fc->feature_vector[i]->name;
        fprintf(outfile, "    <metric name=\"%s\" ",
                vmaf_feature_name_alias(feature_name));

        for (unsigned j = 1; j < VMAF_POOL_METHOD_NB; j++) {
            double score;
            int err = vmaf_feature_score_pooled(vmaf, feature_name, j, &score,
                                                pool_lo,
                                                pool_hi_clamped);
            if (!err)
            {
                leading_zeros_count = count_leading_zeros_d(score);
                if (leading_zeros_count <= 6)
                    fprintf(outfile, "%s=\"%.6f\" ", pool_method_name[j], score);
                else
                    fprintf(outfile, "%s=\"%.16f\" ", pool_method_name[j], score);
            }
        }
        fprintf(outfile, "/>\n");
    }
    fprintf(outfile, "  </pooled_metrics>\n");


    fprintf(outfile, "  <aggregate_metrics ");
    for (unsigned i = 0; i < fc->aggregate_vector.cnt; i++) {
        leading_zeros_count = count_leading_zeros_d(fc->aggregate_vector.metric[i].value);
        if (leading_zeros_count <= 6)
            fprintf(outfile, "%s=\"%.6f\" ",
                fc->aggregate_vector.metric[i].name,
                fc->aggregate_vector.metric[i].value);
        else
            fprintf(outfile, "%s=\"%.16f\" ",
                fc->aggregate_vector.metric[i].name,
                fc->aggregate_vector.metric[i].value);
    }
    fprintf(outfile, "/>\n");

    fprintf(outfile, "</VMAF>\n");

    return 0;
}

int vmaf_write_output_json(VmafContext *vmaf, VmafFeatureCollector *fc,
                           FILE *outfile, unsigned subsample, double fps,
                           unsigned pic_cnt,
                           unsigned emit_start, unsigned emit_end)
{
    int leading_zeros_count;
    fprintf(outfile, "{\n");
    fprintf(outfile, "  \"version\": \"%s\",\n", vmaf_version());
    switch(fpclassify(fps)) {
    case FP_NORMAL:
    case FP_ZERO:
    case FP_SUBNORMAL:
        fprintf(outfile, "  \"fps\": %.2f,\n", fps);
        break;
    case FP_INFINITE:
    case FP_NAN:
        fprintf(outfile, "  \"fps\": null,\n");
    }

    unsigned n_frames = 0;
    fprintf(outfile, "  \"frames\": [");
    for (unsigned i = 0 ; i < max_capacity(fc); i++) {
        if ((subsample > 1) && (i % subsample))
            continue;
        if (!frame_is_emitted(i, emit_start, emit_end))
            continue;

        unsigned cnt = 0;
        for (unsigned j = 0; j < fc->cnt; j++) {
            if (i > fc->feature_vector[j]->capacity)
                continue;
            if (fc->feature_vector[j]->score[i].written)
                cnt++;
        }
        if (!cnt) continue;
        fprintf(outfile, "%s", n_frames > 0 ? ",\n" : "\n");

        fprintf(outfile, "    {\n");
        fprintf(outfile, "      \"frameNum\": %d,\n", i);
        fprintf(outfile, "      \"metrics\": {\n");

        unsigned cnt2 = 0;
        for (unsigned j = 0; j < fc->cnt; j++) {
            if (i > fc->feature_vector[j]->capacity)
                continue;
            if (!fc->feature_vector[j]->score[i].written)
                continue;
            cnt2++;
            switch(fpclassify(fc->feature_vector[j]->score[i].value)) {
            case FP_NORMAL:
            case FP_ZERO:
            case FP_SUBNORMAL:
                leading_zeros_count = count_leading_zeros_d(fc->feature_vector[j]->score[i].value);
                if (leading_zeros_count <= 6)
                    fprintf(outfile, "        \"%s\": %.6f%s\n",
                        vmaf_feature_name_alias(fc->feature_vector[j]->name),
                        fc->feature_vector[j]->score[i].value,
                        cnt2 < cnt ? "," : "");
                else
                    fprintf(outfile, "        \"%s\": %.16f%s\n",
                        vmaf_feature_name_alias(fc->feature_vector[j]->name),
                        fc->feature_vector[j]->score[i].value,
                        cnt2 < cnt ? "," : "");

                break;
            case FP_INFINITE:
            case FP_NAN:
                fprintf(outfile, "        \"%s\": null%s",
                        vmaf_feature_name_alias(fc->feature_vector[j]->name),
                        cnt2 < cnt ? "," : "");
                break;
            }
        }
        fprintf(outfile, "      }\n");
        fprintf(outfile, "    }");
        n_frames++;
    }
    fprintf(outfile, "\n  ],\n");

    fprintf(outfile, "  \"pooled_metrics\": {");
    const unsigned json_pool_lo = emit_start > first_written_index(fc)
                                  ? emit_start : first_written_index(fc);
    const unsigned json_pool_hi =
        pool_index_high(emit_end, pic_cnt) < last_written_index(fc)
        ? pool_index_high(emit_end, pic_cnt) : last_written_index(fc);
    for (unsigned i = 0; i < fc->cnt; i++) {
        const char *feature_name = fc->feature_vector[i]->name;
        fprintf(outfile, "%s", i > 0 ? ",\n" : "\n");
        fprintf(outfile, "    \"%s\": {",
                vmaf_feature_name_alias(feature_name));
        for (unsigned j = 1; j < VMAF_POOL_METHOD_NB; j++) {
            double score;
            int err = vmaf_feature_score_pooled(vmaf, feature_name, j, &score,
                                                json_pool_lo,
                                                json_pool_hi);
            if (!err) {
                fprintf(outfile, "%s", j > 1 ? ",\n" : "\n");
                switch(fpclassify(score)) {
                case FP_NORMAL:
                case FP_ZERO:
                case FP_SUBNORMAL:
                    leading_zeros_count = count_leading_zeros_d((double)score);
                    if (leading_zeros_count <= 6)
                        fprintf(outfile, "      \"%s\": %.6f",
                            pool_method_name[j], score);
                    else
                        fprintf(outfile, "      \"%s\": %.16f",
                            pool_method_name[j], score);
                    break;
                case FP_INFINITE:
                case FP_NAN:
                    fprintf(outfile, "      \"%s\": null",
                            pool_method_name[j]);
                    break;
                }
            }
        }
        fprintf(outfile, "\n");
        fprintf(outfile, "    }");
    }
    fprintf(outfile, "\n  },\n");

    fprintf(outfile, "  \"aggregate_metrics\": {");
    for (unsigned i = 0; i < fc->aggregate_vector.cnt; i++) {
        switch(fpclassify(fc->aggregate_vector.metric[i].value)) {
        case FP_NORMAL:
        case FP_ZERO:
        case FP_SUBNORMAL:
            leading_zeros_count = count_leading_zeros_d(fc->aggregate_vector.metric[i].value);
            if (leading_zeros_count <= 6)
                fprintf(outfile, "\n    \"%s\": %.6f",
                    fc->aggregate_vector.metric[i].name,
                    fc->aggregate_vector.metric[i].value);
            else
                fprintf(outfile, "\n    \"%s\": %.16f",
                    fc->aggregate_vector.metric[i].name,
                    fc->aggregate_vector.metric[i].value);


            break;
        case FP_INFINITE:
        case FP_NAN:
            fprintf(outfile, "\n    \"%s\": null",
                    fc->aggregate_vector.metric[i].name);
            break;
        }
        fprintf(outfile, "%s", i < fc->aggregate_vector.cnt - 1 ? "," : "");
    }
    fprintf(outfile, "\n  }\n");
    fprintf(outfile, "}\n");

    return 0;
}

int vmaf_write_output_csv(VmafFeatureCollector *fc, FILE *outfile,
                           unsigned subsample,
                           unsigned emit_start, unsigned emit_end)
{
    int leading_zeros_count;
    fprintf(outfile, "Frame,");
    for (unsigned i = 0; i < fc->cnt; i++) {
        fprintf(outfile, "%s,",
                vmaf_feature_name_alias(fc->feature_vector[i]->name));
    }
    fprintf(outfile, "\n");

    for (unsigned i = 0 ; i < max_capacity(fc); i++) {
        if ((subsample > 1) && (i % subsample))
            continue;
        if (!frame_is_emitted(i, emit_start, emit_end))
            continue;

        unsigned cnt = 0;
        for (unsigned j = 0; j < fc->cnt; j++) {
            if (i > fc->feature_vector[j]->capacity)
                continue;
            if (fc->feature_vector[j]->score[i].written)
                cnt++;
        }
        if (!cnt) continue;

        fprintf(outfile, "%d,", i);
        for (unsigned j = 0; j < fc->cnt; j++) {
            if (i > fc->feature_vector[j]->capacity)
                continue;
            if (!fc->feature_vector[j]->score[i].written)
                continue;

            leading_zeros_count = count_leading_zeros_d(fc->feature_vector[j]->score[i].value);
            if (leading_zeros_count <= 6)
                fprintf(outfile, "%.6f,", fc->feature_vector[j]->score[i].value);
            else
                fprintf(outfile, "%.16f,", fc->feature_vector[j]->score[i].value);
        }
        fprintf(outfile, "\n");
    }

    return 0;
}

int vmaf_write_output_sub(VmafFeatureCollector *fc, FILE *outfile,
                          unsigned subsample,
                          unsigned emit_start, unsigned emit_end)
{
    int leading_zeros_count;
    for (unsigned i = 0 ; i < max_capacity(fc); i++) {
        if ((subsample > 1) && (i % subsample))
            continue;
        if (!frame_is_emitted(i, emit_start, emit_end))
            continue;

        unsigned cnt = 0;
        for (unsigned j = 0; j < fc->cnt; j++) {
            if (i > fc->feature_vector[j]->capacity)
                continue;
            if (fc->feature_vector[j]->score[i].written)
                cnt++;
        }
        if (!cnt) continue;

        fprintf(outfile, "{%d}{%d}frame: %d|", i, i + 1, i);
        for (unsigned j = 0; j < fc->cnt; j++) {
            if (i > fc->feature_vector[j]->capacity)
                continue;
            if (!fc->feature_vector[j]->score[i].written)
                continue;
            leading_zeros_count = count_leading_zeros_d(fc->feature_vector[j]->score[i].value);
            if (leading_zeros_count <= 6)
                fprintf(outfile, "%s: %.6f|",
                    vmaf_feature_name_alias(fc->feature_vector[j]->name),
                    fc->feature_vector[j]->score[i].value);
            else
                fprintf(outfile, "%s: %.16f|",
                    vmaf_feature_name_alias(fc->feature_vector[j]->name),
                    fc->feature_vector[j]->score[i].value);
        }
        fprintf(outfile, "\n");
    }

    return 0;
}
