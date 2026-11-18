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

#ifndef __VMAF_FEATURE_COLLECTOR_H__
#define __VMAF_FEATURE_COLLECTOR_H__

#include <pthread.h>
#include <stdbool.h>
#include <time.h>

#include "dict.h"
#include "model.h"
#include "metadata_handler.h"

typedef struct {
    char *name;
    struct {
        bool written;
        double value;
    } *score;
    unsigned capacity;
} FeatureVector;

typedef struct {
    struct {
        char *name;
        double value;
    } *metric;
    unsigned cnt, capacity;
} AggregateVector;

typedef struct VmafPredictModel {
    VmafModel *model;
    struct VmafPredictModel *next;
} VmafPredictModel;

typedef struct VmafFeatureCollector {
    FeatureVector **feature_vector;
    AggregateVector aggregate_vector;
    VmafCallbackList *metadata;
    VmafPredictModel *models;
    unsigned cnt, capacity;
    struct { clock_t begin, end; } timer;
    pthread_mutex_t lock;
} VmafFeatureCollector;

int vmaf_feature_collector_init(VmafFeatureCollector **const feature_collector);

int vmaf_feature_collector_mount_model(VmafFeatureCollector *feature_collector, VmafModel *model);

int vmaf_feature_collector_append(VmafFeatureCollector *feature_collector,
                                  const char *feature_name, double score,
                                  unsigned index);

int vmaf_feature_collector_register_metadata(VmafFeatureCollector *feature_collector,
                                             VmafMetadataConfiguration metadata_cfg);

int vmaf_feature_collector_append_with_dict(VmafFeatureCollector *fc,
        VmafDictionary *dict, const char *feature_name, double score,
        unsigned index);

int vmaf_feature_collector_get_score(VmafFeatureCollector *feature_collector,
                                     const char *feature_name, double *score,
                                     unsigned index);

/*
 * Return the range of written indices for `feature_name`.
 * On success 0 is returned, `*first_index` receives the lowest written
 * index and `*last_index` the highest. A feature with no written scores, or
 * an unknown feature name, returns -EINVAL. This lets flush post-processing
 * handle contexts whose first frame carries a non-zero global index
 * (time-sharded runs) without assuming scores start at index 0.
 */
int vmaf_feature_collector_get_written_range(
        VmafFeatureCollector *feature_collector, const char *feature_name,
        unsigned *first_index, unsigned *last_index);

int vmaf_feature_collector_set_aggregate(VmafFeatureCollector *feature_collector,
                                         const char *feature_name,
                                         double score);

int vmaf_feature_collector_get_aggregate(VmafFeatureCollector *feature_collector,
                                         const char *feature_name,
                                         double *score);

void vmaf_feature_collector_destroy(VmafFeatureCollector *feature_collector);

#endif /* __VMAF_FEATURE_COLLECTOR_H__ */
