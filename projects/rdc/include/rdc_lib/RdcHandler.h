/*
Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#ifndef INCLUDE_RDC_LIB_RDCHANDLER_H_
#define INCLUDE_RDC_LIB_RDCHANDLER_H_

#include "rdc_lib/rdc_common.h"
#include "rdc/rdc.h"

namespace amd {
namespace rdc {

// Interface class
class RdcHandler {
 public:
    // Job API
    virtual rdc_status_t rdc_job_start_stats(rdc_gpu_group_t groupId,
                             const char job_id[64], uint64_t update_freq) = 0;
    virtual rdc_status_t rdc_job_get_stats(const char jobId[64],
                rdc_job_info_t* p_job_info)= 0;
    virtual rdc_status_t rdc_job_stop_stats(const char job_id[64]) = 0;
    virtual rdc_status_t rdc_job_remove(const char job_id[64]) = 0;
    virtual rdc_status_t rdc_job_remove_all() = 0;

    // Discovery API
    virtual rdc_status_t rdc_device_get_all(
        uint32_t gpu_index_list[RDC_MAX_NUM_DEVICES], uint32_t* count) = 0;
    virtual rdc_status_t rdc_device_get_attributes(uint32_t gpu_index,
        rdc_device_attributes_t* p_rdc_attr) = 0;

    // Group API
    virtual rdc_status_t rdc_group_gpu_create(rdc_group_type_t type,
                const char* group_name, rdc_gpu_group_t* p_rdc_group_id) = 0;
    virtual rdc_status_t rdc_group_gpu_add(rdc_gpu_group_t groupId,
                uint32_t gpu_index) = 0;
    virtual rdc_status_t rdc_group_field_create(uint32_t num_field_ids,
        rdc_field_t* field_ids, const char* field_group_name,
        rdc_field_grp_t* rdc_field_group_id) = 0;
    virtual rdc_status_t rdc_group_field_get_info(
        rdc_field_grp_t rdc_field_group_id,
        rdc_field_group_info_t* field_group_info) = 0;
    virtual rdc_status_t rdc_group_gpu_get_info(
        rdc_gpu_group_t p_rdc_group_id, rdc_group_info_t* p_rdc_group_info) = 0;
    virtual rdc_status_t rdc_group_get_all_ids(
            rdc_gpu_group_t group_id_list[], uint32_t* count) = 0;
    virtual rdc_status_t rdc_group_field_get_all_ids(
            rdc_field_grp_t field_group_id_list[], uint32_t* count) = 0;
    virtual rdc_status_t rdc_group_gpu_destroy(
        rdc_gpu_group_t p_rdc_group_id) = 0;
    virtual rdc_status_t rdc_group_field_destroy(
        rdc_field_grp_t rdc_field_group_id) = 0;

    // Field API
    virtual rdc_status_t rdc_field_watch(rdc_gpu_group_t group_id,
        rdc_field_grp_t field_group_id, uint64_t update_freq,
        double max_keep_age, uint32_t max_keep_samples) = 0;
    virtual rdc_status_t rdc_field_get_latest_value(uint32_t gpu_index,
        rdc_field_t field, rdc_field_value* value) = 0;
    virtual rdc_status_t rdc_field_get_value_since(uint32_t gpu_index,
        rdc_field_t field, uint64_t since_time_stamp,
        uint64_t *next_since_time_stamp, rdc_field_value* value) = 0;
    virtual rdc_status_t rdc_field_unwatch(rdc_gpu_group_t group_id,
        rdc_field_grp_t field_group_id) = 0;

    // Control API
    virtual rdc_status_t rdc_field_update_all(uint32_t wait_for_update) = 0;

    virtual ~RdcHandler(){}
};

}  // namespace rdc
}  // namespace amd

#endif  // INCLUDE_RDC_LIB_RDCHANDLER_H_
