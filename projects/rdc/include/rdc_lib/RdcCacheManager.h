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
#ifndef RDC_LIB_RDCCACHEMANAGER_H_
#define RDC_LIB_RDCCACHEMANAGER_H_

#include <memory>
#include <utility>
#include <string>
#include <vector>
#include <map>
#include "rdc_lib/rdc_common.h"
#include "rdc/rdc.h"

namespace amd {
namespace rdc {
typedef std::map<uint32_t, uint64_t> rdc_gpu_total_memory_t;

class RdcCacheManager {
 public:
    virtual rdc_status_t rdc_field_get_latest_value(uint32_t gpu_index,
                uint32_t field, rdc_field_value* value) = 0;
    virtual rdc_status_t rdc_field_get_value_since(uint32_t gpu_index,
                uint32_t field, uint64_t since_time_stamp,
                uint64_t *next_since_time_stamp, rdc_field_value* value) = 0;
    virtual rdc_status_t rdc_update_cache(uint32_t gpu_index,
                const rdc_field_value& value) = 0;
    virtual rdc_status_t evict_cache(uint32_t gpu_index, uint32_t field_id,
                uint64_t max_keep_samples, double  max_keep_age) = 0;
    virtual std::string  get_cache_stats() = 0;

    virtual rdc_status_t rdc_job_get_stats(char jobId[64],
        const rdc_gpu_total_memory_t& total_memory,
        rdc_job_info_t* p_job_info) = 0;
    virtual rdc_status_t rdc_job_start_stats(char jobId[64],
        const rdc_group_info_t& group,
        const rdc_field_group_info_t& finfo) = 0;
    virtual rdc_status_t rdc_job_stop_stats(char job_id[64]) = 0;
    virtual rdc_status_t rdc_update_job_stats(uint32_t gpu_index,
        const std::string& job_id, const rdc_field_value& value) = 0;
    virtual rdc_status_t rdc_job_remove(char job_id[64]) = 0;
    virtual rdc_status_t rdc_job_remove_all() = 0;

    virtual ~RdcCacheManager() {}
};

typedef std::shared_ptr<RdcCacheManager> RdcCacheManagerPtr;

//<! The key to identify the field with <gpu_id, field_id>
typedef std::pair<uint32_t, uint32_t> RdcFieldKey;

}  // namespace rdc
}  // namespace amd

#endif  // RDC_LIB_RDCCACHEMANAGER_H_
