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
#include "rdc_lib/impl/RdcCacheManagerImpl.h"
#include <sys/time.h>
#include <ctime>
#include "rdc_lib/rdc_common.h"


namespace amd {
namespace rdc {

rdc_status_t RdcCacheManagerImpl::rdc_get_field_value_since(
    uint32_t gpu_index, uint32_t field_id, uint64_t since_time_stamp,
    uint64_t *next_since_time_stamp, rdc_field_value* value) {
    if (!next_since_time_stamp || !value) {
        return RDC_ST_BAD_PARAMETER;
    }

    std::lock_guard<std::mutex> guard(cache_mutex_);
    RdcFieldKey field{gpu_index, field_id};
    auto cache_samples_ite = cache_samples_.find(field);
    if (cache_samples_ite == cache_samples_.end() ||
             cache_samples_ite->second.size() == 0) {
        return RDC_ST_NOT_FOUND;
    }

    // TODO(bill_liu): Optimize it using the binary search
    auto cache_values = cache_samples_ite->second;
    for (auto cache_value=cache_values.begin();
                cache_value != cache_values.end(); cache_value++) {
        if ( cache_value->last_time >= since_time_stamp ) {
            // move to next potential timestamp
            auto next_iter = std::next(cache_value);
            if (next_iter != cache_values.end()) {
                *next_since_time_stamp = next_iter->last_time;
            } else {  // Last item, set it to the future by adding 1us
                *next_since_time_stamp = cache_value->last_time + 1;
            }
            value->ts = cache_value->last_time;
            value->type = INTEGER;
            value->value.l_int = cache_value->value;
            value->field_id = field_id;
            return RDC_ST_OK;
        }
    }

    *next_since_time_stamp = since_time_stamp;
    return RDC_ST_NOT_FOUND;
}


rdc_status_t RdcCacheManagerImpl::evict_cache(uint32_t gpu_index,
        uint32_t field_id, uint64_t max_keep_samples, double  max_keep_age) {
    std::lock_guard<std::mutex> guard(cache_mutex_);

    RdcFieldKey field{gpu_index, field_id};
    auto cache_samples_ite = cache_samples_.find(field);
    if (cache_samples_ite == cache_samples_.end() ||
             cache_samples_ite->second.size() == 0) {
        return RDC_ST_NOT_FOUND;
    }

    // Check max_keep_samples
    auto& cache_values = cache_samples_ite->second;
    int item_remove = cache_values.size() - max_keep_samples;
    if (item_remove > 0) {
       cache_values.erase(cache_values.begin(),
                cache_values.begin()+item_remove);
    }

    // Check max_keep_age
    struct timeval  tv;
    gettimeofday(&tv, NULL);
    uint64_t now = static_cast<uint64_t>(tv.tv_sec) * 1000 + tv.tv_usec / 1000;

    auto ite = cache_values.begin();
    while (ite != cache_values.end()) {
        if (ite->last_time + max_keep_age*1000 >= now) {
            break;
        } else {
            ite = cache_values.erase(ite);
        }
    }

    return RDC_ST_OK;
}

rdc_status_t RdcCacheManagerImpl::rdc_get_latest_value_for_field(
    uint32_t gpu_index, uint32_t field_id, rdc_field_value* value) {
    if (!value) {
        return RDC_ST_BAD_PARAMETER;
    }

    std::lock_guard<std::mutex> guard(cache_mutex_);
    RdcFieldKey field{gpu_index, field_id};
    auto cache_samples_ite = cache_samples_.find(field);
    if (cache_samples_ite == cache_samples_.end() ||
             cache_samples_ite->second.size() == 0) {
        return RDC_ST_NOT_FOUND;
    }

    auto& cache_value = cache_samples_ite->second.back();
    value->ts = cache_value.last_time;
    value->type = INTEGER;
    value->value.l_int = cache_value.value;
    value->field_id = field_id;

    return RDC_ST_OK;
}

uint32_t RdcCacheManagerImpl::get_cache_size() {
    uint32_t cache_size = 0;
    std::lock_guard<std::mutex> guard(cache_mutex_);

    auto cache_samples_ite = cache_samples_.begin();
    for (; cache_samples_ite != cache_samples_.end(); cache_samples_ite++) {
        cache_size+=cache_samples_ite->second.size();
    }
    return cache_size;
}

rdc_status_t RdcCacheManagerImpl::rdc_update_cache(uint32_t gpu_index,
        const rdc_field_value& value) {
    RdcCacheEntry entry;
    entry.last_time = value.ts;
    if (value.type == INTEGER) {
        entry.value = value.value.l_int;
    } else {
        return RDC_ST_NOT_SUPPORTED;
    }

    std::lock_guard<std::mutex> guard(cache_mutex_);
    RdcFieldKey field{gpu_index, value.field_id};
    auto cache_samples_ite = cache_samples_.find(field);
    if (cache_samples_ite == cache_samples_.end()) {
        std::vector<RdcCacheEntry> ve;
        ve.push_back(entry);
        cache_samples_.insert({field, ve});
    } else {
        cache_samples_ite->second.push_back(entry);
    }

    return RDC_ST_OK;
}

}  // namespace rdc
}  // namespace amd
