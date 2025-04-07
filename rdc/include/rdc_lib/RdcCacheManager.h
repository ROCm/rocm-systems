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
#ifndef INCLUDE_RDC_LIB_RDCCACHEMANAGER_H_
#define INCLUDE_RDC_LIB_RDCCACHEMANAGER_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "rdc/rdc.h"
#include "rdc_lib/rdc_common.h"

namespace amd {
namespace rdc {

class RdcCacheManager {
 public:
  virtual rdc_status_t rdc_field_get_latest_value(uint32_t gpu_index, rdc_field_t field,
                                                  rdc_field_value* value) = 0;
  virtual rdc_status_t rdc_field_get_value_since(uint32_t gpu_index, rdc_field_t field,
                                                 uint64_t since_time_stamp,
                                                 uint64_t* next_since_time_stamp,
                                                 rdc_field_value* value) = 0;
  virtual rdc_status_t rdc_update_cache(uint32_t gpu_index, const rdc_field_value& value) = 0;
  virtual rdc_status_t evict_cache(uint32_t gpu_index, rdc_field_t field_id,
                                   uint64_t max_keep_samples, double max_keep_age) = 0;
  virtual std::string get_cache_stats() = 0;

  virtual rdc_status_t rdc_job_get_stats(const char job_id[64], const rdc_gpu_gauges_t& gpu_gauges,
                                         rdc_job_info_t* p_job_info) = 0;
  virtual rdc_status_t rdc_job_start_stats(const char job_id[64], const rdc_group_info_t& group,
                                           const rdc_field_group_info_t& finfo,
                                           const rdc_gpu_gauges_t& gpu_gauges) = 0;
  virtual rdc_status_t rdc_job_stop_stats(const char job_id[64],
                                          const rdc_gpu_gauges_t& gpu_gauge) = 0;
  virtual rdc_status_t rdc_update_job_stats(uint32_t gpu_index, const std::string& job_id,
                                            const rdc_field_value& value) = 0;
  virtual rdc_status_t rdc_job_remove(const char job_id[64]) = 0;
  virtual rdc_status_t rdc_job_remove_all() = 0;

  virtual rdc_status_t rdc_health_set(rdc_gpu_group_t group_id,
                                      uint32_t gpu_index,
                                      const rdc_field_value& value) = 0;
  virtual rdc_status_t rdc_health_get_values(rdc_gpu_group_t group_id,
                                             uint32_t gpu_index,
                                             rdc_field_t field_id,
                                             uint64_t start_timestamp,
                                             uint64_t end_timestamp,
                                             rdc_field_value* start_value,
                                             rdc_field_value* end_value) = 0;
  virtual rdc_status_t rdc_health_clear(rdc_gpu_group_t group_id) = 0;
  virtual rdc_status_t rdc_update_health_stats(rdc_gpu_group_t group_id,
                                               uint32_t gpu_index,
                                               const rdc_field_value& value) = 0;

  virtual ~RdcCacheManager() {}
};

typedef std::shared_ptr<RdcCacheManager> RdcCacheManagerPtr;

}  // namespace rdc
}  // namespace amd

#endif  // INCLUDE_RDC_LIB_RDCCACHEMANAGER_H_
