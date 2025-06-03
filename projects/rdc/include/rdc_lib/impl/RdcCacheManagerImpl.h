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
#ifndef INCLUDE_RDC_LIB_IMPL_RDCCACHEMANAGERIMPL_H_
#define INCLUDE_RDC_LIB_IMPL_RDCCACHEMANAGERIMPL_H_

#include <map>
#include <memory>
#include <mutex>  // NOLINT(build/c++11)
#include <string>
#include <vector>

#include "rdc/rdc.h"
#include "rdc_lib/RdcCacheManager.h"
#include "rdc_lib/rdc_common.h"

#define HEALTH_MAX_KEEP_SAMPLES 300

namespace amd {
namespace rdc {

// Note, the .cc code relies on RdcCacheEntry only having plain-old-data
// types and arrays (no pointers). If a pointer is added, make sure to update
// any code that copies this structure.
struct RdcCacheEntry {
  uint64_t last_time;
  rdc_field_type_t type;
  rdc_field_value_data value;
};

typedef std::map<RdcFieldKey, std::vector<RdcCacheEntry>> RdcCacheSamples;

struct FieldSummaryStats {
  int64_t max_value;
  int64_t min_value;
  int64_t total_value;

  // Use Welford algorithm to calculate the standard deviations.
  // https://en.wikipedia.org/wiki/Standard_deviation#Rapid_calculation_methods
  // https://www.johndcook.com/blog/standard_deviation/
  double old_m;
  double old_s;
  double new_m;
  double new_s;

  uint64_t last_time;
  uint64_t count;
};

struct GpuSummaryStats {
  uint64_t energy_consumed;
  uint64_t energy_last_time;
  uint64_t ecc_correct_init;    // Init counter when job starts
  uint64_t ecc_uncorrect_init;  // Init counter when job starts
  std::map<uint32_t, FieldSummaryStats> field_summaries;
};

// Per job entry
struct RdcJobStatsCacheEntry {
  uint64_t start_time;
  uint64_t end_time;
  std::map<uint32_t, GpuSummaryStats> gpu_stats;

  uint32_t num_processes = 0;
  std::array<rdc_process_status_info_t, RDC_MAX_NUM_PROCESSES_STATUS> processes{};
  std::map<uint32_t, uint32_t> pid_to_index;
};

// <job_id, job_stats>
typedef std::map<std::string, RdcJobStatsCacheEntry> RdcJobStatsCache;

// <group_id, health_samples>
typedef std::map<rdc_gpu_group_t, RdcCacheSamples> RdcHealthStatsCache;

class RdcCacheManagerImpl : public RdcCacheManager {
 public:
  rdc_status_t rdc_field_get_latest_value(uint32_t gpu_index, rdc_field_t field,
                                          rdc_field_value* value) override;
  rdc_status_t rdc_field_get_value_since(uint32_t gpu_index, rdc_field_t field,
                                         uint64_t since_time_stamp, uint64_t* next_since_time_stamp,
                                         rdc_field_value* value) override;
  rdc_status_t rdc_update_cache(uint32_t gpu_index, const rdc_field_value& value) override;
  rdc_status_t evict_cache(uint32_t gpu_index, rdc_field_t field_id, uint64_t max_keep_samples,
                           double max_keep_age) override;
  std::string get_cache_stats() override;

  rdc_status_t rdc_job_get_stats(const char job_id[64], const rdc_gpu_gauges_t& gpu_gauges,
                                 rdc_job_info_t* p_job_info) override;
  rdc_status_t rdc_job_start_stats(const char job_id[64], const rdc_group_info_t& group,
                                   const rdc_field_group_info_t& finfo,
                                   const rdc_gpu_gauges_t& gpu_gauges) override;
  rdc_status_t rdc_job_stop_stats(const char job_id[64],
                                  const rdc_gpu_gauges_t& gpu_gauge) override;
  rdc_status_t rdc_update_job_stats(uint32_t gpu_index, const std::string& job_id,
                                    const rdc_field_value& value) override;
  rdc_status_t rdc_job_remove(const char job_id[64]) override;
  rdc_status_t rdc_job_remove_all() override;

  rdc_status_t rdc_health_set(rdc_gpu_group_t group_id, uint32_t gpu_index,
                              const rdc_field_value& value) override;
  rdc_status_t rdc_health_get_values(rdc_gpu_group_t group_id, uint32_t gpu_index,
                                     rdc_field_t field_id, uint64_t start_timestamp,
                                     uint64_t end_timestamp, rdc_field_value* start_value,
                                     rdc_field_value* end_value) override;
  rdc_status_t rdc_health_clear(rdc_gpu_group_t group_id) override;
  rdc_status_t rdc_update_health_stats(rdc_gpu_group_t group_id, uint32_t gpu_index,
                                       const rdc_field_value& value) override;

 private:
  void set_summary(const FieldSummaryStats& stats, rdc_stats_summary_t& gpu,
                   rdc_stats_summary_t& summary,  // NOLINT
                   unsigned int adjuster);
  void set_average_summary(rdc_stats_summary_t& summary,
                           uint32_t num_gpus);  // NOLINT
  RdcCacheSamples cache_samples_;
  RdcJobStatsCache cache_jobs_;
  RdcHealthStatsCache cache_health_;
  std::mutex cache_mutex_;
};

}  // namespace rdc
}  // namespace amd

#endif  // INCLUDE_RDC_LIB_IMPL_RDCCACHEMANAGERIMPL_H_
