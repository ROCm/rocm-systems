/*
Copyright (c) 2024 - present Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef RDC_MODULES_RDC_ROCP_RDCROCPSUBPROCESS_H_
#define RDC_MODULES_RDC_ROCP_RDCROCPSUBPROCESS_H_

#include <atomic>
#include <chrono>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <unordered_set>
#include <vector>

#include "rdc/rdc.h"
#include "rdc_lib/RdcTelemetryLibInterface.h"

namespace amd {
namespace rdc {

struct RocpctlGpuMetric {
  std::string field;
  double value;
};

struct RocpctlGpuResult {
  std::string gpu_id;
  std::string drm_render_id;
  std::string logical_node_id;
  std::vector<RocpctlGpuMetric> metrics;
};

struct RocpctlResult {
  std::vector<RocpctlGpuResult> gpu_metrics;
};

class RdcRocpSubprocess {
 public:
  RdcRocpSubprocess();
  ~RdcRocpSubprocess() = default;

  RdcRocpSubprocess(const RdcRocpSubprocess&) = delete;
  RdcRocpSubprocess& operator=(const RdcRocpSubprocess&) = delete;
  RdcRocpSubprocess(RdcRocpSubprocess&&) = delete;
  RdcRocpSubprocess& operator=(RdcRocpSubprocess&&) = delete;

  rdc_status_t rocp_lookup(rdc_gpu_field_t gpu_field, rdc_field_value_data* value,
                           rdc_field_type_t* type);

  rdc_status_t rocp_lookup_bulk(const std::vector<rdc_gpu_field_t>& fields,
                                std::vector<rdc_field_value_data>& values,
                                std::vector<rdc_field_type_t>& types,
                                std::vector<rdc_status_t>& statuses);

  const char* get_field_id_from_name(rdc_field_t);
  const std::vector<rdc_field_t> get_field_ids();

 private:
  static constexpr uint32_t kSubprocessTimeoutSec = 30;
  static constexpr uint32_t kFailThreshold = 3;
  static constexpr uint32_t kCollectionDurationUs = 10000;
  static constexpr uint32_t kDefaultPtlDelayMs = 10;
  static constexpr auto kCacheTTL = std::chrono::seconds(10);

  std::string rocpctl_path_;
  std::map<rdc_field_t, const char*> field_to_metric_;

  std::unordered_set<rdc_field_t> eval_fields_;

  std::atomic<int> consecutive_failures_{0};
  std::atomic<bool> fatal_failure_{false};

  struct CachedResult {
    std::map<std::string, std::map<std::string, double>> gpu_metrics;
    std::chrono::steady_clock::time_point timestamp;
    double elapsed_time_ms = 0.0;
  };
  std::mutex cache_mutex_;
  CachedResult cached_result_;

  std::string find_rocpctl_binary() const;

  rdc_status_t exec_rocpctl(const std::vector<std::string>& metric_names,
                            RocpctlResult& result);

  rdc_status_t parse_json_output(const std::string& json_str, RocpctlResult& result);

  rdc_status_t get_or_refresh_cache(const std::vector<std::string>& metrics);

  rdc_status_t apply_field_transformation(rdc_field_t field, double raw_value,
                                          double elapsed_time_ms,
                                          const std::map<std::string, double>& sampled_values,
                                          rdc_field_value_data* output,
                                          rdc_field_type_t* type);

  bool is_disabled_on_failure() const;
  void inc_failure_count();
  void reset_failure_count();
  void set_fatal_failure();
};

}  // namespace rdc
}  // namespace amd

#endif  // RDC_MODULES_RDC_ROCP_RDCROCPSUBPROCESS_H_
