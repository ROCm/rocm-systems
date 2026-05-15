/*
Copyright (c) 2022 - present Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef RDC_MODULES_RDC_ROCP_RDCROCPBASE_H_
#define RDC_MODULES_RDC_ROCP_RDCROCPBASE_H_
#include <rocprofiler-sdk/agent.h>

#include <cstdint>
#include <map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "rdc/rdc.h"
#include "rdc_lib/RdcTelemetryLibInterface.h"
#include "rdc_modules/rdc_rocp/RdcRocpCounterSampler.h"

namespace amd {
namespace rdc {

/// Common interface for RocP tests and samples
class RdcRocpBase {
 public:
  RdcRocpBase();
  RdcRocpBase(const RdcRocpBase&) = default;
  RdcRocpBase(RdcRocpBase&&) = delete;
  RdcRocpBase& operator=(const RdcRocpBase&) = delete;
  RdcRocpBase& operator=(RdcRocpBase&&) = delete;
  ~RdcRocpBase();

  /**
   * @brief Lookup ROCProfiler counter
   *
   * @param[in] gpu_field GPU_ID and FIELD_ID of requested metric
   * @param[out] value A pointer that will be populated with returned value
   *
   * @retval ::ROCMTOOLS_STATUS_SUCCESS The function has been executed
   * successfully.
   */
  rdc_status_t rocp_lookup(rdc_gpu_field_t gpu_field, rdc_field_value_data* value,
                           rdc_field_type_t* type);

  /**
   * @brief Bulk lookup of multiple ROCProfiler counters for a single GPU
   *
   * @param[in] fields Vector of fields to lookup (all for the same GPU)
   * @param[out] values Vector to be populated with returned values
   * @param[out] types Vector to be populated with returned types
   * @param[out] statuses Vector to be populated with status for each field
   *
   * @retval ::RDC_ST_OK The function has been executed successfully.
   */
  rdc_status_t rocp_lookup_bulk(const std::vector<rdc_gpu_field_t>& fields,
                                std::vector<rdc_field_value_data>& values,
                                std::vector<rdc_field_type_t>& types,
                                std::vector<rdc_status_t>& statuses);

  const char* get_field_id_from_name(rdc_field_t);
  const std::vector<rdc_field_t> get_field_ids();

 protected:
 private:
  typedef std::pair<uint32_t, rdc_field_t> rdc_field_pair_t;
  /**
   * @brief Tweak this to change for how long each metric is collected
   */
  static const uint32_t collection_duration_us_k = 10000;

  /**
   * @brief By default all profiler values are read as doubles
   */
  rdc_status_t run_profiler(uint32_t agent_index, rdc_field_t field, double* value);

  /**
   * @description Create a map from entity_id to profiler agent_index.
   * This is required due to different structure and ordering.
   * Populates entity_to_prof_map.
   */
  rdc_status_t map_entity_to_profiler();

  void init_rocp_if_not();

  std::vector<rocprofiler_agent_v0_t> agents = {};
  std::vector<std::shared_ptr<CounterSampler>> samplers = {};
  std::map<rdc_field_t, const char*> field_to_metric = {};
  std::map<uint32_t, uint32_t> entity_to_prof_map = {};

  bool m_is_initialized = false;

  // these fields must be divided by time passed
  std::unordered_set<rdc_field_t> eval_fields = {
      RDC_FI_PROF_EVAL_MEM_R_BW,         RDC_FI_PROF_EVAL_MEM_W_BW,
      RDC_FI_PROF_EVAL_FLOPS_16,         RDC_FI_PROF_EVAL_FLOPS_32,
      RDC_FI_PROF_EVAL_FLOPS_64,         RDC_FI_PROF_EVAL_FLOPS_16_PERCENT,
      RDC_FI_PROF_EVAL_FLOPS_32_PERCENT, RDC_FI_PROF_EVAL_FLOPS_64_PERCENT,
  };

  /**
   * @brief Apply field-specific transformations to raw profiler values
   *
   * @param[in] field Field ID to transform
   * @param[in] agent_index Index of the agent/GPU
   * @param[in] raw_value Raw value from profiler
   * @param[in] elapsed_time_ms Elapsed time in milliseconds (for eval fields)
   * @param[in] sampled_values Map of all sampled values (for fields needing multiple metrics)
   * @param[out] output Transformed output value
   * @param[out] type Output type
   *
   * @retval ::RDC_ST_OK Transformation successful
   */
  rdc_status_t apply_field_transformation(rdc_field_t field, uint32_t agent_index,
                                          double raw_value, double elapsed_time_ms,
                                          const std::map<std::string, double>& sampled_values,
                                          rdc_field_value_data* output,
                                          rdc_field_type_t* type);

  /**
   * @brief Convert from profiler status into RDC status
   */
  rdc_status_t Rocp2RdcError(rocprofiler_status_t status);
};

}  // namespace rdc
}  // namespace amd

#endif  // RDC_MODULES_RDC_ROCP_RDCROCPBASE_H_
