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
#include <rocprofiler/rocprofiler.h>

#include <cstdint>
#include <map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "rdc/rdc.h"
#include "rdc_lib/RdcTelemetryLibInterface.h"

namespace amd {
namespace rdc {

typedef struct {
  hsa_agent_t* agents;
  unsigned count;
  unsigned capacity;
} hsa_agent_arr_t;

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
  rdc_status_t rocp_lookup(rdc_gpu_field_t gpu_field, double* value);
  const char* get_field_id_from_name(rdc_field_t);
  const std::vector<rdc_field_t> get_field_ids();

 protected:
 private:
  typedef std::pair<uint32_t, rdc_field_t> rdc_field_pair_t;
  static const size_t buffer_length_k = 5;
  /**
   * @brief Tweak this to change for how long each metric is collected
   */
  static const uint32_t collection_duration_us_k = 10000;

  double read_feature(rocprofiler_t* context, uint32_t gpu_index);
  double run_profiler(uint32_t gpu_index, rdc_field_t field);

  hsa_agent_arr_t agent_arr = {};
  std::vector<hsa_queue_t*> queues;
  std::map<uint32_t, rocprofiler_feature_t> gpuid_to_feature;
  std::map<rdc_field_t, const char*> field_to_metric = {};

  // these fields must be divided by time passed
  std::unordered_set<rdc_field_t> eval_fields = {
      RDC_FI_PROF_EVAL_MEM_R_BW, RDC_FI_PROF_EVAL_MEM_W_BW, RDC_FI_PROF_EVAL_FLOPS_16,
      RDC_FI_PROF_EVAL_FLOPS_32, RDC_FI_PROF_EVAL_FLOPS_64,
  };

  /**
   * @brief Convert from rocmtools status into RDC status
   */
  rdc_status_t Rocp2RdcError(hsa_status_t status);
};

}  // namespace rdc
}  // namespace amd

#endif  // RDC_MODULES_RDC_ROCP_RDCROCPBASE_H_
