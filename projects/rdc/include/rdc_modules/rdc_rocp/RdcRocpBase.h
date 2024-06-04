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
#include <vector>

#include "rdc/rdc.h"

namespace amd {
namespace rdc {

typedef struct {
  hsa_agent_t* agents;
  unsigned count;
  unsigned capacity;
} hsa_agent_arr_t;

/// Common interface for RocP tests and samples
class RdcRocpBase {
  // typedef const char* rocp_metric_name_t;

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
   * @param[in] field An existing field already added to sessions dictionary
   * @param[out] value A pointer that will be populated with returned value
   *
   * @retval ::ROCMTOOLS_STATUS_SUCCESS The function has been executed
   * successfully.
   */
  rdc_status_t rocp_lookup(uint32_t gpu_index, rdc_field_t field, double* value);
  const char* get_field_id_from_name(rdc_field_t);
  const std::vector<rdc_field_t> get_field_ids();

 protected:
 private:
  std::map<const char*, double> metric_to_value = {};
  // array of features for each device
  std::map<uint32_t, rocprofiler_feature_t> feature;
  // rocprofiler_feature_t features[dev_count][features_count] = {};
  void read_feature(rocprofiler_t* context, const unsigned feature_count);
  int run_profiler(uint32_t gpu_index, rdc_field_t field);
  std::vector<hsa_queue_t*> queues;
  hsa_agent_arr_t agent_arr = {};
  std::map<rdc_field_t, const char*> field_to_metric = {};

  /**
   * @brief Convert from rocmtools status into RDC status
   */
  rdc_status_t Rocp2RdcError(hsa_status_t status);
};

}  // namespace rdc
}  // namespace amd

#endif  // RDC_MODULES_RDC_ROCP_RDCROCPBASE_H_
