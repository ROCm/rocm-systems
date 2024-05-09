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

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <list>
#include <map>
#include <string>
#include <typeinfo>
#include <unordered_map>
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
  static const int dev_count = 1;
  typedef std::pair<uint32_t, rdc_field_t> pair_gpu_field_t;

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
  rdc_status_t rocp_lookup(pair_gpu_field_t gpu_field, double* value);
  const char* get_field_id_from_name(rdc_field_t);
  const std::vector<rdc_field_t> get_field_ids();

 protected:
 private:
  rocprofiler_t* contexts[dev_count] = {nullptr};
  static const int features_count = 1;
  std::map<const char*, double> metrics = {};
  rocprofiler_feature_t features[dev_count][features_count] = {};
  void read_features(rocprofiler_t* context, const unsigned feature_count);
  int run_profiler(const char* feature_name);
  hsa_queue_t* queues[dev_count] = {nullptr};
  hsa_agent_arr_t agent_arr = {};
  std::map<rdc_field_t, const char*> counter_map_k = {};

  /**
   * @brief Convert from rocmtools status into RDC status
   */
  rdc_status_t Rocp2RdcError(hsa_status_t rocm_status);
};

}  // namespace rdc
}  // namespace amd

#endif  // RDC_MODULES_RDC_ROCP_RDCROCPBASE_H_
