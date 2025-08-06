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
#ifndef INCLUDE_RDC_LIB_IMPL_RDCROCPLIB_H_
#define INCLUDE_RDC_LIB_IMPL_RDCROCPLIB_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "rdc_lib/RdcLibraryLoader.h"
#include "rdc_lib/RdcTelemetry.h"

namespace amd {
namespace rdc {

class RdcRocpLib : public RdcTelemetry {
 public:
  /* Telemetry */

  // get support field ids
  rdc_status_t rdc_telemetry_fields_query(uint32_t field_ids[MAX_NUM_FIELDS],
                                          uint32_t* field_count) override;
  // Fetch
  rdc_status_t rdc_telemetry_fields_value_get(rdc_gpu_field_t* fields, uint32_t fields_count,
                                              rdc_field_value_f callback, void* user_data) override;
  rdc_status_t rdc_telemetry_fields_watch(rdc_gpu_field_t* fields, uint32_t fields_count) override;
  rdc_status_t rdc_telemetry_fields_unwatch(rdc_gpu_field_t* fields,
                                            uint32_t fields_count) override;
  RdcRocpLib();
  ~RdcRocpLib();

 private:
  RdcLibraryLoader lib_loader_;
  rdc_status_t (*telemetry_fields_query_)(uint32_t field_ids[MAX_NUM_FIELDS],
                                          uint32_t* field_count);
  rdc_status_t (*telemetry_fields_value_get_)(rdc_gpu_field_t* fields, uint32_t fields_count,
                                              rdc_field_value_f callback, void* user_data);
  rdc_status_t (*telemetry_fields_watch_)(rdc_gpu_field_t* fields, uint32_t fields_count);
  rdc_status_t (*telemetry_fields_unwatch_)(rdc_gpu_field_t* fields, uint32_t fields_count);

  rdc_status_t (*rdc_module_init_)(uint64_t);
  rdc_status_t (*rdc_module_destroy_)();
  /**
   * @brief Make sure HSA_TOOLS_LIB is not set as it breaks rocprofiler-sdk
   * @details
   * Rocprofilerv1 needed HSA_TOOLS_LIB set to librocprofiler64.so.1.
   * That breaks rocprofiler-sdk because it tries to load both v1 and sdk libraries.
   */
  void rdc_unset_hsa_tools_lib();
};

using RdcRocpLibPtr = std::shared_ptr<RdcRocpLib>;

}  // namespace rdc
}  // namespace amd

#endif  // INCLUDE_RDC_LIB_IMPL_RDCROCPLIB_H_
