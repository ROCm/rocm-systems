/*
Copyright (c) 2021 - present Advanced Micro Devices, Inc. All rights reserved.

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
#ifndef INCLUDE_RDC_LIB_IMPL_RDCSMIDIAGNOSTICIMPL_H_
#define INCLUDE_RDC_LIB_IMPL_RDCSMIDIAGNOSTICIMPL_H_
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "rdc/rdc.h"

namespace amd {
namespace rdc {

class RdcSmiDiagnosticImpl {
 public:
  RdcSmiDiagnosticImpl();

  rdc_status_t check_smi_process_info(uint32_t gpu_index[RDC_MAX_NUM_DEVICES], uint32_t gpu_count,
                                      rdc_diag_test_result_t* result);
  rdc_status_t check_smi_topo_info(uint32_t gpu_index[RDC_MAX_NUM_DEVICES], uint32_t gpu_count,
                                   rdc_diag_test_result_t* result);
  rdc_status_t check_smi_param_info(uint32_t gpu_index[RDC_MAX_NUM_DEVICES], uint32_t gpu_count,
                                    rdc_diag_test_result_t* result);

 private:
  // What the GPU parameters check found on one GPU.
  struct GpuLimits {
    uint32_t compared = 0;  // limits compared with the current value
    uint32_t crossed = 0;   // limits the current value is at or beyond
    rdc_diag_result_t status = RDC_DIAG_RESULT_PASS;
    std::vector<std::string> messages;  // a sentence per crossed limit

    void compare(const std::string& sensor, int64_t current, int64_t limit, bool upper,
                 const char* name, rdc_diag_result_t level);
  };

  void check_temperature_level(amdsmi_processor_handle processor_handle,
                               amdsmi_temperature_type_t type, GpuLimits* gpu) const;
  std::string get_temperature_string(amdsmi_temperature_type_t type) const;

  void check_voltage_level(amdsmi_processor_handle processor_handle, amdsmi_voltage_type_t type,
                           GpuLimits* gpu) const;
  std::string get_voltage_string(amdsmi_voltage_type_t type) const;
};

typedef std::shared_ptr<RdcSmiDiagnosticImpl> RdcSmiDiagnosticPtr;

}  // namespace rdc
}  // namespace amd
#endif  // INCLUDE_RDC_LIB_IMPL_RDCSMIDIAGNOSTICIMPL_H_
