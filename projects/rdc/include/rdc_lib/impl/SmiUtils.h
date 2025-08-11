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

#ifndef INCLUDE_RDC_LIB_IMPL_RSMIUTILS_H_
#define INCLUDE_RDC_LIB_IMPL_RSMIUTILS_H_

#include <vector>

#include "amd_smi/amdsmi.h"
#include "rdc/rdc.h"

namespace amd {
namespace rdc {

rdc_status_t Smi2RdcError(amdsmi_status_t rsmi);
amdsmi_status_t get_processor_handle_from_id(uint32_t gpu_id,
                                             amdsmi_processor_handle* processor_handle);
amdsmi_status_t get_gpu_id_from_processor_handle(amdsmi_processor_handle processor_handle,
                                                 uint32_t* gpu_index);
amdsmi_status_t get_processor_count(uint32_t& all_processor_count);
amdsmi_status_t get_socket_handles(std::vector<amdsmi_socket_handle>& sockets);
amdsmi_status_t get_processor_handles(amdsmi_socket_handle socket,
                                      std::vector<amdsmi_processor_handle>& processors);
amdsmi_status_t get_kfd_partition_id(amdsmi_processor_handle proc, uint32_t* partition_id);
amdsmi_status_t get_metrics_info(amdsmi_processor_handle proc, amdsmi_gpu_metrics_t* metrics);
amdsmi_status_t get_num_partition(uint32_t index, uint16_t* num_partition);

}  // namespace rdc
}  // namespace amd

#endif  // INCLUDE_RDC_LIB_IMPL_RSMIUTILS_H_
