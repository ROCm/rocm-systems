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

#include "rdc_lib/impl/SmiUtils.h"

#include <cstdint>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "rdc/rdc.h"
#include "rdc_lib/RdcLogger.h"

namespace amd {
namespace rdc {

rdc_status_t Smi2RdcError(amdsmi_status_t rsmi) {
  switch (rsmi) {
    case AMDSMI_STATUS_SUCCESS:
      return RDC_ST_OK;

    case AMDSMI_STATUS_INVAL:
      return RDC_ST_BAD_PARAMETER;

    case AMDSMI_STATUS_NOT_SUPPORTED:
      return RDC_ST_NOT_SUPPORTED;

    case AMDSMI_STATUS_NOT_FOUND:
      return RDC_ST_NOT_FOUND;

    case AMDSMI_STATUS_OUT_OF_RESOURCES:
      return RDC_ST_INSUFF_RESOURCES;

    case AMDSMI_STATUS_FILE_ERROR:
      return RDC_ST_FILE_ERROR;

    case AMDSMI_STATUS_NO_DATA:
      return RDC_ST_NO_DATA;

    case AMDSMI_STATUS_NO_PERM:
      return RDC_ST_PERM_ERROR;

    case AMDSMI_STATUS_BUSY:
    case AMDSMI_STATUS_UNKNOWN_ERROR:
    case AMDSMI_STATUS_INTERNAL_EXCEPTION:
    case AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS:
    case AMDSMI_STATUS_INIT_ERROR:
    case AMDSMI_STATUS_NOT_YET_IMPLEMENTED:
    case AMDSMI_STATUS_INSUFFICIENT_SIZE:
    case AMDSMI_STATUS_INTERRUPT:
    case AMDSMI_STATUS_UNEXPECTED_SIZE:
    case AMDSMI_STATUS_UNEXPECTED_DATA:
    case AMDSMI_STATUS_REFCOUNT_OVERFLOW:
    default:
      return RDC_ST_UNKNOWN_ERROR;
  }
}

amdsmi_status_t get_processor_handle_from_id(uint32_t gpu_id,
                                             amdsmi_processor_handle* processor_handle) {
  uint32_t socket_count;
  uint32_t processor_count;
  auto ret = amdsmi_get_socket_handles(&socket_count, nullptr);
  if (ret != AMDSMI_STATUS_SUCCESS) {
    return ret;
  }
  std::vector<amdsmi_socket_handle> sockets(socket_count);
  std::vector<amdsmi_processor_handle> all_processors{};
  ret = amdsmi_get_socket_handles(&socket_count, sockets.data());
  for (auto& socket : sockets) {
    ret = amdsmi_get_processor_handles(socket, &processor_count, nullptr);
    if (ret != AMDSMI_STATUS_SUCCESS) {
      return ret;
    }
    std::vector<amdsmi_processor_handle> processors(processor_count);
    ret = amdsmi_get_processor_handles(socket, &processor_count, processors.data());
    if (ret != AMDSMI_STATUS_SUCCESS) {
      return ret;
    }

    for (auto& processor : processors) {
      processor_type_t processor_type = {};
      ret = amdsmi_get_processor_type(processor, &processor_type);
      if (processor_type != AMD_GPU) {
        RDC_LOG(RDC_ERROR, "Expect AMD_GPU device type!");
        return AMDSMI_STATUS_NOT_SUPPORTED;
      }
      all_processors.push_back(processor);
    }
  }

  if (gpu_id >= all_processors.size()) {
    return AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS;
  }

  // Get processor handle from GPU id
  *processor_handle = all_processors[gpu_id];

  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t get_processor_count(uint32_t& all_processor_count) {
  uint32_t total_processor_count = 0;
  uint32_t socket_count;
  auto ret = amdsmi_get_socket_handles(&socket_count, nullptr);
  if (ret != AMDSMI_STATUS_SUCCESS) {
    return ret;
  }
  std::vector<amdsmi_socket_handle> sockets(socket_count);
  ret = amdsmi_get_socket_handles(&socket_count, sockets.data());
  for (auto& socket : sockets) {
    uint32_t processor_count;
    ret = amdsmi_get_processor_handles(socket, &processor_count, nullptr);
    if (ret != AMDSMI_STATUS_SUCCESS) {
      return ret;
    }
    total_processor_count += processor_count;
  }
  all_processor_count = total_processor_count;
  return AMDSMI_STATUS_SUCCESS;
}

}  // namespace rdc
}  // namespace amd
