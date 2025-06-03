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
#include <cstring>
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

    case AMDSMI_STATUS_CORRUPTED_EEPROM:
      return RDC_ST_CORRUPTED_EEPROM;

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
  uint32_t socket_count = 0;
  amdsmi_status_t ret = amdsmi_get_socket_handles(&socket_count, nullptr);
  if (ret != AMDSMI_STATUS_SUCCESS) {
    return ret;
  }

  std::vector<amdsmi_socket_handle> sockets(socket_count);
  ret = amdsmi_get_socket_handles(&socket_count, sockets.data());
  if (ret != AMDSMI_STATUS_SUCCESS) {
    return ret;
  }

  std::vector<std::vector<amdsmi_processor_handle>> procs_by_socket;
  procs_by_socket.resize(socket_count);

  for (size_t s = 0; s < sockets.size(); s++) {
    uint32_t proc_count = 0;
    ret = amdsmi_get_processor_handles(sockets[s], &proc_count, nullptr);
    if (ret != AMDSMI_STATUS_SUCCESS) {
      return ret;
    }

    std::vector<amdsmi_processor_handle> procs(proc_count);
    ret = amdsmi_get_processor_handles(sockets[s], &proc_count, procs.data());
    if (ret != AMDSMI_STATUS_SUCCESS) {
      return ret;
    }

    for (auto& proc : procs) {
      processor_type_t proc_type = {};
      ret = amdsmi_get_processor_type(proc, &proc_type);
      if (proc_type != AMDSMI_PROCESSOR_TYPE_AMD_GPU) {
        return AMDSMI_STATUS_NOT_SUPPORTED;
      }
    }

    procs_by_socket[s] = procs;
  }

  rdc_entity_info_t info = rdc_get_info_from_entity_index(gpu_id);
  uint32_t socket_index = info.device_index;
  uint32_t instance_index = info.instance_index;

  if (socket_index >= procs_by_socket.size()) {
    return AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS;
  }

  const auto& handles = procs_by_socket[socket_index];
  if (instance_index >= handles.size()) {
    return AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS;
  }

  *processor_handle = handles[instance_index];
  return AMDSMI_STATUS_SUCCESS;
}

amdsmi_status_t get_gpu_id_from_processor_handle(amdsmi_processor_handle processor_handle,
                                                 uint32_t* gpu_index) {
  if (!gpu_index) {
    return AMDSMI_STATUS_INVAL;
  }

  std::vector<amdsmi_socket_handle> sockets;
  auto ret = get_socket_handles(sockets);
  if (ret != AMDSMI_STATUS_SUCCESS) {
    return ret;
  }

  uint32_t idx = 0;
  for (auto const& sock : sockets) {
    std::vector<amdsmi_processor_handle> procs;
    ret = get_processor_handles(sock, procs);
    if (ret != AMDSMI_STATUS_SUCCESS) {
      return ret;
    }
    for (auto const& h : procs) {
      if (h == processor_handle) {
        *gpu_index = idx;
        return AMDSMI_STATUS_SUCCESS;
      }
      ++idx;
    }
  }

  return AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS;
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

amdsmi_status_t get_socket_handles(std::vector<amdsmi_socket_handle>& sockets) {
  uint32_t socket_count = 0;
  amdsmi_status_t ret = amdsmi_get_socket_handles(&socket_count, nullptr);
  if (ret != AMDSMI_STATUS_SUCCESS) {
    return ret;
  }

  sockets.resize(socket_count);

  ret = amdsmi_get_socket_handles(&socket_count, sockets.data());

  return ret;
}

amdsmi_status_t get_processor_handles(amdsmi_socket_handle socket,
                                      std::vector<amdsmi_processor_handle>& processors) {
  uint32_t processor_count = 0;
  amdsmi_status_t ret = amdsmi_get_processor_handles(socket, &processor_count, nullptr);
  if (ret != AMDSMI_STATUS_SUCCESS) {
    return ret;
  }

  processors.resize(processor_count);

  ret = amdsmi_get_processor_handles(socket, &processor_count, processors.data());

  return ret;
}

amdsmi_status_t get_kfd_partition_id(amdsmi_processor_handle proc, uint32_t* partition_id) {
  amdsmi_kfd_info_t kfd_info = {};
  amdsmi_status_t ret = amdsmi_get_gpu_kfd_info(proc, &kfd_info);
  if (ret != AMDSMI_STATUS_SUCCESS) {
    return ret;
  }
  *partition_id = kfd_info.current_partition_id;
  return ret;
}

amdsmi_status_t get_metrics_info(amdsmi_processor_handle proc, amdsmi_gpu_metrics_t* metrics) {
  amdsmi_status_t ret = amdsmi_get_gpu_metrics_info(proc, metrics);
  return ret;
}

amdsmi_status_t get_num_partition(uint32_t index, uint16_t* num_partition) {
  // Get the processor handle for the physical device.
  amdsmi_processor_handle proc_handle;
  amdsmi_status_t ret = get_processor_handle_from_id(index, &proc_handle);
  if (ret != AMDSMI_STATUS_SUCCESS) {
    return ret;
  }

  amdsmi_gpu_metrics_t metrics;
  memset(&metrics, 0, sizeof(metrics));
  ret = get_metrics_info(proc_handle, &metrics);
  if (ret != AMDSMI_STATUS_SUCCESS) {
    return ret;
  }

  *num_partition = metrics.num_partition;

  return ret;
}

}  // namespace rdc
}  // namespace amd
