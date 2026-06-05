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
  const auto& table = get_flat_gpu_table();
  rdc_entity_info_t info = rdc_get_info_from_entity_index(gpu_id);

  if (info.instance_index == 0 && info.entity_role == RDC_DEVICE_ROLE_PHYSICAL) {
    uint32_t flat_idx = info.device_index;
    if (flat_idx >= table.size()) {
      return AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS;
    }
    *processor_handle = table[flat_idx].handle;
    return AMDSMI_STATUS_SUCCESS;
  }

  // Entity-encoded index (partition instances): use socket/proc mapping
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

  uint32_t socket_index = info.device_index;
  uint32_t instance_index = info.instance_index;

  if (socket_index >= socket_count) {
    return AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS;
  }

  uint32_t proc_count = 0;
  ret = amdsmi_get_processor_handles(sockets[socket_index], &proc_count, nullptr);
  if (ret != AMDSMI_STATUS_SUCCESS) {
    return ret;
  }

  if (instance_index >= proc_count) {
    return AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS;
  }

  std::vector<amdsmi_processor_handle> procs(proc_count);
  ret = amdsmi_get_processor_handles(sockets[socket_index], &proc_count, procs.data());
  if (ret != AMDSMI_STATUS_SUCCESS) {
    return ret;
  }

  *processor_handle = procs[instance_index];
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
  uint32_t socket_count = 0;
  auto ret = amdsmi_get_socket_handles(&socket_count, nullptr);
  if (ret != AMDSMI_STATUS_SUCCESS) {
    return ret;
  }
  std::vector<amdsmi_socket_handle> sockets(socket_count);
  ret = amdsmi_get_socket_handles(&socket_count, sockets.data());
  for (auto& socket : sockets) {
    uint32_t processor_count = 0;
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
  if (num_partition == nullptr) {
    return AMDSMI_STATUS_INVAL;
  }

  // Use flat table to find the physical GPU (first processor in the socket)
  const auto& table = get_flat_gpu_table();
  if (index >= table.size()) {
    return AMDSMI_STATUS_INPUT_OUT_OF_BOUNDS;
  }

  // Find the first processor in the same socket to query the physical GPU profile.
  uint32_t target_socket = table[index].socket_index;
  amdsmi_processor_handle phys_handle = nullptr;
  for (const auto& entry : table) {
    if (entry.socket_index == target_socket) {
      phys_handle = entry.handle;
      break;
    }
  }

  if (phys_handle == nullptr) {
    return AMDSMI_STATUS_NOT_FOUND;
  }

  amdsmi_accelerator_partition_profile_t profile;
  memset(&profile, 0, sizeof(profile));
  uint32_t partition_id{};

  amdsmi_status_t ret =
      amdsmi_get_gpu_accelerator_partition_profile(phys_handle, &profile, &partition_id);
  if (ret != AMDSMI_STATUS_SUCCESS) {
    return ret;
  }

  if (partition_id != 0) {
    return AMDSMI_STATUS_UNEXPECTED_DATA;
  }

  *num_partition = profile.num_partitions;

  return ret;
}


static std::vector<GpuHandleEntry> s_flat_gpu_table;
static bool s_flat_gpu_table_initialized = false;

static void build_flat_gpu_table() {
  s_flat_gpu_table.clear();

  uint32_t socket_count = 0;
  if (amdsmi_get_socket_handles(&socket_count, nullptr) != AMDSMI_STATUS_SUCCESS) {
    return;
  }
  std::vector<amdsmi_socket_handle> sockets(socket_count);
  if (amdsmi_get_socket_handles(&socket_count, sockets.data()) != AMDSMI_STATUS_SUCCESS) {
    return;
  }

  for (uint32_t s = 0; s < socket_count; s++) {
    uint32_t proc_count = 0;
    if (amdsmi_get_processor_handles(sockets[s], &proc_count, nullptr) != AMDSMI_STATUS_SUCCESS) {
      continue;
    }
    std::vector<amdsmi_processor_handle> procs(proc_count);
    if (amdsmi_get_processor_handles(sockets[s], &proc_count, procs.data()) != AMDSMI_STATUS_SUCCESS) {
      continue;
    }
    for (uint32_t p = 0; p < proc_count; p++) {
      processor_type_t proc_type = AMDSMI_PROCESSOR_TYPE_UNKNOWN;
      if (amdsmi_get_processor_type(procs[p], &proc_type) != AMDSMI_STATUS_SUCCESS) {
        continue;
      }
      if (proc_type == AMDSMI_PROCESSOR_TYPE_AMD_GPU) {
        GpuHandleEntry entry;
        entry.handle = procs[p];
        entry.socket_index = s;
        entry.proc_index = p;
        s_flat_gpu_table.push_back(entry);
      }
    }
  }
  s_flat_gpu_table_initialized = true;
}

const std::vector<GpuHandleEntry>& get_flat_gpu_table() {
  if (!s_flat_gpu_table_initialized) {
    build_flat_gpu_table();
  }
  return s_flat_gpu_table;
}

void reset_flat_gpu_table() {
  s_flat_gpu_table.clear();
  s_flat_gpu_table_initialized = false;
}

}  // namespace rdc
}  // namespace amd
