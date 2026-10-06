// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "mock.h"

amdsmi_status_t amdsmi_get_gpu_kfd_info(amdsmi_processor_handle h, amdsmi_kfd_info_t* out) {
  amdsmi_status_t status = mock_begin(__func__, h);
  if (status != AMDSMI_STATUS_SUCCESS) return mock_finish(status);
  if (!out || !mock_is_zero(out, sizeof(*out))) return mock_finish(AMDSMI_STATUS_UNEXPECTED_DATA);
  out->kfd_id = UINT64_MAX;
  out->node_id = UINT32_MAX;
  out->current_partition_id = 7;
  return mock_finish(AMDSMI_STATUS_SUCCESS);
}

amdsmi_status_t amdsmi_get_gpu_memory_partition_config(amdsmi_processor_handle h,
                                                       amdsmi_memory_partition_config_t* out) {
  amdsmi_status_t status = mock_begin(__func__, h);
  if (status != AMDSMI_STATUS_SUCCESS) return mock_finish(status);
  if (!out || !mock_is_zero(out, sizeof(*out))) return mock_finish(AMDSMI_STATUS_UNEXPECTED_DATA);
  out->partition_caps.nps_cap_mask = 5;
  out->mp_mode = AMDSMI_MEMORY_PARTITION_NPS4;
  if (mock_mode(__func__) == 1) {
    out->num_numa_ranges = AMDSMI_MAX_NUM_NUMA_NODES;
    for (uint32_t i = 0; i < out->num_numa_ranges; ++i) {
      out->numa_range[i].memory_type = AMDSMI_VRAM_TYPE_HBM3;
      out->numa_range[i].start = (UINT64_C(1) << 40) + i;
      out->numa_range[i].end = UINT64_MAX - i;
    }
  } else if (mock_mode(__func__) == 2) {
    out->num_numa_ranges = AMDSMI_MAX_NUM_NUMA_NODES + 1;
  } else if (mock_mode(__func__) == 3) {
    out->partition_caps.nps_cap_mask = UINT32_C(0x80000005);
  }
  return mock_finish(AMDSMI_STATUS_SUCCESS);
}

amdsmi_status_t amdsmi_get_gpu_accelerator_partition_profile(
    amdsmi_processor_handle h, amdsmi_accelerator_partition_profile_t* out,
    uint32_t* partition_id) {
  amdsmi_status_t status = mock_begin(__func__, h);
  if (status != AMDSMI_STATUS_SUCCESS) return mock_finish(status);
  if (!out || !mock_is_zero(out, sizeof(*out)) || !partition_id ||
      !mock_is_zero(partition_id, AMDSMI_MAX_ACCELERATOR_PARTITIONS * sizeof(*partition_id)))
    return mock_finish(AMDSMI_STATUS_UNEXPECTED_DATA);
  out->profile_type = AMDSMI_ACCELERATOR_PARTITION_CPX;
  out->num_partitions = AMDSMI_MAX_ACCELERATOR_PARTITIONS;
  out->memory_caps.nps_cap_mask = 5;
  out->profile_index = 3;
  *partition_id = 7;
  if (mock_mode(__func__) == 5) {
    for (uint32_t i = 1; i < AMDSMI_MAX_ACCELERATOR_PARTITIONS; ++i) {
      partition_id[i] = i;
    }
  }
  if (mock_mode(__func__) == 1) {
    out->num_resources = AMDSMI_MAX_CP_PROFILE_RESOURCES;
    for (uint32_t i = 0; i < out->num_partitions; ++i) {
      for (uint32_t j = 0; j < out->num_resources; ++j) {
        out->resources[i][j] = UINT32_MAX - i * out->num_resources - j;
      }
    }
  } else if (mock_mode(__func__) == 2) {
    out->num_partitions = AMDSMI_MAX_ACCELERATOR_PARTITIONS + 1;
    out->num_resources = 1;
  } else if (mock_mode(__func__) == 3) {
    out->num_resources = AMDSMI_MAX_CP_PROFILE_RESOURCES + 1;
  } else if (mock_mode(__func__) == 4) {
    out->num_partitions = UINT32_MAX;
    out->profile_index = UINT32_MAX;
  } else if (mock_mode(__func__) == 6) {
    out->num_partitions = AMDSMI_MAX_ACCELERATOR_PARTITIONS + 1;
  }
  return mock_finish(AMDSMI_STATUS_SUCCESS);
}
