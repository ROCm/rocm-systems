/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Regression test for graphics-aware available memory on discrete GPUs.
//
// The graphics-aware query is intended for conservative capacity planning.
// When it is supported, the reported available VRAM must never exceed the
// agent's total global pool size. Platforms without the query may skip it.

#include <stdint.h>

#include "gtest/gtest.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

namespace {

struct LargestGlobalPool {
  uint64_t size = 0;
};

hsa_status_t FindLargestGlobalPool(hsa_amd_memory_pool_t pool, void* data) {
  hsa_amd_segment_t segment;
  if (hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT,
                                   &segment) != HSA_STATUS_SUCCESS) {
    return HSA_STATUS_SUCCESS;
  }
  if (segment != HSA_AMD_SEGMENT_GLOBAL) {
    return HSA_STATUS_SUCCESS;
  }

  size_t pool_size = 0;
  if (hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SIZE,
                                   &pool_size) == HSA_STATUS_SUCCESS) {
    auto* largest = static_cast<LargestGlobalPool*>(data);
    if (pool_size > largest->size) {
      largest->size = pool_size;
    }
  }
  return HSA_STATUS_SUCCESS;
}

hsa_status_t CheckGpuAgent(hsa_agent_t agent, void* /*data*/) {
  hsa_device_type_t device_type;
  if (hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type) !=
      HSA_STATUS_SUCCESS) {
    return HSA_STATUS_SUCCESS;
  }
  if (device_type != HSA_DEVICE_TYPE_GPU) {
    return HSA_STATUS_SUCCESS;
  }

  uint64_t available = 0;
  hsa_status_t status = hsa_agent_get_info(
      agent, static_cast<hsa_agent_info_t>(
                 HSA_AMD_AGENT_INFO_MEMORY_AVAIL_GRAPHICS_AWARE),
      &available);
  if (status == HSA_STATUS_ERROR_INVALID_ARGUMENT) {
    return HSA_STATUS_SUCCESS;
  }
  EXPECT_EQ(status, HSA_STATUS_SUCCESS);
  if (status != HSA_STATUS_SUCCESS) {
    return HSA_STATUS_SUCCESS;
  }

  LargestGlobalPool largest;
  hsa_amd_agent_iterate_memory_pools(agent, FindLargestGlobalPool, &largest);

  // Available free memory must never exceed the device's total global pool.
  if (largest.size > 0) {
    EXPECT_LE(available, largest.size);
  }
  return HSA_STATUS_SUCCESS;
}

}  // namespace

// Self-registering gtest case; runs under the rocrtst functional binary.
TEST(rocrtstFunc, MemoryAvailGraphicsAware_NotOverReported) {
  ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
  hsa_iterate_agents(CheckGpuAgent, nullptr);
  EXPECT_EQ(hsa_shut_down(), HSA_STATUS_SUCCESS);
}
