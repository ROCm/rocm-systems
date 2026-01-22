/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#include "util.hpp"

#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <algorithm>
#include <vector>

#include "rocshmem/rocshmem_config.h"  // NOLINT(build/include_subdir)

namespace rocshmem {

__constant__ int* print_lock;

typedef struct device_agent {
  hsa_agent_t agent;
  hsa_amd_memory_pool_t pool;
} device_agent_t;

std::vector<device_agent_t> gpu_agents;
std::vector<device_agent_t> cpu_agents;

std::vector<device_prop_t> device_properties;

static void device_properties_init(void) {
  int numDevices;
  CHECK_HIP(hipGetDeviceCount(&numDevices));

  device_prop_t prop;
  hipDeviceProp_t hipprop;
  for (int i=0; i<numDevices; i++) {
    CHECK_HIP(hipGetDeviceProperties(&hipprop, i));
    prop.warpSize = hipprop.warpSize;
    prop.maxThreadsPerBlock = hipprop.maxThreadsPerBlock;
    device_properties.push_back(prop);
  }
}
hsa_status_t rocm_hsa_amd_memory_pool_callback(
    hsa_amd_memory_pool_t memory_pool, void* data) {
  hsa_amd_memory_pool_global_flag_t pool_flag{};

  hsa_status_t status{hsa_amd_memory_pool_get_info(
      memory_pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &pool_flag)};

  if (status != HSA_STATUS_SUCCESS) {
    printf("Failure to get pool info: 0x%x", status);
    return status;
  }

  if (pool_flag == (HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT |
                    HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED)) {
    *static_cast<hsa_amd_memory_pool_t*>(data) = memory_pool;
  }

  return HSA_STATUS_SUCCESS;
}

hsa_status_t rocm_hsa_agent_callback(hsa_agent_t agent,
                                     [[maybe_unused]] void* data) {
  hsa_device_type_t device_type{};

  hsa_status_t status{
      hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type)};

  if (status != HSA_STATUS_SUCCESS) {
    printf("Failure to get device type: 0x%x", status);
    return status;
  }

  if (device_type == HSA_DEVICE_TYPE_GPU) {
    gpu_agents.emplace_back();
    gpu_agents.back().agent = agent;
    status = hsa_amd_agent_iterate_memory_pools(
        agent, rocm_hsa_amd_memory_pool_callback, &(gpu_agents.back().pool));
  }

  if (device_type == HSA_DEVICE_TYPE_CPU) {
    cpu_agents.emplace_back();
    cpu_agents.back().agent = agent;
    status = hsa_amd_agent_iterate_memory_pools(
        agent, rocm_hsa_amd_memory_pool_callback, &(cpu_agents.back().pool));
  }

  return status;
}

int rocm_init() {
  hsa_status_t status{hsa_init()};

  if (status != HSA_STATUS_SUCCESS) {
    printf("Failure to open HSA connection: 0x%x", status);
    return 1;
  }

  status = hsa_iterate_agents(rocm_hsa_agent_callback, nullptr);

  if (status != HSA_STATUS_SUCCESS && status != HSA_STATUS_INFO_BREAK) {
    printf("Failure to iterate HSA agents: 0x%x", status);
    return 1;
  }

  device_properties_init();

  return 0;
}

void rocm_memory_lock_to_fine_grain(void* ptr, size_t size, void** gpu_ptr,
                                    int gpu_id) {
  hsa_status_t status{
      hsa_amd_memory_lock_to_pool(ptr, size, &(gpu_agents[gpu_id].agent), 1,
                                  cpu_agents[0].pool, 0, gpu_ptr)};

  if (status != HSA_STATUS_SUCCESS) {
    printf("Failed to lock memory pool (%p): 0x%x\n", ptr, status);
    exit(-1);
  }
}

}  // namespace rocshmem
