/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2018, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */


#include <sys/mman.h>
#include <fcntl.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <memory>
#include <sys/socket.h>

#include "suites/functional/virtual_memory.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "common/hsatimer.h"
#include "common/concurrent_utils.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"

// Wrap printf to add first or second process indicator
#define PROCESS_LOG(format, ...)                                                                   \
  {                                                                                                \
    if (verbosity() >= VERBOSE_STANDARD || !parentProcess_) {                                      \
      fprintf(stdout, "line:%d P%u: " format, __LINE__, static_cast<int>(!parentProcess_),         \
              ##__VA_ARGS__);                                                                      \
    }                                                                                              \
  }

// Fork safe ASSERT_EQ.
#define MSG(y, msg, ...) msg
#define Y(y, ...) y

#define FORK_ASSERT_EQ(x, ...)                                                                     \
  if ((x) != (Y(__VA_ARGS__))) {                                                                   \
    if ((x) != (Y(__VA_ARGS__))) {                                                                 \
      std::cout << MSG(__VA_ARGS__, "");                                                           \
      if (parentProcess_) {                                                                        \
        shared_->parent_status = -1;                                                               \
      } else {                                                                                     \
        shared_->child_status = -1;                                                                \
      }                                                                                            \
      ASSERT_EQ(x, Y(__VA_ARGS__));                                                                \
    }                                                                                              \
  }

static const char kSubTestSeparator[] = "  **************************";

static void PrintMemorySubtestHeader(const char* header) {
  std::cout << "  *** Virtual Memory Functional Subtest: " << header << " ***" << std::endl;
}

VirtMemoryTestBasic::VirtMemoryTestBasic(void) : TestBase() {
  set_title("ROCr Virtual Memory Basic Tests");
  set_description(" Tests virtual memory API functions");
}

VirtMemoryTestBasic::~VirtMemoryTestBasic(void) {}

void VirtMemoryTestBasic::TestCreateDestroy(hsa_agent_t agent, hsa_amd_memory_pool_t pool) {
  std::vector<hsa_agent_t> gpus;
  rocrtst::pool_info_t pool_i;
  hsa_device_type_t ag_type;
  char ag_name[64];
  void* addrRangeUnmapped;
  hsa_status_t err;
  void* addrRange;

  ASSERT_SUCCESS(hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &ag_type));

  ASSERT_SUCCESS(rocrtst::AcquirePoolInfo(pool, &pool_i));

  if (ag_type != HSA_DEVICE_TYPE_GPU || !pool_i.alloc_allowed) return;

  size_t granule_size = pool_i.alloc_granule;

  ASSERT_SUCCESS(hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus));
  ASSERT_SUCCESS(hsa_amd_vmem_address_reserve(&addrRange, 20 * granule_size, 0, 0));
  ASSERT_SUCCESS(hsa_amd_vmem_address_reserve(&addrRangeUnmapped, 10 * granule_size, 0, 0));

  hsa_amd_vmem_alloc_handle_t mem_handle;
  ASSERT_SUCCESS(
      hsa_amd_vmem_handle_create(pool, 10 * granule_size, MEMORY_TYPE_NONE, 0, &mem_handle));

  /* Test alloc properties returns correct memory type and pool handle */
  hsa_amd_memory_pool_t poolRet;
  hsa_amd_memory_type_t memTypeRet;
  ASSERT_SUCCESS(hsa_amd_vmem_get_alloc_properties_from_handle(mem_handle, &poolRet, &memTypeRet));

  ASSERT_EQ(poolRet.handle, pool.handle);
  ASSERT_EQ(memTypeRet, MEMORY_TYPE_NONE);

  hsa_amd_vmem_alloc_handle_t mem_handleTypePinned;
  ASSERT_SUCCESS(hsa_amd_vmem_handle_create(pool, 10 * granule_size, MEMORY_TYPE_PINNED, 0,
                                            &mem_handleTypePinned));

  ASSERT_SUCCESS(
      hsa_amd_vmem_get_alloc_properties_from_handle(mem_handleTypePinned, &poolRet, &memTypeRet));
  ASSERT_EQ(poolRet.handle, pool.handle);
  ASSERT_EQ(memTypeRet, MEMORY_TYPE_PINNED);


  ASSERT_SUCCESS(hsa_amd_vmem_map(addrRange, 10 * granule_size, 0, mem_handle, 0));

  // Access to each GPU should be None
  for (auto gpuIt = gpus.begin(); gpuIt != gpus.end(); ++gpuIt) {
    hsa_access_permission_t perm = HSA_ACCESS_PERMISSION_RW;

    ASSERT_SUCCESS(hsa_amd_vmem_get_access(addrRange, &perm, *gpuIt));
    ASSERT_EQ(perm, HSA_ACCESS_PERMISSION_NONE);
  }

  /* Set RO Access to all GPUs */
  {
    int descIndex = 0;
    hsa_amd_memory_access_desc_t desc[gpus.size()];
    for (auto gpuIt = gpus.begin(); gpuIt != gpus.end(); ++gpuIt) {
      desc[descIndex++] = {HSA_ACCESS_PERMISSION_RO, *gpuIt};
    }

    ASSERT_SUCCESS(hsa_amd_vmem_set_access(addrRange, 10 * granule_size, desc, gpus.size()));
  }

  for (auto gpuIt = gpus.begin(); gpuIt != gpus.end(); ++gpuIt) {
    hsa_access_permission_t perm = HSA_ACCESS_PERMISSION_NONE;

    ASSERT_SUCCESS(hsa_amd_vmem_get_access(addrRange, &perm, *gpuIt));
    ASSERT_EQ(perm, HSA_ACCESS_PERMISSION_RO);

    /* addrRangeUnmapped was never mapped, so this is an invalid mapping */
    err = hsa_amd_vmem_get_access(addrRangeUnmapped, &perm, *gpuIt);
    ASSERT_EQ(err, HSA_STATUS_ERROR_INVALID_ALLOCATION);
  }

  if (gpus.size() > 1) {
    /* Call set_access with a smaller list of agents, this should leave access to
     * the other GPUs unchanged */
    hsa_amd_memory_access_desc_t desc = {HSA_ACCESS_PERMISSION_RW, gpus[1]};
    ASSERT_SUCCESS(hsa_amd_vmem_set_access(addrRange, 10 * granule_size, &desc, 1));

    size_t i = 0;
    for (i = 0; i < gpus.size(); i++) {
      hsa_access_permission_t perm = HSA_ACCESS_PERMISSION_NONE;

      /* Only 2nd GPU should have RW access */
      ASSERT_SUCCESS(hsa_amd_vmem_get_access(addrRange, &perm, gpus[i]));
      if (i == 1) {
        ASSERT_EQ(perm, HSA_ACCESS_PERMISSION_RW);
      } else {
        ASSERT_EQ(perm, HSA_ACCESS_PERMISSION_RO);
      }
    }
  }

  ASSERT_SUCCESS(hsa_amd_vmem_unmap(addrRange, 10 * granule_size));
  ASSERT_SUCCESS(hsa_amd_vmem_handle_release(mem_handle));
  ASSERT_SUCCESS(hsa_amd_vmem_address_free(addrRange, 20 * granule_size));
  ASSERT_SUCCESS(hsa_amd_vmem_address_free(addrRangeUnmapped, 10 * granule_size));
}

void VirtMemoryTestBasic::TestCreateDestroy(void) {
  hsa_status_t err;
  std::vector<std::shared_ptr<rocrtst::agent_pools_t>> agent_pools;

  if (verbosity() > 0) {
    PrintMemorySubtestHeader("CreateDestroy Test");
  }
  bool supp = false;
  ASSERT_SUCCESS(hsa_system_get_info(HSA_AMD_SYSTEM_INFO_VIRTUAL_MEM_API_SUPPORTED, (void*)&supp));
  if (!supp) {
    if (verbosity() > 0) {
      std::cout << "    Virtual Memory API not supported on this system - Skipping." << std::endl;
      std::cout << kSubTestSeparator << std::endl;
    }
    return;
  }

  ASSERT_SUCCESS(rocrtst::GetAgentPools(&agent_pools));

  auto pool_idx = 0;
  for (auto a : agent_pools) {
    for (auto p : a->pools) {
      TestCreateDestroy(a->agent, p);
    }
  }

  if (verbosity() > 0) {
    std::cout << "    Subtest finished" << std::endl;
    std::cout << kSubTestSeparator << std::endl;
  }
}

void VirtMemoryTestBasic::TestRefCount(hsa_agent_t agent, hsa_amd_memory_pool_t pool) {
  rocrtst::pool_info_t pool_i;
  hsa_device_type_t ag_type;
  char ag_name[64];
  void* addrRangeUnmapped;
  hsa_status_t err;
  void* addrRange;

  ASSERT_SUCCESS(hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, ag_name));
  ASSERT_SUCCESS(hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &ag_type));
  ASSERT_SUCCESS(rocrtst::AcquirePoolInfo(pool, &pool_i));

  if (ag_type != HSA_DEVICE_TYPE_GPU || !pool_i.alloc_allowed) return;

  size_t granule_size = pool_i.alloc_granule;

  ASSERT_SUCCESS(hsa_amd_vmem_address_reserve(&addrRange, 10 * granule_size, 0, 0));

  hsa_amd_vmem_alloc_handle_t mem_handleA1;
  ASSERT_SUCCESS(
      hsa_amd_vmem_handle_create(pool, 10 * granule_size, MEMORY_TYPE_NONE, 0, &mem_handleA1));
  ASSERT_SUCCESS(hsa_amd_vmem_map(addrRange, 10 * granule_size, 0, mem_handleA1, 0));

  /* Allocate duplicate handle */
  hsa_amd_vmem_alloc_handle_t mem_handleA1Dup;
  ASSERT_SUCCESS(hsa_amd_vmem_retain_alloc_handle(&mem_handleA1Dup, addrRange));

  /* Try to unmap with incorrect size */
  err = hsa_amd_vmem_unmap(addrRange, 5 * granule_size);
  ASSERT_NE(err, HSA_STATUS_SUCCESS);

  ASSERT_SUCCESS(hsa_amd_vmem_handle_release(mem_handleA1));

  /* Try to release duplicate handle twice - second time should fail */
  ASSERT_SUCCESS(hsa_amd_vmem_handle_release(mem_handleA1Dup));

  /* Already released so should fail*/
  err = hsa_amd_vmem_handle_release(mem_handleA1Dup);
  ASSERT_NE(err, HSA_STATUS_SUCCESS);

  /* Unmap with correct size - un-mapping after releasing the handle is valid */
  ASSERT_SUCCESS(hsa_amd_vmem_unmap(addrRange, 10 * granule_size));

  /* Try to free with incorrect size */
  err = hsa_amd_vmem_address_free(addrRange, 5 * granule_size);
  ASSERT_NE(err, HSA_STATUS_SUCCESS);

  /* Free with correct size */
  ASSERT_SUCCESS(hsa_amd_vmem_address_free(addrRange, 10 * granule_size));
}

void VirtMemoryTestBasic::TestRefCount(void) {
  hsa_status_t err;
  std::vector<std::shared_ptr<rocrtst::agent_pools_t>> agent_pools;

  if (verbosity() > 0) {
    PrintMemorySubtestHeader("Reference Count Test");
  }
  bool supp = false;
  ASSERT_SUCCESS(hsa_system_get_info(HSA_AMD_SYSTEM_INFO_VIRTUAL_MEM_API_SUPPORTED, (void*)&supp));
  if (!supp) {
    if (verbosity() > 0) {
      std::cout << "    Virtual Memory API not supported on this system - Skipping." << std::endl;
      std::cout << kSubTestSeparator << std::endl;
    }
    return;
  }
  ASSERT_SUCCESS(rocrtst::GetAgentPools(&agent_pools));

  auto pool_idx = 0;
  for (auto a : agent_pools) {
    for (auto p : a->pools) TestRefCount(a->agent, p);
  }

  if (verbosity() > 0) {
    std::cout << "    Subtest finished" << std::endl;
    std::cout << kSubTestSeparator << std::endl;
  }
}

void VirtMemoryTestBasic::TestPartialMapping(hsa_agent_t agent, hsa_amd_memory_pool_t pool) {
  rocrtst::pool_info_t pool_i;
  hsa_device_type_t ag_type;
  char ag_name[64];
  void* addrRangeUnmapped;
  hsa_status_t err;
  void* addrRange;

  ASSERT_SUCCESS(hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &ag_type));

  ASSERT_SUCCESS(rocrtst::AcquirePoolInfo(pool, &pool_i));

  if (ag_type != HSA_DEVICE_TYPE_GPU || !pool_i.alloc_allowed) return;

  size_t granule_size = pool_i.alloc_granule;

  /************************************************************************************************
    Map partial chunks within the address range and confirm what overlaps fail.
    Units below are in multiples of granule_size.

              ------------------------------------------------------------------
              | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
              ------------------------------------------------------------------
    Step 1:             A   A   A   A   A   A
    Step 2:                                                  B    B    B
    Step 3:                                                                 B
    Step 4:                                     B   B    B

   ***********************************************************************************************/

  ASSERT_SUCCESS(hsa_amd_vmem_address_reserve(&addrRange, 15 * granule_size, 0, 0));

  hsa_amd_vmem_alloc_handle_t mem_handleA;

  // Step 1
  ASSERT_SUCCESS(
      hsa_amd_vmem_handle_create(pool, 8 * granule_size, MEMORY_TYPE_NONE, 0, &mem_handleA));

  ASSERT_SUCCESS(hsa_amd_vmem_map((void*)((uint64_t)addrRange + (2 * granule_size)),
                                  6 * granule_size, 0, mem_handleA, 0));

  // Step 2
  hsa_amd_vmem_alloc_handle_t mem_handleB;
  ASSERT_SUCCESS(
      hsa_amd_vmem_handle_create(pool, 8 * granule_size, MEMORY_TYPE_NONE, 0, &mem_handleB));

  ASSERT_SUCCESS(hsa_amd_vmem_map((void*)((uint64_t)addrRange + (11 * granule_size)),
                                  3 * granule_size, 0, mem_handleB, 0));

  // Step 3
  // Should fail as this is exceeding size of address range
  err = hsa_amd_vmem_map((void*)((uint64_t)addrRange + (14 * granule_size)),
                                  2 * granule_size, 0, mem_handleB, 0);
  ASSERT_NE(err, HSA_STATUS_SUCCESS);

  ASSERT_SUCCESS(hsa_amd_vmem_map((void*)((uint64_t)addrRange + (14 * granule_size)),
                                  1 * granule_size, 0, mem_handleB, 0));

  // Step 4
  // Should fail as this is overlapping with AddressRange[11] already mapped
  err = hsa_amd_vmem_map((void*)((uint64_t)addrRange + (8 * granule_size)),
                                  4 * granule_size, 0, mem_handleB, 0);
  ASSERT_NE(err, HSA_STATUS_SUCCESS);

  ASSERT_SUCCESS(hsa_amd_vmem_map((void*)((uint64_t)addrRange + (8 * granule_size)),
                                  3 * granule_size, 0, mem_handleB, 0));

  // Done, unmap all
  ASSERT_SUCCESS(
      hsa_amd_vmem_unmap((void*)((uint64_t)addrRange + (2 * granule_size)), 6 * granule_size));
  ASSERT_SUCCESS(
      hsa_amd_vmem_unmap((void*)((uint64_t)addrRange + (8 * granule_size)), 3 * granule_size));
  ASSERT_SUCCESS(
      hsa_amd_vmem_unmap((void*)((uint64_t)addrRange + (11 * granule_size)), 3 * granule_size));
  ASSERT_SUCCESS(
      hsa_amd_vmem_unmap((void*)((uint64_t)addrRange + (14 * granule_size)), 1 * granule_size));
  ASSERT_SUCCESS(hsa_amd_vmem_address_free(addrRange, 15 * granule_size));
}

void VirtMemoryTestBasic::TestPartialMapping(void) {
  hsa_status_t err;
  std::vector<std::shared_ptr<rocrtst::agent_pools_t>> agent_pools;

  if (verbosity() > 0) {
    PrintMemorySubtestHeader("Partial Mapping Test");
  }

  bool supp = false;
  ASSERT_SUCCESS(hsa_system_get_info(HSA_AMD_SYSTEM_INFO_VIRTUAL_MEM_API_SUPPORTED, (void*)&supp));
  if (!supp) {
    if (verbosity() > 0) {
      std::cout << "    Virtual Memory API not supported on this system - Skipping." << std::endl;
      std::cout << kSubTestSeparator << std::endl;
    }
    return;
  }

  ASSERT_SUCCESS(rocrtst::GetAgentPools(&agent_pools));

  auto pool_idx = 0;
  for (auto a : agent_pools) {
    for (auto p : a->pools) TestPartialMapping(a->agent, p);
  }

  if (verbosity() > 0) {
    std::cout << "    Subtest finished" << std::endl;
    std::cout << kSubTestSeparator << std::endl;
  }
}

typedef struct __attribute__((aligned(16))) args_t {
  int* a;
  int* b;
  int* c;
} args;

args* kernArgsVirt = NULL;

// Test to check CPU can read & write to GPU memory
void VirtMemoryTestBasic::CPUAccessToGPUMemoryTest(hsa_agent_t cpuAgent, hsa_agent_t gpuAgent,
                                                   hsa_amd_memory_pool_t device_pool) {
  hsa_status_t err;

  rocrtst::pool_info_t pool_i;
  ASSERT_SUCCESS(rocrtst::AcquirePoolInfo(device_pool, &pool_i));

  if (!(pool_i.segment == HSA_AMD_SEGMENT_GLOBAL &&
        pool_i.global_flag == HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED))
    return;

  hsa_amd_memory_pool_access_t access;
  hsa_amd_agent_memory_pool_get_info(cpuAgent, device_pool, HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS,
                                     &access);
  if (access == HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
    if (verbosity() > 0) {
      std::cout << "    Test not applicable as system is not large bar - Skipping." << std::endl;
      std::cout << kSubTestSeparator << std::endl;
    }
    return;
  }
  if (!pool_i.alloc_allowed || pool_i.alloc_granule == 0 || pool_i.alloc_alignment == 0) {
    if (verbosity() > 0) {
      std::cout << "    Test not applicable. Skipping." << std::endl;
      std::cout << kSubTestSeparator << std::endl;
    }
    return;
  }

  auto max_alloc_size = pool_i.alloc_granule * 100;
  unsigned int max_element = max_alloc_size / sizeof(unsigned int);
  unsigned int* dev_data = NULL;
  unsigned int* host_data = NULL;
  host_data = (unsigned int*)malloc(max_alloc_size);

  ASSERT_NE(host_data, nullptr);

  for (unsigned int i = 0; i < max_element; ++i) {
    host_data[i] = i;
  }

  hsa_amd_memory_access_desc_t permsAccess[] = {{HSA_ACCESS_PERMISSION_RW, cpuAgent},
                                                {HSA_ACCESS_PERMISSION_RW, gpuAgent}};

  hsa_amd_vmem_alloc_handle_t mem_handle_host, mem_handle_dev;
  ASSERT_SUCCESS(
      hsa_amd_vmem_address_reserve(reinterpret_cast<void**>(&dev_data), max_alloc_size, 0, 0));

  ASSERT_NE(dev_data, nullptr);

  ASSERT_SUCCESS(hsa_amd_vmem_handle_create(device_pool, max_alloc_size, MEMORY_TYPE_NONE, 0,
                                            &mem_handle_dev));
  ASSERT_SUCCESS(
      hsa_amd_vmem_map(reinterpret_cast<void*>(dev_data), max_alloc_size, 0, mem_handle_dev, 0));

  // Give device access to host data
  ASSERT_SUCCESS(hsa_amd_vmem_set_access(dev_data, max_alloc_size, permsAccess, 2));

  // Verify CPU can read & write to GPU memory
  std::cout << "    Verify CPU can read & write to GPU memory" << std::endl;
  for (unsigned int i = 0; i < max_element; ++i) {
    dev_data[i] = i;  // Write to gpu memory directly
  }

  for (unsigned int i = 0; i < max_element; ++i) {
    if (host_data[i] != dev_data[i]) {  // Reading GPU memory
      fprintf(stdout,
              "    Values not mathing !! host_data[%d]:%d ,"
              "dev_data[%d]\n",
              host_data[i], i, dev_data[i]);
    }
  }
  std::cout << "    CPU have read & write to GPU memory successfully" << std::endl;

  ASSERT_SUCCESS(hsa_amd_vmem_unmap(dev_data, max_alloc_size));
  ASSERT_SUCCESS(hsa_amd_vmem_handle_release(mem_handle_dev));
  ASSERT_SUCCESS(hsa_amd_vmem_address_free(reinterpret_cast<void*>(dev_data), max_alloc_size));
  free(host_data);
}

void VirtMemoryTestBasic::CPUAccessToGPUMemoryTest(void) {
  hsa_status_t err;
  // find all cpu agents
  std::vector<hsa_agent_t> cpus;
  ASSERT_SUCCESS(hsa_iterate_agents(rocrtst::IterateCPUAgents, &cpus));

  // find all gpu agents
  std::vector<hsa_agent_t> gpus;
  ASSERT_SUCCESS(hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus));

  if (verbosity() > 0) PrintMemorySubtestHeader("CPU To GPU Access test");

  bool supp = false;
  ASSERT_SUCCESS(hsa_system_get_info(HSA_AMD_SYSTEM_INFO_VIRTUAL_MEM_API_SUPPORTED, (void*)&supp));
  if (!supp) {
    if (verbosity() > 0) {
      std::cout << "    Virtual Memory API not supported on this system - Skipping." << std::endl;
      std::cout << kSubTestSeparator << std::endl;
    }
    return;
  }

  for (unsigned int i = 0; i < gpus.size(); ++i) {
    hsa_amd_memory_pool_t gpu_pool;
    memset(&gpu_pool, 0, sizeof(gpu_pool));
    ASSERT_SUCCESS(
        hsa_amd_agent_iterate_memory_pools(gpus[i], rocrtst::GetGlobalMemoryPool, &gpu_pool));
    if (gpu_pool.handle == 0) {
      std::cout << "    No global mempool in gpu agent" << std::endl;
      return;
    }
    CPUAccessToGPUMemoryTest(cpus[0], gpus[i], gpu_pool);
  }
  if (verbosity() > 0) {
    std::cout << "    Subtest finished" << std::endl;
    std::cout << kSubTestSeparator << std::endl;
  }
}

// Test to check GPU can read & write to CPU memory
void VirtMemoryTestBasic::GPUAccessToCPUMemoryTest(hsa_agent_t cpuAgent, hsa_agent_t gpuAgent,
                                                   hsa_amd_memory_pool_t device_pool) {
  rocrtst::pool_info_t pool_i;
  hsa_device_type_t ag_type;
  char ag_name[64];
  hsa_status_t err;

  ASSERT_SUCCESS(rocrtst::AcquirePoolInfo(device_pool, &pool_i));

  if (!pool_i.alloc_allowed || pool_i.segment != HSA_AMD_SEGMENT_GLOBAL ||
      pool_i.global_flag != HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED)
    return;

  hsa_amd_memory_pool_access_t access;
  ASSERT_SUCCESS(hsa_amd_agent_memory_pool_get_info(
      cpuAgent, device_pool, HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &access));

  if (access == HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
    if (verbosity() > 0) {
      std::cout << "    Test not applicable as system is not large bar - Skipping." << std::endl;
      std::cout << kSubTestSeparator << std::endl;
      return;
    }
  }

  hsa_queue_t* queue = NULL;  // command queue
  hsa_signal_t signal = {0};  // completion signal

  size_t& granule_size = pool_i.alloc_granule;
  size_t alloc_size = granule_size * 100;
  static const int kMemoryAllocSize = 1024;
  unsigned int max_element = alloc_size / sizeof(unsigned int);

  // get queue size
  uint32_t queue_size = 0;
  ASSERT_SUCCESS(hsa_agent_get_info(gpuAgent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size));

  // create queue
  ASSERT_SUCCESS(
      hsa_queue_create(gpuAgent, queue_size, HSA_QUEUE_TYPE_MULTI, NULL, NULL, 0, 0, &queue));

  // Find a memory pool that supports kernel arguments.
  hsa_amd_memory_pool_t kernarg_pool;
  ASSERT_SUCCESS(
      hsa_amd_agent_iterate_memory_pools(cpuAgent, rocrtst::GetKernArgMemoryPool, &kernarg_pool));

  // Get System Memory Pool on the cpuAgent to allocate host side buffers
  hsa_amd_memory_pool_t global_pool;
  ASSERT_SUCCESS(
      hsa_amd_agent_iterate_memory_pools(cpuAgent, rocrtst::GetGlobalMemoryPool, &global_pool));

  struct host_data_t {
    int data[kMemoryAllocSize * 4];
    int dup_data[kMemoryAllocSize * 4];
    int result[kMemoryAllocSize * 4];
  };

  struct dev_data_t {
    int result[kMemoryAllocSize * 4];
  };


  struct host_data_t* host_data;
  struct dev_data_t* dev_data;

  ASSERT_SUCCESS(hsa_amd_memory_pool_allocate(global_pool, sizeof(*host_data), 0,
                                              reinterpret_cast<void**>(&host_data)));

  // Allow gpuAgent access to all allocated system memory.
  ASSERT_SUCCESS(hsa_amd_agents_allow_access(1, &gpuAgent, NULL, host_data));
  ASSERT_SUCCESS(hsa_amd_vmem_address_reserve((void**)&dev_data, sizeof(*dev_data), 0, 0));

  hsa_amd_vmem_alloc_handle_t mem_handle;

  ASSERT_SUCCESS(
      hsa_amd_vmem_handle_create(device_pool, sizeof(*dev_data), MEMORY_TYPE_NONE, 0, &mem_handle));
  ASSERT_SUCCESS(hsa_amd_vmem_map(dev_data, sizeof(*dev_data), 0, mem_handle, 0));

  // Give host and device access to device data
  hsa_amd_memory_access_desc_t permsAccess[] = {{HSA_ACCESS_PERMISSION_RW, gpuAgent},
                                                {HSA_ACCESS_PERMISSION_RW, cpuAgent}};

  ASSERT_SUCCESS(hsa_amd_vmem_set_access(dev_data, sizeof(*dev_data), permsAccess, 2));

  // Allocate the kernel argument buffer from the kernarg_pool.
  ASSERT_SUCCESS(hsa_amd_memory_pool_allocate(kernarg_pool, sizeof(args_t), 0,
                                              reinterpret_cast<void**>(&kernArgsVirt)));

  // initialize the host buffers
  for (int i = 0; i < kMemoryAllocSize; ++i) {
    unsigned int seed = time(NULL);
    host_data->data[i] = 1 + rand_r(&seed) % 1;
    host_data->dup_data[i] = host_data->data[i];
  }

  memset(host_data->result, 0, sizeof(host_data->result));
  memset(dev_data->result, 0, sizeof(dev_data->result));

  ASSERT_SUCCESS(hsa_amd_agents_allow_access(1, &gpuAgent, NULL, kernArgsVirt));

  kernArgsVirt->a = host_data->data;
  kernArgsVirt->b = host_data->result;  // system memory passed to gpu for write
  kernArgsVirt->c = dev_data->result;   // gpu memory to verify that gpu read system data

  // Create the executable, get symbol by name and load the code object
  set_kernel_file_name("gpuReadWrite_kernels.hsaco");
  set_kernel_name("gpuReadWrite");
  ASSERT_SUCCESS(rocrtst::LoadKernelFromObjFile(this, &gpuAgent));

  // Fill the dispatch packet with
  // workgroup_size, grid_size, kernelArgs and completion signal
  // Put it on the queue and launch the kernel by ringing the doorbell

  // create completion signal
  ASSERT_SUCCESS(hsa_signal_create(1, 0, NULL, &signal));

  // create aql packet
  hsa_kernel_dispatch_packet_t aql;
  memset(&aql, 0, sizeof(aql));

  // initialize aql packet
  aql.workgroup_size_x = 256;
  aql.workgroup_size_y = 1;
  aql.workgroup_size_z = 1;
  aql.grid_size_x = kMemoryAllocSize;
  aql.grid_size_y = 1;
  aql.grid_size_z = 1;
  aql.private_segment_size = 0;
  aql.group_segment_size = 0;
  aql.kernel_object = kernel_object();  // kernel_code;
  aql.kernarg_address = kernArgsVirt;
  aql.completion_signal = signal;

  // const uint32_t queue_size = queue->size;
  const uint32_t queue_mask = queue->size - 1;

  // write to command queue
  uint64_t index = hsa_queue_load_write_index_relaxed(queue);
  hsa_queue_store_write_index_relaxed(queue, index + 1);

  rocrtst::WriteAQLToQueueLoc(queue, index, &aql);

  hsa_kernel_dispatch_packet_t* q_base_addr =
      reinterpret_cast<hsa_kernel_dispatch_packet_t*>(queue->base_address);
  rocrtst::AtomicSetPacketHeader(
      (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
          (1 << HSA_PACKET_HEADER_BARRIER) |
          (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
          (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE),
      (1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS),
      reinterpret_cast<hsa_kernel_dispatch_packet_t*>(&q_base_addr[index & queue_mask]));

  // ringdoor bell
  hsa_signal_store_relaxed(queue->doorbell_signal, index);
  // wait for the signal and reset it for future use
  while (hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1, (uint64_t)-1,
                                   HSA_WAIT_STATE_ACTIVE)) {
  }
  hsa_signal_store_relaxed(signal, 1);

  // compare device and host side results
  if (verbosity() > 0) {
    std::cout << "    Check GPU has read the system memory" << std::endl;
  }
  for (int i = 0; i < kMemoryAllocSize; ++i) {
    // printf("Verifying data at index[%d]\n", i);
    ASSERT_EQ(dev_data->result[i], host_data->dup_data[i]);
  }

  if (verbosity() > 0) {
    std::cout << "    GPU has read the system memory successfully" << std::endl;
    std::cout << "    Check GPU has written to system memory" << std::endl;
  }
  for (int i = 0; i < kMemoryAllocSize; ++i) {
    ASSERT_EQ(host_data->result[i], i);
  }

  if (verbosity() > 0) {
    std::cout << "    GPU has written to system memory successfully" << std::endl;
  }

  ASSERT_SUCCESS(hsa_amd_vmem_unmap(dev_data, sizeof(*dev_data)));
  ASSERT_SUCCESS(hsa_amd_vmem_handle_release(mem_handle));

  if (dev_data) {
    ASSERT_SUCCESS(hsa_amd_vmem_address_free(dev_data, sizeof(*dev_data)));
  }

  if (host_data) hsa_memory_free(host_data);
  if (kernArgsVirt) {
    hsa_memory_free(kernArgsVirt);
  }
  if (signal.handle) {
    hsa_signal_destroy(signal);
  }
  if (queue) {
    hsa_queue_destroy(queue);
  }
}

void VirtMemoryTestBasic::GPUAccessToCPUMemoryTest(void) {
  hsa_status_t err;
  // find all cpu agents
  std::vector<hsa_agent_t> cpus;
  ASSERT_SUCCESS(hsa_iterate_agents(rocrtst::IterateCPUAgents, &cpus));

  // find all gpu agents
  std::vector<hsa_agent_t> gpus;
  ASSERT_SUCCESS(hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus));

  if (verbosity() > 0) PrintMemorySubtestHeader("CPU To GPU Access test");

  bool supp = false;
  ASSERT_SUCCESS(hsa_system_get_info(HSA_AMD_SYSTEM_INFO_VIRTUAL_MEM_API_SUPPORTED, (void*)&supp));
  if (!supp) {
    if (verbosity() > 0) {
      std::cout << "    Virtual Memory API not supported on this system - Skipping." << std::endl;
      std::cout << kSubTestSeparator << std::endl;
    }
    return;
  }

  for (unsigned int i = 0; i < gpus.size(); ++i) {
    hsa_amd_memory_pool_t gpu_pool;
    memset(&gpu_pool, 0, sizeof(gpu_pool));
    ASSERT_SUCCESS(
        hsa_amd_agent_iterate_memory_pools(gpus[i], rocrtst::GetGlobalMemoryPool, &gpu_pool));
    if (gpu_pool.handle == 0) {
      std::cout << "no global mempool in GPU agent" << std::endl;
      return;
    }
    GPUAccessToCPUMemoryTest(cpus[0], gpus[i], gpu_pool);
  }
  if (verbosity() > 0) {
    std::cout << "    Subtest finished" << std::endl;
    std::cout << kSubTestSeparator << std::endl;
  }
}

// Test to check GPU can read & write to GPU memory
void VirtMemoryTestBasic::GPUAccessToGPUMemoryTest(hsa_agent_t cpuAgent, hsa_agent_t gpuAgent,
                                                   hsa_amd_memory_pool_t device_pool) {
  rocrtst::pool_info_t pool_i;
  hsa_device_type_t ag_type;
  char ag_name[64];
  hsa_status_t err;

  ASSERT_SUCCESS(rocrtst::AcquirePoolInfo(device_pool, &pool_i));

  if (!pool_i.alloc_allowed || pool_i.segment != HSA_AMD_SEGMENT_GLOBAL ||
      pool_i.global_flag != HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED)
    return;

  hsa_amd_memory_pool_access_t access;
  ASSERT_SUCCESS(hsa_amd_agent_memory_pool_get_info(
      cpuAgent, device_pool, HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &access));

  if (access == HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
    if (verbosity() > 0) {
      std::cout << "    Test not applicable as system is not large bar - Skipping." << std::endl;
      std::cout << kSubTestSeparator << std::endl;
      return;
    }
  }

  hsa_queue_t* queue = NULL;  // command queue
  hsa_signal_t signal = {0};  // completion signal

  size_t& granule_size = pool_i.alloc_granule;
  size_t alloc_size = granule_size * 100;
  static const int kMemoryAllocSize = 4096;
  unsigned int max_element = alloc_size / sizeof(unsigned int);

  // get queue size
  uint32_t queue_size = 0;
  ASSERT_SUCCESS(hsa_agent_get_info(gpuAgent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size));

  // create queue
  ASSERT_SUCCESS(
      hsa_queue_create(gpuAgent, queue_size, HSA_QUEUE_TYPE_MULTI, NULL, NULL, 0, 0, &queue));

  // Find a memory pool that supports kernel arguments.
  hsa_amd_memory_pool_t kernarg_pool;
  ASSERT_SUCCESS(
      hsa_amd_agent_iterate_memory_pools(cpuAgent, rocrtst::GetKernArgMemoryPool, &kernarg_pool));

  // Get System Memory Pool on the cpuAgent to allocate host side buffers
  hsa_amd_memory_pool_t global_pool;
  ASSERT_SUCCESS(
      hsa_amd_agent_iterate_memory_pools(cpuAgent, rocrtst::GetGlobalMemoryPool, &global_pool));

  struct host_data_t {
    int data[kMemoryAllocSize * 4];
    int gpuWrite[kMemoryAllocSize * 4];
    int result[kMemoryAllocSize * 4];
  };

  struct dev_data_t {
    int data[kMemoryAllocSize * 4];
    int result[kMemoryAllocSize * 4];
  };


  struct host_data_t* host_data;
  struct dev_data_t* dev_data;

  ASSERT_SUCCESS(hsa_amd_memory_pool_allocate(global_pool, sizeof(*host_data), 0,
                                              reinterpret_cast<void**>(&host_data)));

  // Allow gpuAgent access to all allocated system memory.
  ASSERT_SUCCESS(hsa_amd_agents_allow_access(1, &gpuAgent, NULL, host_data));
  ASSERT_SUCCESS(hsa_amd_vmem_address_reserve((void**)&dev_data, sizeof(*dev_data), 0, 0));

  hsa_amd_vmem_alloc_handle_t mem_handle;

  ASSERT_SUCCESS(hsa_amd_vmem_handle_create(device_pool, sizeof(*dev_data), MEMORY_TYPE_PINNED, 0,
                                            &mem_handle));

  ASSERT_SUCCESS(hsa_amd_vmem_map(dev_data, sizeof(*dev_data), 0, mem_handle, 0));

  // Give host and device access to device data
  hsa_amd_memory_access_desc_t permsAccess[] = {{HSA_ACCESS_PERMISSION_RW, gpuAgent}};

  ASSERT_SUCCESS(
      hsa_amd_vmem_set_access(dev_data, sizeof(*dev_data), permsAccess, ARRAY_SIZE(permsAccess)));

  // Allocate the kernel argument buffer from the kernarg_pool.
  ASSERT_SUCCESS(hsa_amd_memory_pool_allocate(kernarg_pool, sizeof(args_t), 0,
                                              reinterpret_cast<void**>(&kernArgsVirt)));

  // create completion signal
  ASSERT_SUCCESS(hsa_signal_create(1, 0, NULL, &signal));

  // initialize the host buffers
  for (int i = 0; i < kMemoryAllocSize; ++i) {
    unsigned int seed = time(NULL);
    host_data->data[i] = 1 + rand_r(&seed) % 1;
  }

  ASSERT_SUCCESS(hsa_amd_memory_async_copy(dev_data->data, gpuAgent, host_data->data, cpuAgent,
                                           kMemoryAllocSize * 4, 0, NULL, signal));

  while (hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1, (uint64_t)-1,
                                   HSA_WAIT_STATE_ACTIVE)) {
  }
  hsa_signal_store_relaxed(signal, 1);

  memset(host_data->result, 0, sizeof(host_data->result));

  ASSERT_SUCCESS(hsa_amd_agents_allow_access(1, &gpuAgent, NULL, kernArgsVirt));


  kernArgsVirt->a = dev_data->data;
  kernArgsVirt->b = host_data->gpuWrite;  // system memory passed to gpu for write
  kernArgsVirt->c = dev_data->result;     // gpu memory to verify that gpu read system data

  // Create the executable, get symbol by name and load the code object
  set_kernel_file_name("gpuReadWrite_kernels.hsaco");
  set_kernel_name("gpuReadWrite");
  ASSERT_SUCCESS(rocrtst::LoadKernelFromObjFile(this, &gpuAgent));

  // Fill the dispatch packet with
  // workgroup_size, grid_size, kernelArgs and completion signal
  // Put it on the queue and launch the kernel by ringing the doorbell

  // create aql packet
  hsa_kernel_dispatch_packet_t aql;
  memset(&aql, 0, sizeof(aql));

  // initialize aql packet
  aql.workgroup_size_x = 256;
  aql.workgroup_size_y = 1;
  aql.workgroup_size_z = 1;
  aql.grid_size_x = kMemoryAllocSize;
  aql.grid_size_y = 1;
  aql.grid_size_z = 1;
  aql.private_segment_size = 0;
  aql.group_segment_size = 0;
  aql.kernel_object = kernel_object();  // kernel_code;
  aql.kernarg_address = kernArgsVirt;
  aql.completion_signal = signal;

  const uint32_t queue_mask = queue->size - 1;

  // write to command queue
  uint64_t index = hsa_queue_load_write_index_relaxed(queue);
  hsa_queue_store_write_index_relaxed(queue, index + 1);

  rocrtst::WriteAQLToQueueLoc(queue, index, &aql);

  hsa_kernel_dispatch_packet_t* q_base_addr =
      reinterpret_cast<hsa_kernel_dispatch_packet_t*>(queue->base_address);
  rocrtst::AtomicSetPacketHeader(
      (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
          (1 << HSA_PACKET_HEADER_BARRIER) |
          (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
          (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE),
      (1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS),
      reinterpret_cast<hsa_kernel_dispatch_packet_t*>(&q_base_addr[index & queue_mask]));

  // ringdoor bell
  hsa_signal_store_relaxed(queue->doorbell_signal, index);
  // wait for the signal and reset it for future use
  while (hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1, (uint64_t)-1,
                                   HSA_WAIT_STATE_ACTIVE)) {
  }
  hsa_signal_store_relaxed(signal, 1);

  ASSERT_SUCCESS(hsa_amd_memory_async_copy(host_data->result, cpuAgent, dev_data->result, gpuAgent,
                                           kMemoryAllocSize * 4, 0, NULL, signal));

  while (hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1, (uint64_t)-1,
                                   HSA_WAIT_STATE_ACTIVE)) {
  }
  // compare device and host side results
  if (verbosity() > 0) {
    std::cout << "    Check GPU has read the system memory" << std::endl;
  }
  for (int i = 0; i < kMemoryAllocSize; ++i) {
    // printf("Verifying data at index[%d]\n", i);
    ASSERT_EQ(host_data->result[i], host_data->data[i]);
  }

  if (verbosity() > 0) {
    std::cout << "    GPU has read the system memory successfully" << std::endl;
    std::cout << "    Check GPU has written to system memory" << std::endl;
  }
  for (int i = 0; i < kMemoryAllocSize; ++i) {
    ASSERT_EQ(host_data->gpuWrite[i], i);
  }

  if (verbosity() > 0) {
    std::cout << "    GPU has written to system memory successfully" << std::endl;
  }

  ASSERT_SUCCESS(hsa_amd_vmem_unmap(dev_data, sizeof(*dev_data)));
  ASSERT_SUCCESS(hsa_amd_vmem_handle_release(mem_handle));

  if (dev_data) {
    ASSERT_SUCCESS(hsa_amd_vmem_address_free(dev_data, sizeof(*dev_data)));
  }

  if (host_data) hsa_memory_free(host_data);
  if (kernArgsVirt) {
    hsa_memory_free(kernArgsVirt);
  }
  if (signal.handle) {
    hsa_signal_destroy(signal);
  }
  if (queue) {
    hsa_queue_destroy(queue);
  }
}

void VirtMemoryTestBasic::GPUAccessToGPUMemoryTest(void) {
  hsa_status_t err;
  // find all cpu agents
  std::vector<hsa_agent_t> cpus;
  ASSERT_SUCCESS(hsa_iterate_agents(rocrtst::IterateCPUAgents, &cpus));

  // find all gpu agents
  std::vector<hsa_agent_t> gpus;
  ASSERT_SUCCESS(hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus));

  if (verbosity() > 0) PrintMemorySubtestHeader("GPU To GPU Access test");

  bool supp = false;
  ASSERT_SUCCESS(hsa_system_get_info(HSA_AMD_SYSTEM_INFO_VIRTUAL_MEM_API_SUPPORTED, (void*)&supp));
  if (!supp) {
    if (verbosity() > 0) {
      std::cout << "    Virtual Memory API not supported on this system - Skipping." << std::endl;
      std::cout << kSubTestSeparator << std::endl;
    }
    return;
  }

  for (unsigned int i = 0; i < gpus.size(); ++i) {
    hsa_amd_memory_pool_t gpu_pool;
    memset(&gpu_pool, 0, sizeof(gpu_pool));
    ASSERT_SUCCESS(
        hsa_amd_agent_iterate_memory_pools(gpus[i], rocrtst::GetGlobalMemoryPool, &gpu_pool));
    if (gpu_pool.handle == 0) {
      std::cout << "no global mempool in GPU agent" << std::endl;
      return;
    }
    GPUAccessToGPUMemoryTest(cpus[0], gpus[i], gpu_pool);
  }
  if (verbosity() > 0) {
    std::cout << "    Subtest finished" << std::endl;
    std::cout << kSubTestSeparator << std::endl;
  }
}

void VirtMemoryTestBasic::NonContiguousChunks(hsa_agent_t cpuAgent, hsa_agent_t gpuAgent,
                                              hsa_amd_memory_pool_t device_pool) {
  rocrtst::pool_info_t pool_i;
  hsa_device_type_t ag_type;
  char ag_name[64];
  hsa_status_t err;

  ASSERT_SUCCESS(rocrtst::AcquirePoolInfo(device_pool, &pool_i));

  if (!pool_i.alloc_allowed || pool_i.segment != HSA_AMD_SEGMENT_GLOBAL ||
      pool_i.global_flag != HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED)
    return;

  hsa_amd_memory_pool_access_t access;
  ASSERT_SUCCESS(hsa_amd_agent_memory_pool_get_info(
      cpuAgent, device_pool, HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &access));

  if (access == HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
    if (verbosity() > 0) {
      std::cout << "    Test not applicable as system is not large bar - Skipping." << std::endl;
      std::cout << kSubTestSeparator << std::endl;
      return;
    }
  }

  size_t& granule_size = pool_i.alloc_granule;
  size_t alloc_size = granule_size * 512;
  const unsigned NUM_BUFFERS = 6;

  void* addr;
  void* addr_chunks[NUM_BUFFERS];
  hsa_amd_vmem_alloc_handle_t mem_handles[NUM_BUFFERS];

  static const int kMemoryAllocSize = 4096;
  unsigned int max_element = alloc_size / sizeof(unsigned int);

  ASSERT_SUCCESS(hsa_amd_vmem_address_reserve((void**)&addr, NUM_BUFFERS * alloc_size, 0, 0));

  for (unsigned i = 0; i < NUM_BUFFERS; i++) {
    // Allocate 6 separate memory memory handles
    ASSERT_SUCCESS(hsa_amd_vmem_handle_create(device_pool, alloc_size, MEMORY_TYPE_PINNED, 0,
                                              &(mem_handles[i])));
    addr_chunks[i] = ((uint8_t*)addr) + (i * alloc_size);
  }

  for (unsigned i = 0; i < NUM_BUFFERS; i++) {
    // Map each chunk in reverse order
    ASSERT_SUCCESS(hsa_amd_vmem_map(addr_chunks[i], alloc_size, 0, mem_handles[NUM_BUFFERS - i - 1],
                                    alloc_size));
  }

  hsa_amd_memory_access_desc_t permsAccess[] = {{HSA_ACCESS_PERMISSION_RW, gpuAgent}};

  ASSERT_SUCCESS(hsa_amd_vmem_set_access(addr, NUM_BUFFERS * alloc_size, permsAccess,
                                         ARRAY_SIZE(permsAccess)));

  for (unsigned i = 0; i < NUM_BUFFERS; i++) {
    // TODO Map them in opposite order
    ASSERT_SUCCESS(hsa_amd_vmem_unmap(addr_chunks[i], alloc_size));
  }

  ASSERT_SUCCESS(hsa_amd_vmem_address_free(addr, NUM_BUFFERS * alloc_size));
}

void VirtMemoryTestBasic::NonContiguousChunks(void) {
  hsa_status_t err;

  if (verbosity() > 0) PrintMemorySubtestHeader("GPU To GPU Access test");

  bool supp = false;
  ASSERT_SUCCESS(hsa_system_get_info(HSA_AMD_SYSTEM_INFO_VIRTUAL_MEM_API_SUPPORTED, (void*)&supp));
  if (!supp) {
    if (verbosity() > 0) {
      std::cout << "    Virtual Memory API not supported on this system - Skipping." << std::endl;
      std::cout << kSubTestSeparator << std::endl;
    }
    return;
  }

  // find all cpu agents
  std::vector<hsa_agent_t> cpus;
  ASSERT_SUCCESS(hsa_iterate_agents(rocrtst::IterateCPUAgents, &cpus));

  // find all gpu agents
  std::vector<hsa_agent_t> gpus;
  ASSERT_SUCCESS(hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus));

  for (unsigned int i = 0; i < gpus.size(); ++i) {
    hsa_amd_memory_pool_t gpu_pool;
    memset(&gpu_pool, 0, sizeof(gpu_pool));
    ASSERT_SUCCESS(
        hsa_amd_agent_iterate_memory_pools(gpus[i], rocrtst::GetGlobalMemoryPool, &gpu_pool));
    if (gpu_pool.handle == 0) {
      std::cout << "no global mempool in GPU agent" << std::endl;
      return;
    }
    NonContiguousChunks(cpus[0], gpus[i], gpu_pool);
  }
  if (verbosity() > 0) {
    std::cout << "    Subtest finished" << std::endl;
    std::cout << kSubTestSeparator << std::endl;
  }
}

void VirtMemoryTestBasic::SetUp(void) {
  hsa_status_t err;

  TestBase::SetUp();

  ASSERT_SUCCESS(rocrtst::SetDefaultAgents(this));
  ASSERT_SUCCESS(rocrtst::SetPoolsTypical(this));

  return;
}

void VirtMemoryTestBasic::Run(void) {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  TestBase::Run();
}

void VirtMemoryTestBasic::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void VirtMemoryTestBasic::DisplayResults(void) const {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  return;
}

void VirtMemoryTestBasic::Close() {
  // This will close handles opened within rocrtst utility calls and call
  // hsa_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

VirtMemoryTestInterProcess::VirtMemoryTestInterProcess(void) : TestBase() {
  set_title("ROCr Virtual Memory Test - InterProcess ");
  set_description(" Tests Virtual Memory API with memory shared between two processes");
}

VirtMemoryTestInterProcess::~VirtMemoryTestInterProcess(void) {}

// See if the other process wrote an error value to the token; if not, write
// the newVal to the token.
static int CheckAndSetToken(std::atomic<int>* token, int newVal) {
  if (*token == -1) {
    return -1;
  } else {
    *token = newVal;
  }

  return 0;
}

static void ClearShared(SharedVirtMem* s) {
  s->token = 0;
  s->count = 0;
  s->size = 0;
  s->child_status = 0;
  s->parent_status = 0;
  memset(&s->sv, 0, sizeof(s->sv));
}

// Any 1-time setup involving member variables used in the rest of the test
// should be done here.
void VirtMemoryTestInterProcess::SetUp(void) {
  hsa_status_t err;

  // We must fork process before doing HSA stuff, specifically, hsa_init, as
  // each process needs to do this.
  // Allocate linux shared_ memory.
  shared_ = reinterpret_cast<SharedVirtMem*>(mmap(
      nullptr, sizeof(SharedVirtMem), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0));
  ASSERT_NE(shared_, MAP_FAILED) << "mmap failed to allocated shared_ memory";

  // Initialize shared control block to zeros. The field "token"
  // is used to signal state changes between the 2 processes.
  ClearShared(shared_);

  if (socketpair(AF_UNIX, SOCK_DGRAM, 0, shared_->sv) != 0) {
    std::cout << "Failed to create Unix-domain socket pair" << std::endl;
    ASSERT_EQ(0, 1);
  }

  // Spawn second process and verify communication
  child_ = 0;
  child_ = fork();
  ASSERT_NE(-1, child_) << "fork failed";
  std::atomic<int>* token = &shared_->token;
  if (child_ != 0) {
    parentProcess_ = true;

    // Signal to other process we are waiting, and then wait...
    *token = 1;
    while (*token == 1) {
      sched_yield();
    }

    PROCESS_LOG("Second process observed, handshake...\n");
    *token = 1;
    while (*token == 1) {
      sched_yield();
    }

  } else {
    parentProcess_ = false;
    set_verbosity(0);
    PROCESS_LOG("Second process running.\n");

    while (*token == 0) {
      sched_yield();
    }

    int ret;
    ret = CheckAndSetToken(token, 0);
    ASSERT_EQ(0, ret) << "Error detected in child process\n";
    // Wait for handshake
    while (*token == 0) {
      sched_yield();
    }
    ret = CheckAndSetToken(token, 0);
    ASSERT_EQ(0, ret) << "Error detected in child process\n";
  }

  TestBase::SetUp();

  ASSERT_SUCCESS(rocrtst::SetDefaultAgents(this));
  ASSERT_SUCCESS(rocrtst::SetPoolsTypical(this));

  ASSERT_SUCCESS(hsa_amd_memory_pool_get_info(
      device_pool(), HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE, &min_gpu_mem_granule));

  ASSERT_SUCCESS(hsa_amd_memory_pool_get_info(
      device_pool(), HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_REC_GRANULE, &rec_gpu_mem_granule));

  return;
}

void VirtMemoryTestInterProcess::Run(void) {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  TestBase::Run();

  // Note: Close() (and hsa_shut_down()) will be called from main()
  // processOne is true for parent process, false for child process
  if (parentProcess_) {
    ParentProcessImpl();
  } else {
    ChildProcessImpl();
    exit(0);
  }
}

void VirtMemoryTestInterProcess::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void VirtMemoryTestInterProcess::DisplayResults(void) const {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  return;
}

void VirtMemoryTestInterProcess::Close() {
  // This will close handles opened within rocrtst utility calls and call
  // hsa_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

/* Send the dmabuf_fd to another process via Unix socket */
int VirtMemoryTestInterProcess::SendDmaBufFd(int socket, int dmabuf_fd) {
  char* iov_str = (char*)"rocrtst";
  struct msghdr msg = {0};
  char buf[CMSG_SPACE(sizeof(dmabuf_fd))];

  memset(buf, '\0', sizeof(buf));

  struct iovec io = {.iov_base = iov_str, .iov_len = strlen(iov_str)};

  msg.msg_iov = &io;
  msg.msg_iovlen = 1;
  msg.msg_control = buf;
  msg.msg_controllen = sizeof(buf);

  struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_RIGHTS;
  cmsg->cmsg_len = CMSG_LEN(sizeof(dmabuf_fd));

  // memmove(CMSG_DATA(cmsg), &dmabuf_fd, sizeof(dmabuf_fd));
  memcpy(CMSG_DATA(cmsg), &dmabuf_fd, sizeof(dmabuf_fd));

  msg.msg_controllen = CMSG_SPACE(sizeof(dmabuf_fd));

  size_t sent = sendmsg(socket, &msg, 0);

  return (sent < 0) ? -1 : 0;
}

/* Receive the dmabuf_fd to from process via Unix socket */
int VirtMemoryTestInterProcess::ReceiveDmaBufFd(int socket) {
  struct msghdr msg = {0};

  /* On Mac OS X, the struct iovec is needed, even if it points to minimal data */
  char m_buffer[1];
  struct iovec io = {.iov_base = m_buffer, .iov_len = sizeof(m_buffer)};
  msg.msg_iov = &io;
  msg.msg_iovlen = 1;

  char c_buffer[256];
  msg.msg_control = c_buffer;
  msg.msg_controllen = sizeof(c_buffer);

  size_t rcv = recvmsg(socket, &msg, 0);
  if (rcv < 0) return -1;

  struct cmsghdr* cmsg = CMSG_FIRSTHDR(&msg);

  int fd;
  memmove(&fd, CMSG_DATA(cmsg), sizeof(fd));

  return fd;
}

void VirtMemoryTestInterProcess::ParentProcessImpl() {
  hsa_status_t err;

  void* addrRange = NULL;

  bool supp = false;
  ASSERT_SUCCESS(hsa_system_get_info(HSA_AMD_SYSTEM_INFO_VIRTUAL_MEM_API_SUPPORTED, (void*)&supp));
  if (!supp) {
    if (verbosity() > 0) {
      std::cout << "    Virtual Memory API not supported on this system - Skipping." << std::endl;
      std::cout << kSubTestSeparator << std::endl;
    }
    return;
  }

  ASSERT_SUCCESS(hsa_amd_vmem_address_reserve(&addrRange, 20 * rec_gpu_mem_granule, 0, 0));

  hsa_amd_vmem_alloc_handle_t exported_handle;
  ASSERT_SUCCESS(hsa_amd_vmem_handle_create(device_pool(), 20 * rec_gpu_mem_granule,
                                            MEMORY_TYPE_NONE, 0, &exported_handle));

  int dmabuf_fd;
  ASSERT_SUCCESS(hsa_amd_vmem_export_shareable_handle(&dmabuf_fd, exported_handle, 0));
  ASSERT_GE(dmabuf_fd, 0);

  // Signal child process that the gpu buffer is ready to read.
  PROCESS_LOG("Parent: Signalling child proces process\n");
  CheckAndSetToken(&shared_->token, 1);

  close(shared_->sv[1]);
  ASSERT_EQ(SendDmaBufFd(shared_->sv[0], dmabuf_fd), 0);

  hsa_amd_vmem_alloc_handle_t imported_handle;
  ASSERT_SUCCESS(hsa_amd_vmem_import_shareable_handle(dmabuf_fd, &imported_handle));

  /* Test importing same handle twice */
  hsa_amd_vmem_alloc_handle_t imported_handle2;
  ASSERT_SUCCESS(hsa_amd_vmem_import_shareable_handle(dmabuf_fd, &imported_handle2));
  ASSERT_SUCCESS(hsa_amd_vmem_map(addrRange, 10 * rec_gpu_mem_granule, 0, imported_handle, 0));
  ASSERT_SUCCESS(hsa_amd_vmem_unmap(addrRange, 10 * rec_gpu_mem_granule));
  ASSERT_SUCCESS(hsa_amd_vmem_handle_release(imported_handle));
  ASSERT_SUCCESS(hsa_amd_vmem_handle_release(imported_handle2));

  PROCESS_LOG("Parent: Waiting for child process to signal\n");
  while (shared_->token == 1) {
    sched_yield();
  }
  if (shared_->token != 2) {
    shared_->token = -1;
  }
  FORK_ASSERT_EQ(2, shared_->token, "Parent: Error detected in signaling token\n");
  PROCESS_LOG("Parent: Waking upon signal from child process\n");

  ASSERT_SUCCESS(hsa_amd_vmem_handle_release(exported_handle));

  ASSERT_SUCCESS(hsa_amd_vmem_address_free(addrRange, 20 * rec_gpu_mem_granule));

  PROCESS_LOG("Parent: Virtual Memory test PASSED\n");
}

void VirtMemoryTestInterProcess::ChildProcessImpl() {
  int dmabuf_fd = -1;
  bool supp = false;
  hsa_status_t err;
  ASSERT_SUCCESS(hsa_system_get_info(HSA_AMD_SYSTEM_INFO_VIRTUAL_MEM_API_SUPPORTED, (void*)&supp));
  if (!supp) {
    if (verbosity() > 0) {
      std::cout << "    Virtual Memory API not supported on this system - Skipping." << std::endl;
      std::cout << kSubTestSeparator << std::endl;
    }
    return;
  }

  void* addrRange = NULL;
  ASSERT_SUCCESS(hsa_amd_vmem_address_reserve(&addrRange, 20 * rec_gpu_mem_granule, 0, 0));

  // Yield until shared token value changes i.e. is updated by parent.
  // Validate parent's update is per expectation
  PROCESS_LOG("Child: Waiting for parent process to signal\n");
  while (shared_->token == 0) {
    sched_yield();
  }
  if (shared_->token != 1) {
    shared_->token = -1;
  }
  FORK_ASSERT_EQ(1, shared_->token, "Child: Error detected in signaling token\n");
  PROCESS_LOG("Child: Waking upon signal from parent process\n");

  close(shared_->sv[0]);
  dmabuf_fd = ReceiveDmaBufFd(shared_->sv[1]);

  hsa_amd_vmem_alloc_handle_t imported_handle;
  ASSERT_SUCCESS(hsa_amd_vmem_import_shareable_handle(dmabuf_fd, &imported_handle));
  ASSERT_SUCCESS(hsa_amd_vmem_map(addrRange, 10 * rec_gpu_mem_granule, 0, imported_handle, 0));
  ASSERT_SUCCESS(hsa_amd_vmem_unmap(addrRange, 10 * rec_gpu_mem_granule));

  PROCESS_LOG("Child: Signalling parent process\n");
  CheckAndSetToken(&shared_->token, 2);

  ASSERT_SUCCESS(hsa_amd_vmem_handle_release(imported_handle));

  PROCESS_LOG("Child: Virtual Memory test PASSED\n");
}
