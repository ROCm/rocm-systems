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


#include <fcntl.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include <string>
#include <memory>

#include "suites/functional/memory_allocation.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "common/hsatimer.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

static const uint32_t kNumBufferElements = 256;
static const int kValue = 5;


MemoryAllocationTest::MemoryAllocationTest(bool launch_GroupMemory,
                                           bool launch_BasicAllocateFree) : TestBase() {
  set_num_iteration(10);  // Number of iterations to execute of the main test;
                          // This is a default value which can be overridden
                          // on the command line.
  std::string name;
  std::string desc;

  name = "RocR Memory Test ";
  if (launch_GroupMemory) {
    name += " For Kernel Dynamic Memory Alocation";
    desc += " This test Allocate group memory in kernel dynamically.";
  } else if (launch_BasicAllocateFree) {
    name += " For BasicAllocateFree";
    desc += " This test Allocate And free Memory on all the availble pool "
            " on which allocation is allowed on RocR Agents.";
  }
  set_title(name);
  set_description(desc);

  memset(&aql(), 0, sizeof(hsa_kernel_dispatch_packet_t));
}

MemoryAllocationTest::~MemoryAllocationTest(void) {
}

// Any 1-time setup involving member variables used in the rest of the test
// should be done here.
void MemoryAllocationTest::SetUp(void) {
  hsa_status_t err;

  TestBase::SetUp();

  err = rocrtst::SetDefaultAgents(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  err = rocrtst::SetPoolsTypical(this);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  // Fill up the kernel packet except header
  err = rocrtst::InitializeAQLPacket(this, &aql());
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);
  return;
}

void MemoryAllocationTest::Run(void) {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  TestBase::Run();
}

void MemoryAllocationTest::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void MemoryAllocationTest::DisplayResults(void) const {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  return;
}

void MemoryAllocationTest::Close() {
  // This will close handles opened within rocrtst utility calls and call
  // hsa_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

void MemoryAllocationTest::WriteAQLPktToQueue(hsa_queue_t* q) {
  void* queue_base = q->base_address;
  const uint32_t queue_mask = q->size - 1;
  uint64_t index = hsa_queue_add_write_index_relaxed(q, 1);

      reinterpret_cast<hsa_kernel_dispatch_packet_t *>(
                                     queue_base)[index & queue_mask] = aql();
}





typedef struct  __attribute__ ((aligned(16)))  args_t {
     uint32_t *a;
     uint32_t *b;
     uint32_t grp_offset;
     uint32_t count;
  } args;


static const char kSubTestSeparator[] = "  **************************";

static void PrintMemorySubtestHeader(const char *header) {
  std::cout << "  *** Memory Allocation  Test: " << header << " ***" << std::endl;
}

static const int kMemoryAllocSize = 1024;

void MemoryAllocationTest::GroupMemoryDynamicAllocation(hsa_agent_t cpuAgent,
                                                   hsa_agent_t gpuAgent) {
  hsa_status_t err;

  // Get Global Memory Pool on the gpuAgent to allocate gpu buffers
  hsa_amd_memory_pool_t gpu_pool;
  err = hsa_amd_agent_iterate_memory_pools(gpuAgent,
                                            rocrtst::GetGlobalMemoryPool,
                                            &gpu_pool);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  hsa_amd_memory_pool_access_t access;
  hsa_amd_agent_memory_pool_get_info(cpuAgent, gpu_pool,
                                       HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS,
                                       &access);
  if (access != HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
    // hsa objects
    hsa_queue_t *queue = NULL;  // command queue

    // get queue size
    uint32_t queue_size = 0;
    err = hsa_agent_get_info(gpuAgent,
                                HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    // create queue
    err = hsa_queue_create(gpuAgent,
                              queue_size, HSA_QUEUE_TYPE_MULTI,
                              NULL, NULL, 0, 0, &queue);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    // Get System Memory Pool on the cpuAgent to allocate host side buffers
    hsa_amd_memory_pool_t global_pool;
    err = hsa_amd_agent_iterate_memory_pools(cpuAgent,
                                              rocrtst::GetGlobalMemoryPool,
                                              &global_pool);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    // Find a memory pool that supports kernel arguments.
    hsa_amd_memory_pool_t kernarg_pool;
    err = hsa_amd_agent_iterate_memory_pools(cpuAgent,
                                              rocrtst::GetKernArgMemoryPool,
                                              &kernarg_pool);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    // Allocate the host side buffers
    // (Indata,kernArg) on system memory
    uint32_t *Indata = NULL;
    args *kernArgs = NULL;

    err = hsa_amd_memory_pool_allocate(global_pool,
                                      kMemoryAllocSize*sizeof(uint32_t), 0,
                                      reinterpret_cast<void **>(&Indata));
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);


    // Allocate the kernel argument buffer from the kernarg_pool.
    err = hsa_amd_memory_pool_allocate(kernarg_pool, sizeof(args_t), 0,
                                        reinterpret_cast<void **>(&kernArgs));
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    // initialize the host buffers
    for (int i = 0; i < kMemoryAllocSize; ++i) {
      // unsigned int seed = time(NULL);
      Indata[i] = i;
    }

    // for the dGPU, we have coarse grained local memory,
    // so allocate memory for it on the GPU's GLOBAL segment .

    // Get local memory of GPU to allocate device side buffers
    uint32_t *OutData = NULL;
    err = hsa_amd_memory_pool_allocate(gpu_pool, kMemoryAllocSize*sizeof(uint32_t), 0,
                                        reinterpret_cast<void **>(&OutData));
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);


    // Allow cpuAgent access to all allocated GPU memory.
    err = hsa_amd_agents_allow_access(1, &cpuAgent, NULL, OutData);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
    memset(OutData, 0, kMemoryAllocSize * sizeof(int));

    // Allow gpuAgent access to all allocated system memory.
    err = hsa_amd_agents_allow_access(1, &gpuAgent, NULL, Indata);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
    err = hsa_amd_agents_allow_access(1, &gpuAgent, NULL, kernArgs);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    uint32_t grp_offset = group_segment_size();
    kernArgs->a = Indata;
    // gpu memory where data will be copied from dynamically group memory
    kernArgs->b = OutData;
    kernArgs->grp_offset = grp_offset;
    kernArgs->count = kMemoryAllocSize;

    // Fill up the kernel packet except header
    err = rocrtst::InitializeAQLPacket(this, &aql());
    ASSERT_EQ(HSA_STATUS_SUCCESS, err);

    // Create the executable, get symbol by name and load the code object
    set_kernel_file_name("groupMemoryDynamic_kernels.hsaco");
    set_kernel_name("group_memory_dynamic");
    err = rocrtst::LoadKernelFromObjFile(this, &gpuAgent);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    // The total byte size of group memory, static + dynamic
    uint32_t total_grp_byte_size = group_segment_size() + kMemoryAllocSize * sizeof(uint32_t);
    if (verbosity() > 0) {
      std::cout << "aql.total_grp_byte_size" << total_grp_byte_size << std::endl;
    }

    // Fill up the kernel packet except header
    err = rocrtst::InitializeAQLPacket(this, &aql());
    ASSERT_EQ(HSA_STATUS_SUCCESS, err);

    aql().workgroup_size_x = 256;
    aql().workgroup_size_y = 1;
    aql().workgroup_size_z = 1;
    aql().grid_size_y = 1;
    aql().grid_size_z = 1;
    aql().private_segment_size = 0;
    aql().grid_size_x = kMemoryAllocSize;
    aql().group_segment_size = total_grp_byte_size;
    aql().kernel_object = kernel_object();
    aql().kernarg_address = kernArgs;

    const uint32_t queue_mask = queue->size - 1;

    // Load index for writing header later to command queue at same index
    uint64_t index = hsa_queue_load_write_index_relaxed(queue);
    hsa_queue_store_write_index_relaxed(queue, index + 1);

    // This function simply copies the data we've collected so far into our
    // local AQL packet, except the the setup and header fields.
    rocrtst::WriteAQLToQueueLoc(queue, index, &aql());

    aql().header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
    aql().header |= HSA_FENCE_SCOPE_SYSTEM <<
                 HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
    aql().header |= HSA_FENCE_SCOPE_SYSTEM <<
                 HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;

    void* q_base = queue->base_address;
    // Set the Aql packet header
    rocrtst::AtomicSetPacketHeader(aql().header, aql().setup,
                        &(reinterpret_cast<hsa_kernel_dispatch_packet_t*>
                            (q_base))[index & queue_mask]);

    // ringdoor bell
    hsa_signal_store_relaxed(queue->doorbell_signal, index);

    // wait for the signal and reset it for future use
    while (hsa_signal_wait_scacquire(aql().completion_signal, HSA_SIGNAL_CONDITION_LT, 1,
                                      (uint64_t)-1, HSA_WAIT_STATE_ACTIVE)) { }

    hsa_signal_store_relaxed(aql().completion_signal, 1);

    // compare Results
    for (int i = 0; i < kMemoryAllocSize; ++i) {
      if (verbosity() > 0) {
        // std::cout<< i << "OutData[i]" << OutData[i] << "Indata[i]" << Indata[i] <<std::endl;
      }
      ASSERT_EQ(OutData[i], Indata[i]);
    }
    if (Indata) { hsa_amd_memory_pool_free(Indata); }
    if (OutData) { hsa_amd_memory_pool_free(OutData); }
    if (kernArgs) { hsa_amd_memory_pool_free(kernArgs); }
    if (queue) { hsa_queue_destroy(queue); }
  } else {
    if (verbosity() > 0) {
      std::cout<< "Test not applicable as system is not large bar."
                   "Skipping."<< std::endl;
      std::cout << kSubTestSeparator << std::endl;
    }
    return;
  }
}



void MemoryAllocationTest::GroupMemoryDynamicAllocation(void) {
  hsa_status_t err;
  if (verbosity() > 0) {
    PrintMemorySubtestHeader("Memory Group dynamic allocation");
  }
  // find all cpu agents
  std::vector<hsa_agent_t> cpus;
  err = hsa_iterate_agents(rocrtst::IterateCPUAgents, &cpus);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  // find all gpu agents
  std::vector<hsa_agent_t> gpus;
  err = hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  for (unsigned int i = 0 ; i< gpus.size(); ++i) {
    GroupMemoryDynamicAllocation(cpus[0], gpus[i]);
  }

  if (verbosity() > 0) {
    std::cout << "subtest Passed" << std::endl;
    std::cout << kSubTestSeparator << std::endl;
  }
}


static void PrintAgentNameAndType(hsa_agent_t agent) {
  hsa_status_t err;

  char ag_name[64];
  hsa_device_type_t ag_type;

  err = hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, ag_name);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  err = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &ag_type);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  std::cout << "  Agent: " << ag_name << " (";
  switch (ag_type) {
    case HSA_DEVICE_TYPE_CPU:
      std::cout << "CPU)";
      break;
    case HSA_DEVICE_TYPE_GPU:
      std::cout << "GPU)";
      break;
    case HSA_DEVICE_TYPE_DSP:
      std::cout << "DSP)";
      break;
    case HSA_DEVICE_TYPE_AIE:
      std::cout << "AIE)";
      break;
    }
  std::cout << std::endl;
  return;
}

static void PrintSegmentNameAndType(uint32_t segment) {
  switch (segment) {
    case HSA_AMD_SEGMENT_GLOBAL:
      std::cout << "  GLOBAL SEGMENT";
      break;
    case HSA_AMD_SEGMENT_GROUP:
      std::cout << "  GROUP SEGMENT";
      break;
    case HSA_AMD_SEGMENT_PRIVATE:
      std::cout << "  PRIVATE SEGMENT";
      break;
    case HSA_AMD_SEGMENT_READONLY:
      std::cout << "  READONLY SEGMENT";
      break;
    default:
      std::cout << "  no segment";
      break;
    }
  std::cout << std::endl;
  return;
}

void MemoryAllocationTest::MemoryBasicAllocationAndFree(hsa_agent_t agent,
                                               hsa_amd_memory_pool_t pool) {
  hsa_status_t err;

  rocrtst::pool_info_t pool_i;
  err = rocrtst::AcquirePoolInfo(pool, &pool_i);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  if (verbosity() > 0) {
    PrintAgentNameAndType(agent);
  }

  // if allocation is allowed in this pool allocate the memory
  // and then free it
  if (pool_i.alloc_allowed) {
    if (verbosity() > 0) {
      PrintSegmentNameAndType(pool_i.segment);
    }
    size_t max_size;
    err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE,
                                      &max_size);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
    max_size = (max_size > kMemoryAllocSize) ? kMemoryAllocSize : max_size;

    char *memoryPtr;
    err = hsa_amd_memory_pool_allocate(pool, max_size , 0,
                                       reinterpret_cast<void**>(&memoryPtr));
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
    if (memoryPtr) {
      err = hsa_amd_memory_pool_free(memoryPtr);
      ASSERT_EQ(err, HSA_STATUS_SUCCESS);
    }
  }
  return;
}


void MemoryAllocationTest::MemoryBasicAllocationAndFree(void) {
  hsa_status_t err;
  std::vector<std::shared_ptr<rocrtst::agent_pools_t>> agent_pools;
  if (verbosity() > 0) {
    PrintMemorySubtestHeader("MemoryBasicAllocationAndFree");
  }

  err = rocrtst::GetAgentPools(&agent_pools);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  auto pool_idx = 0;
  for (auto a : agent_pools) {
    for (auto p : a->pools) {
      if (verbosity() > 0) {
        std::cout << "  Pool " << pool_idx++ << ":" << std::endl;
      }
      MemoryBasicAllocationAndFree(a->agent, p);
    }
  }

  if (verbosity() > 0) {
    std::cout << "subtest Passed" << std::endl;
    std::cout << kSubTestSeparator << std::endl;
  }
}

void MemoryAllocationTest::MemoryAllocateContiguousTest(hsa_agent_t agent,
                                                        hsa_amd_memory_pool_t pool) {
  rocrtst::pool_info_t pool_i;
  hsa_device_type_t ag_type;
  ASSERT_SUCCESS(rocrtst::AcquirePoolInfo(pool, &pool_i));

  if (verbosity() > 0) PrintAgentNameAndType(agent);

  ASSERT_SUCCESS(hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &ag_type));

  // if allocation is allowed in this pool allocate the memory
  // and then free it
  if (ag_type != HSA_DEVICE_TYPE_GPU || !pool_i.alloc_allowed || !pool_i.alloc_granule ||
      !pool_i.alloc_alignment) {
    return;
  }

  if (verbosity() > 0) PrintSegmentNameAndType(pool_i.segment);

  // find all gpu agents
  std::vector<hsa_agent_t> gpus;
  ASSERT_SUCCESS(hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus));

  const size_t alloc_size = pool_i.alloc_granule * 1024;

  char* memoryPtr;

  ASSERT_SUCCESS(hsa_amd_memory_pool_allocate(pool, alloc_size, HSA_AMD_MEMORY_POOL_CONTIGUOUS_FLAG,
                                              reinterpret_cast<void**>(&memoryPtr)));
  if (!memoryPtr) return;

  int dmabuf = -1;
  uint64_t offset;
  ASSERT_SUCCESS(hsa_amd_portable_export_dmabuf(memoryPtr, alloc_size, &dmabuf, &offset));

  std::vector<hsa_agent_t> accessible_gpus;
  for (auto gpuIter: gpus) {
    hsa_amd_memory_pool_access_t access = HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED;
    ASSERT_SUCCESS(hsa_amd_agent_memory_pool_get_info(gpuIter, pool, HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS, &access));
    if (access != HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED)
      accessible_gpus.push_back(gpuIter);
  }

  void* importedPtr = nullptr;
  size_t importedSz;

  ASSERT_SUCCESS(hsa_amd_interop_map_buffer(accessible_gpus.size(), accessible_gpus.data(), dmabuf, 0, &importedSz,
                                                   &importedPtr, 0, NULL));

  ASSERT_NE(importedPtr, nullptr);
  ASSERT_EQ(importedSz, alloc_size);

  close(dmabuf);

  ASSERT_SUCCESS(hsa_amd_interop_unmap_buffer(importedPtr));

  ASSERT_SUCCESS(hsa_amd_memory_pool_free(memoryPtr));
  return;
}

void MemoryAllocationTest::MemoryAllocateContiguousTest(void) {
  hsa_status_t err;
  std::vector<std::shared_ptr<rocrtst::agent_pools_t>> agent_pools;
  if (verbosity() > 0) {
    PrintMemorySubtestHeader("MemoryAllocateContiguousTest");
  }

  ASSERT_SUCCESS(rocrtst::GetAgentPools(&agent_pools));

  auto pool_idx = 0;
  for (auto a : agent_pools) {
    for (auto p : a->pools) {
      if (verbosity() > 0) {
        std::cout << "  Pool " << pool_idx++ << ":" << std::endl;
      }
      MemoryAllocateContiguousTest(a->agent, p);
    }
  }

  if (verbosity() > 0) {
    std::cout << "subtest Passed" << std::endl;
    std::cout << kSubTestSeparator << std::endl;
  }
}
