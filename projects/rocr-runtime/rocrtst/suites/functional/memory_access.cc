/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2017, Advanced Micro Devices, Inc.
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
#include <memory>

#include "suites/functional/memory_access.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "common/hsatimer.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"


#define RET_IF_HSA_ERR(err) { \
  if ((err) != HSA_STATUS_SUCCESS) { \
    const char* msg = 0; \
    hsa_status_string(err, &msg); \
    std::cout << "hsa api call failure at line " << __LINE__ << ", file: " << \
                          __FILE__ << ". Call returned " << err << std::endl; \
    std::cout << msg << std::endl; \
    return (err); \
  } \
}



MemoryAccessTest::MemoryAccessTest(void) :
    TestBase() {
  set_num_iteration(10);  // Number of iterations to execute of the main test;
                          // This is a default value which can be overridden
                          // on the command line.

  set_title("RocR Memory Access Tests");
  set_description("This series of tests check memory allocation"
    "on GPU and CPU, i.e. GPU access to system memory "
    "and CPU access to GPU memory.");
}

MemoryAccessTest::~MemoryAccessTest(void) {
}

// Any 1-time setup involving member variables used in the rest of the test
// should be done here.
void MemoryAccessTest::SetUp(void) {
  hsa_status_t err;

  TestBase::SetUp();

  err = rocrtst::SetDefaultAgents(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  err = rocrtst::SetPoolsTypical(this);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  return;
}

void MemoryAccessTest::Run(void) {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  TestBase::Run();
}

void MemoryAccessTest::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void MemoryAccessTest::DisplayResults(void) const {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  return;
}

void MemoryAccessTest::Close() {
  // This will close handles opened within rocrtst utility calls and call
  // hsa_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}





typedef struct  __attribute__ ((aligned(16)))  args_t {
     int *a;
     int *b;
     int *c;
  } args;

  args *kernArgs = NULL;

static const char kSubTestSeparator[] = "  **************************";

static void PrintMemorySubtestHeader(const char *header) {
  std::cout << "  *** Memory Subtest: " << header << " ***" << std::endl;
}

#if ROCRTST_EMULATOR_BUILD
static const int kMemoryAllocSize = 8;
#else
static const int kMemoryAllocSize = 1024;
#endif


// Test to check GPU can read & write to system memory
void MemoryAccessTest::GPUAccessToCPUMemoryTest(hsa_agent_t cpuAgent,
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
    hsa_signal_t signal = {0};  // completion signal


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
    // (sys_data,dup_sys_data,cpuResult,kernArg) on system memory
    int *sys_data = NULL;
    int *dup_sys_data = NULL;
    int *cpuResult = NULL;
    int *gpuResult = NULL;

    err = hsa_amd_memory_pool_allocate(global_pool,
                                      kMemoryAllocSize*sizeof(int), 0,
                                      reinterpret_cast<void **>(&cpuResult));
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    err = hsa_amd_memory_pool_allocate(global_pool,
                                      kMemoryAllocSize*sizeof(int), 0,
                                      reinterpret_cast<void **>(&sys_data));
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    err = hsa_amd_memory_pool_allocate(global_pool,
                                      kMemoryAllocSize*sizeof(int), 0,
                                      reinterpret_cast<void **>(&dup_sys_data));
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);


    // Allocate the kernel argument buffer from the kernarg_pool.
    err = hsa_amd_memory_pool_allocate(kernarg_pool, sizeof(args_t), 0,
                                        reinterpret_cast<void **>(&kernArgs));
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    // initialize the host buffers
    for (int i = 0; i < kMemoryAllocSize; ++i) {
      unsigned int seed = time(NULL);
      sys_data[i] = 1 + rand_r(&seed) % 1;
      dup_sys_data[i] = sys_data[i];
    }

    memset(cpuResult, 0, kMemoryAllocSize * sizeof(int));

    // for the dGPU, we have coarse grained local memory,
    // so allocate memory for it on the GPU's GLOBAL segment .

    // Get local memory of GPU to allocate device side buffers

    err = hsa_amd_memory_pool_allocate(gpu_pool,
      kMemoryAllocSize*sizeof(int), 0, reinterpret_cast<void **>(&gpuResult));
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);


    // Allow cpuAgent access to all allocated GPU memory.
    err = hsa_amd_agents_allow_access(1, &cpuAgent, NULL, gpuResult);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
    memset(gpuResult, 0, kMemoryAllocSize * sizeof(int));

    // Allow gpuAgent access to all allocated system memory.
    err = hsa_amd_agents_allow_access(1, &gpuAgent, NULL, cpuResult);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
    err = hsa_amd_agents_allow_access(1, &gpuAgent, NULL, sys_data);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
    err = hsa_amd_agents_allow_access(1, &gpuAgent, NULL, dup_sys_data);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
    err = hsa_amd_agents_allow_access(1, &gpuAgent, NULL, kernArgs);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    kernArgs->a = sys_data;
    kernArgs->b = cpuResult;  // system memory passed to gpu for write
    kernArgs->c = gpuResult;  // gpu memory to verify that gpu read system data


    // Create the executable, get symbol by name and load the code object
    set_kernel_file_name("gpuReadWrite_kernels.hsaco");
    set_kernel_name("gpuReadWrite");
    err = rocrtst::LoadKernelFromObjFile(this, &gpuAgent);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    // Fill the dispatch packet with
    // workgroup_size, grid_size, kernelArgs and completion signal
    // Put it on the queue and launch the kernel by ringing the doorbell

    // create completion signal
    err = hsa_signal_create(1, 0, NULL, &signal);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

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
    aql.kernarg_address = kernArgs;
    aql.completion_signal = signal;

    // const uint32_t queue_size = queue->size;
    const uint32_t queue_mask = queue->size - 1;

    // write to command queue
    uint64_t index = hsa_queue_load_write_index_relaxed(queue);
    hsa_queue_store_write_index_relaxed(queue, index + 1);

    rocrtst::WriteAQLToQueueLoc(queue, index, &aql);

    hsa_kernel_dispatch_packet_t *q_base_addr =
        reinterpret_cast<hsa_kernel_dispatch_packet_t *>(queue->base_address);
    rocrtst::AtomicSetPacketHeader(
        (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
           (1 << HSA_PACKET_HEADER_BARRIER) |
          (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE) |
           (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE),
                  (1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS),
        reinterpret_cast<hsa_kernel_dispatch_packet_t *>
                                          (&q_base_addr[index & queue_mask]));

    // ringdoor bell
    hsa_signal_store_relaxed(queue->doorbell_signal, index);
    // wait for the signal and reset it for future use
    while (hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1,
                                      (uint64_t)-1, HSA_WAIT_STATE_ACTIVE)) { }
    hsa_signal_store_relaxed(signal, 1);

    // compare device and host side results
    if (verbosity() > 0) {
      std::cout<< "check gpu has read the system memory"<< std::endl;
    }
    for (int i = 0; i < kMemoryAllocSize; ++i) {
      ASSERT_EQ(gpuResult[i], dup_sys_data[i]);
    }

    if (verbosity() > 0) {
      std::cout<< "gpu has read the system memory successfully"<< std::endl;
      std::cout<< "check gpu has written to system memory"<< std::endl;
    }
    for (int i = 0; i < kMemoryAllocSize; ++i) {
      ASSERT_EQ(cpuResult[i], i);
    }

    if (verbosity() > 0) {
      std::cout<< "gpu has written to system memory successfully"<< std::endl;
    }

    if (sys_data) { hsa_amd_memory_pool_free(sys_data); }
    if (dup_sys_data) { hsa_amd_memory_pool_free(dup_sys_data); }
    if (cpuResult) {hsa_amd_memory_pool_free(cpuResult); }
    if (gpuResult) {hsa_amd_memory_pool_free(gpuResult); }
    if (kernArgs) { hsa_amd_memory_pool_free(kernArgs); }
    if (signal.handle) { hsa_signal_destroy(signal); }
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

// Test to check cpu can read & write to GPU memory
void MemoryAccessTest::CPUAccessToGPUMemoryTest(hsa_agent_t cpuAgent,
                                                 hsa_agent_t gpuAgent,
                                                 hsa_amd_memory_pool_t pool) {
  hsa_status_t err;

  rocrtst::pool_info_t pool_i;
  err = rocrtst::AcquirePoolInfo(pool, &pool_i);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  if (pool_i.segment == HSA_AMD_SEGMENT_GLOBAL &&
        pool_i.global_flag == HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) {
    hsa_amd_memory_pool_access_t access;
    hsa_amd_agent_memory_pool_get_info(cpuAgent, pool,
                                         HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS,
                                         &access);
    if (access != HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
      if (!pool_i.alloc_allowed || pool_i.alloc_granule == 0 ||
                                           pool_i.alloc_alignment == 0) {
        if (verbosity() > 0) {
          std::cout << "  Test not applicable. Skipping." << std::endl;
          std::cout << kSubTestSeparator << std::endl;
        }
        return;
      }


      auto gran_sz = pool_i.alloc_granule;
      auto pool_sz = pool_i.size / gran_sz;
      auto max_alloc_size = pool_sz/2;
      unsigned int max_element = max_alloc_size/sizeof(unsigned int);
      unsigned int *gpu_data;
      unsigned int *sys_data;
      sys_data = (unsigned int*)malloc(max_alloc_size);

      ASSERT_NE(sys_data, nullptr);

      for (unsigned int i = 0; i < max_element; ++i) {
        sys_data[i] = i;
      }
      // err = hsa_amd_agents_allow_access(1, &gpuAgent, NULL, sys_data);
      // EXPECT_EQ(err, HSA_STATUS_SUCCESS);
      err = hsa_amd_memory_pool_allocate(pool, max_alloc_size, 0,
                                          reinterpret_cast<void**>(&gpu_data));
      ASSERT_EQ(err, HSA_STATUS_SUCCESS);
      /*
      if (err == HSA_STATUS_ERROR) {
        err = hsa_amd_memory_pool_free(gpu_data);
      }*/

      err = hsa_amd_agents_allow_access(1, &cpuAgent, NULL, gpu_data);
      ASSERT_EQ(err, HSA_STATUS_SUCCESS);
      // EXPECT_EQ(HSA_STATUS_SUCCESS, err);

      // Verify CPU can read & write to GPU memory
      std::cout<< "Verify CPU can read & write to GPU memory"<< std::endl;
      for (unsigned int i = 0; i < max_element; ++i) {
        gpu_data[i] = i;  // Write to gpu memory directly
      }

     for (unsigned int  i = 0; i < max_element; ++i) {
       if (sys_data[i] != gpu_data[i]) {  // Reading GPU memory
            fprintf(stdout, "Values not mathing !! sys_data[%d]:%d ,"
                "gpu_data[%d]\n", sys_data[i], i, gpu_data[i]);
       }
     }
     std::cout<< "CPU have read & write to GPU memory successfully"<< std::endl;
     err = hsa_amd_memory_pool_free(gpu_data);
     free(sys_data);
     } else {
        if (verbosity() > 0) {
          std::cout<< "Test not applicable as system is not large bar."
                         "Skipping."<< std::endl;
          std::cout << kSubTestSeparator << std::endl;
        }
        return;
    }
  }
}


void MemoryAccessTest::CPUAccessToGPUMemoryTest(void) {
  hsa_status_t err;

  PrintMemorySubtestHeader("CPUAccessToGPUMemoryTest in Memory Pools");
  // find all cpu agents
  std::vector<hsa_agent_t> cpus;
  err = hsa_iterate_agents(rocrtst::IterateCPUAgents, &cpus);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  // find all gpu agents
  std::vector<hsa_agent_t> gpus;
  err = hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  for (unsigned int i = 0 ; i< gpus.size(); ++i) {
    hsa_amd_memory_pool_t gpu_pool;
    memset(&gpu_pool, 0, sizeof(gpu_pool));
    err = hsa_amd_agent_iterate_memory_pools(gpus[i],
                                              rocrtst::GetGlobalMemoryPool,
                                              &gpu_pool);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
    if (gpu_pool.handle == 0) {
      std::cout << "no global mempool in gpu agent" << std::endl;
      return;
    }
    CPUAccessToGPUMemoryTest(cpus[0], gpus[i], gpu_pool);
  }
  if (verbosity() > 0) {
    std::cout << "subtest Passed" << std::endl;
    std::cout << kSubTestSeparator << std::endl;
  }
}

void MemoryAccessTest::GPUAccessToCPUMemoryTest(void) {
  hsa_status_t err;

  PrintMemorySubtestHeader("GPUAccessToCPUMemoryTest in Memory Pools");
  // find all cpu agents
  std::vector<hsa_agent_t> cpus;
  err = hsa_iterate_agents(rocrtst::IterateCPUAgents, &cpus);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  // find all gpu agents
  std::vector<hsa_agent_t> gpus;
  err = hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  for (unsigned int i = 0 ; i< gpus.size(); ++i) {
    GPUAccessToCPUMemoryTest(cpus[0], gpus[i]);
  }

  if (verbosity() > 0) {
    std::cout << "subtest Passed" << std::endl;
    std::cout << kSubTestSeparator << std::endl;
  }
}

#undef RET_IF_HSA_ERR
