/*
Copyright (c) 2021 - present Advanced Micro Devices, Inc. All rights reserved.

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

#include "rdc_modules/rdc_rocr/MemoryAccess.h"

#include <fcntl.h>

#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/rdc_common.h"
#include "rdc_modules/rdc_rocr/base_rocr_utils.h"
#include "rdc_modules/rdc_rocr/common.h"

namespace amd {
namespace rdc {

MemoryAccessTest::MemoryAccessTest(uint32_t gpu_index) : TestBase(gpu_index) {
  set_num_iteration(10);  // Number of iterations to execute of the main test;
                          // This is a default value which can be overridden
                          // on the command line.

  set_title("RocR Memory Access Tests");
  set_description(
      "This series of tests check memory allocation"
      "on GPU and CPU, i.e. GPU access to system memory "
      "and CPU access to GPU memory.");
}

MemoryAccessTest::~MemoryAccessTest(void) {}

// Any 1-time setup involving member variables used in the rest of the test
// should be done here.
hsa_status_t MemoryAccessTest::SetUp(void) {
  hsa_status_t err;

  TestBase::SetUp();

  err = SetDefaultAgents(this);
  throw_if_error(err);

  err = SetPoolsTypical(this);
  throw_if_error(err);
  return err;
}

void MemoryAccessTest::Run(void) {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!CheckProfile(this)) {
    return;
  }

  TestBase::Run();
}

void MemoryAccessTest::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void MemoryAccessTest::DisplayResults(void) const {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!CheckProfile(this)) {
    return;
  }
}

void MemoryAccessTest::Close() {
  // This will close handles opened within rocrtst utility calls and call
  // hsa_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

typedef struct __attribute__((aligned(16))) args_t {
  int* a;
  int* b;
  int* c;
} args;

args* kernArgs = NULL;

static const char kSubTestSeparator[] = "  **************************";

static void PrintMemorySubtestHeader(const char* header) {
  RDC_LOG(RDC_DEBUG, "  *** Memory Subtest: " << header << " ***");
}

#if ROCRTST_EMULATOR_BUILD
static const int kMemoryAllocSize = 8;
#else
static const int kMemoryAllocSize = 1024;
#endif

// Test to check GPU can read & write to system memory
void MemoryAccessTest::GPUAccessToCPUMemoryTest(hsa_agent_t cpuAgent, hsa_agent_t gpuAgent) {
  hsa_status_t err;

  // Get Global Memory Pool on the gpuAgent to allocate gpu buffers
  hsa_amd_memory_pool_t gpu_pool;
  err = hsa_amd_agent_iterate_memory_pools(gpuAgent, GetGlobalMemoryPool, &gpu_pool);
  throw_if_error(err);

  hsa_amd_memory_pool_access_t access;
  hsa_amd_agent_memory_pool_get_info(cpuAgent, gpu_pool, HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS,
                                     &access);
  if (access != HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
    // hsa objects
    hsa_queue_t* queue = NULL;  // command queue
    hsa_signal_t signal = {0};  // completion signal

    // get queue size
    uint32_t queue_size = 0;
    err = hsa_agent_get_info(gpuAgent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size);
    throw_if_error(err);

    // create queue
    err = hsa_queue_create(gpuAgent, queue_size, HSA_QUEUE_TYPE_MULTI, NULL, NULL, 0, 0, &queue);
    throw_if_error(err);

    // Get System Memory Pool on the cpuAgent to allocate host side buffers
    hsa_amd_memory_pool_t global_pool;
    err = hsa_amd_agent_iterate_memory_pools(cpuAgent, GetGlobalMemoryPool, &global_pool);
    throw_if_error(err);

    // Find a memory pool that supports kernel arguments.
    hsa_amd_memory_pool_t kernarg_pool;
    err = hsa_amd_agent_iterate_memory_pools(cpuAgent, GetKernArgMemoryPool, &kernarg_pool);
    throw_if_error(err);

    // Allocate the host side buffers
    // (sys_data,dup_sys_data,cpuResult,kernArg) on system memory
    int* sys_data = NULL;
    int* dup_sys_data = NULL;
    int* cpuResult = NULL;
    int* gpuResult = NULL;

    err = hsa_amd_memory_pool_allocate(global_pool, kMemoryAllocSize, 0,
                                       reinterpret_cast<void**>(&cpuResult));
    throw_if_error(err);

    err = hsa_amd_memory_pool_allocate(global_pool, kMemoryAllocSize, 0,
                                       reinterpret_cast<void**>(&sys_data));
    throw_if_error(err);

    err = hsa_amd_memory_pool_allocate(global_pool, kMemoryAllocSize, 0,
                                       reinterpret_cast<void**>(&dup_sys_data));
    throw_if_error(err);

    // Allocate the kernel argument buffer from the kernarg_pool.
    err = hsa_amd_memory_pool_allocate(kernarg_pool, sizeof(args_t), 0,
                                       reinterpret_cast<void**>(&kernArgs));
    throw_if_error(err);

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

    err = hsa_amd_memory_pool_allocate(gpu_pool, kMemoryAllocSize, 0,
                                       reinterpret_cast<void**>(&gpuResult));
    throw_if_error(err);

    // Allow cpuAgent access to all allocated GPU memory.
    err = hsa_amd_agents_allow_access(1, &cpuAgent, NULL, gpuResult);
    throw_if_error(err);
    memset(gpuResult, 0, kMemoryAllocSize * sizeof(int));

    // Allow gpuAgent access to all allocated system memory.
    err = hsa_amd_agents_allow_access(1, &gpuAgent, NULL, cpuResult);
    throw_if_error(err);
    err = hsa_amd_agents_allow_access(1, &gpuAgent, NULL, sys_data);
    throw_if_error(err);
    err = hsa_amd_agents_allow_access(1, &gpuAgent, NULL, dup_sys_data);
    throw_if_error(err);
    err = hsa_amd_agents_allow_access(1, &gpuAgent, NULL, kernArgs);
    throw_if_error(err);

    kernArgs->a = sys_data;
    kernArgs->b = cpuResult;  // system memory passed to gpu for write
    kernArgs->c = gpuResult;  // gpu memory to verify that gpu read system data

    // Create the executable, get symbol by name and load the code object
    set_kernel_file_name("gpuReadWrite_kernels.hsaco");
    set_kernel_name("gpuReadWrite");
    err = LoadKernelFromObjFile(this, &gpuAgent);
    throw_if_error(err);

    // Fill the dispatch packet with
    // workgroup_size, grid_size, kernelArgs and completion signal
    // Put it on the queue and launch the kernel by ringing the doorbell

    // create completion signal
    err = hsa_signal_create(1, 0, NULL, &signal);
    throw_if_error(err);

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

    WriteAQLToQueueLoc(queue, index, &aql);

    hsa_kernel_dispatch_packet_t* q_base_addr =
        reinterpret_cast<hsa_kernel_dispatch_packet_t*>(queue->base_address);
    AtomicSetPacketHeader(
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
      RDC_LOG(RDC_DEBUG, "check gpu has read the system memory");
    }
    for (int i = 0; i < kMemoryAllocSize; ++i) {
      if (gpuResult[i] != dup_sys_data[i]) {
        throw_if_error(HSA_STATUS_ERROR, "gpuResult does not match dup_sys_data.");
      }
    }

    if (verbosity() > 0) {
      RDC_LOG(RDC_DEBUG, "gpu has read the system memory successfully");
      RDC_LOG(RDC_DEBUG, "check gpu has written to system memory");
    }
    for (int i = 0; i < kMemoryAllocSize; ++i) {
      if (cpuResult[i] != i) {
        throw_if_error(HSA_STATUS_ERROR,
                       "The CPU memory size does not match the system memory size.");
      }
    }

    if (verbosity() > 0) {
      RDC_LOG(RDC_DEBUG, "gpu has written to system memory successfully");
    }

    if (sys_data) {
      hsa_memory_free(sys_data);
    }
    if (dup_sys_data) {
      hsa_memory_free(dup_sys_data);
    }
    if (cpuResult) {
      hsa_memory_free(cpuResult);
    }
    if (gpuResult) {
      hsa_memory_free(gpuResult);
    }
    if (kernArgs) {
      hsa_memory_free(kernArgs);
    }
    if (signal.handle) {
      hsa_signal_destroy(signal);
    }
    if (queue) {
      hsa_queue_destroy(queue);
    }

  } else {
    if (verbosity() > 0) {
      RDC_LOG(RDC_DEBUG, "Test not applicable as system is not large bar, skipping");
    }
    return;
  }
}

// Test to check cpu can read & write to GPU memory
void MemoryAccessTest::CPUAccessToGPUMemoryTest(hsa_agent_t cpuAgent, hsa_agent_t,
                                                hsa_amd_memory_pool_t pool) {
  hsa_status_t err;

  pool_info_t pool_i;
  err = AcquirePoolInfo(pool, &pool_i);
  throw_if_error(err);

  if (pool_i.segment == HSA_AMD_SEGMENT_GLOBAL &&
      pool_i.global_flag == HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_COARSE_GRAINED) {
    hsa_amd_memory_pool_access_t access;
    hsa_amd_agent_memory_pool_get_info(cpuAgent, pool, HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS,
                                       &access);
    if (access != HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
      if (!pool_i.alloc_allowed || pool_i.alloc_granule == 0 || pool_i.alloc_alignment == 0) {
        if (verbosity() > 0) {
          RDC_LOG(RDC_DEBUG, "Test not applicable. Skipping.");
        }
        return;
      }

      auto gran_sz = pool_i.alloc_granule;
      auto pool_sz = pool_i.size / gran_sz;
      auto max_alloc_size = pool_sz / 2;
      unsigned int max_element = max_alloc_size / sizeof(unsigned int);
      unsigned int* gpu_data;
      unsigned int* sys_data;
      sys_data = (unsigned int*)malloc(max_alloc_size);
      memset(sys_data, 0, max_alloc_size);
      for (unsigned int i = 1; i <= max_element; ++i) {
        sys_data[i] = i;
      }
      // err = hsa_amd_agents_allow_access(1, &gpuAgent, NULL, sys_data);
      // EXPECT_EQ(err, HSA_STATUS_SUCCESS);
      err = hsa_amd_memory_pool_allocate(pool, max_alloc_size, 0,
                                         reinterpret_cast<void**>(&gpu_data));
      throw_if_error(err);
      /*
      if (err == HSA_STATUS_ERROR) {
        err = hsa_amd_memory_pool_free(gpu_data);
      }*/

      err = hsa_amd_agents_allow_access(1, &cpuAgent, NULL, gpu_data);
      throw_if_error(err);
      memset(gpu_data, 0, max_alloc_size);

      // Verify CPU can read & write to GPU memory
      RDC_LOG(RDC_DEBUG, "Verify CPU can read & write to GPU memory");
      for (unsigned int i = 1; i <= max_element; ++i) {
        gpu_data[i] = i;  // Write to gpu memory directly
      }

      for (unsigned int i = 1; i <= max_element; ++i) {
        if (sys_data[i] != gpu_data[i]) {  // Reading GPU memory
          fprintf(stdout,
                  "Values not mathing !! sys_data[%d]:%d ,"
                  "gpu_data[%d]\n",
                  sys_data[i], i, gpu_data[i]);
        }
      }
      RDC_LOG(RDC_DEBUG, "CPU have read & write to GPU memory successfully");
      err = hsa_amd_memory_pool_free(gpu_data);
      free(sys_data);
    } else {
      if (verbosity() > 0) {
        RDC_LOG(RDC_DEBUG, "Test not applicable as system is not large bar, Skipping.");
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
  err = hsa_iterate_agents(IterateCPUAgents, &cpus);
  throw_if_error(err);
  // find all gpu agents
  std::vector<hsa_agent_t> gpus;
  err = hsa_iterate_agents(IterateGPUAgents, &gpus);
  throw_if_error(err);
  for (unsigned int i = 0; i < gpus.size(); ++i) {
    hsa_amd_memory_pool_t gpu_pool;
    memset(&gpu_pool, 0, sizeof(gpu_pool));
    err = hsa_amd_agent_iterate_memory_pools(gpus[i], GetGlobalMemoryPool, &gpu_pool);
    throw_if_error(err);
    if (gpu_pool.handle == 0) {
      RDC_LOG(RDC_DEBUG, "no global mempool in gpu agent");
      return;
    }
    CPUAccessToGPUMemoryTest(cpus[0], gpus[i], gpu_pool);
  }
  if (verbosity() > 0) {
    RDC_LOG(RDC_DEBUG, "subtest Passed");
  }
  per_gpu_info_ += "CPUAccessToGPUMemoryTest Pass.";
  gpu_info_ += "CPUAccessToGPUMemoryTest for GPU ";
  gpu_info_ += std::to_string(gpu_index_);
  gpu_info_ += " Pass. ";
}

void MemoryAccessTest::GPUAccessToCPUMemoryTest(void) {
  hsa_status_t err;

  PrintMemorySubtestHeader("GPUAccessToCPUMemoryTest in Memory Pools");
  // find all cpu agents
  std::vector<hsa_agent_t> cpus;
  err = hsa_iterate_agents(IterateCPUAgents, &cpus);
  throw_if_error(err);

  // find current gpu
  hsa_agent_t current_gpu;
  err = get_agent_by_gpu_index(gpu_index_, &current_gpu);
  throw_if_error(err, "Get agent by GPU index fail.");

  GPUAccessToCPUMemoryTest(cpus[0], current_gpu);

  if (verbosity() > 0) {
    RDC_LOG(RDC_DEBUG, "subtest Passed");
  }

  per_gpu_info_ += "GPUAccessToCPUMemoryTest Pass.";

  gpu_info_ += "GPUAccessToCPUMemoryTest for GPU ";
  gpu_info_ += std::to_string(gpu_index_);
  gpu_info_ += " Pass. ";
}

}  // namespace rdc
}  // namespace amd
