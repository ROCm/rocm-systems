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
#include <memory>
#include <string>

#include "suites/functional/memory_atomics.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "common/hsatimer.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"

static const uint32_t kNumBufferElements = 256;
static const int kValue = 5;

MemoryAtomic::MemoryAtomic(AtomicTest testtype) :
    TestBase() {
  set_num_iteration(10);  // Number of iterations to execute of the main test;
                          // This is a default value which can be overridden
                          // on the command line.
  testtype_ = testtype;
  std::string name;
  std::string desc;

  name = "RocR Memory Atomic Test";
  desc = "";

  if (testtype_ == ADD) {
    name += " For ADD";
    desc += " This test will do Add kernel atomic"
            " operation on GPU and system memory.";
  } else if (testtype_ == SUB) {
    name += " For Sub";
    desc += " This test will do Sub kernel atomic"
            " operation on GPU and system memory.";
  } else if (testtype_ == AND) {
    name += " For And";
    desc += " This test will do AND kernel atomic"
            " operation on GPU and system memory.";
  } else if (testtype_ == OR) {
    name += " For Or";
    desc += " This test will do OR kernel atomic"
            " operation on GPU and system memory.";
  } else if (testtype_ == XOR) {
    name += " For Xor";
    desc += " This test will do XOR kernel atomic"
            " operation on GPU and system memory.";
  } else if (testtype_ == MIN) {
    name += " For Minimum";
    desc += " This test will do Minimum kernel atomic"
            " operation on GPU and system memory.";
  } else if (testtype_ == MAX) {
    name += " For Maximum";
    desc += " This test will do Maximum kernel atomic"
            " operation on GPU and system memory.";
  } else if (testtype_ == XCHG) {
    name += " For Exchange";
    desc += " This test will do Xchg kernel atomic"
            " operation on GPU and system memory.";
  } else if (testtype_ == INC) {
    name += " For Increment";
    desc += " This test will do Increment kernel atomic"
            " operation on GPU and system memory.";
  } else if (testtype_ == DEC) {
    name += " For Decremnet";
    desc += " This test will do decrement kernel atomic"
            " operation on GPU and system memory.";
  }

  set_title(name);
  set_description(desc);
  memset(&aql(), 0, sizeof(hsa_kernel_dispatch_packet_t));
}

MemoryAtomic::~MemoryAtomic(void) {
}

// Any 1-time setup involving member variables used in the rest of the test
// should be done here.
void MemoryAtomic::SetUp(void) {
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

void MemoryAtomic::Run(void) {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  TestBase::Run();
}

void MemoryAtomic::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void MemoryAtomic::DisplayResults(void) const {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  return;
}

void MemoryAtomic::Close() {
  // This will close handles opened within rocrtst utility calls and call
  // hsa_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

typedef struct  __attribute__ ((aligned(16)))  args_t {
  int *a;
  int *b;
  int *c;
  int d;
  int n;
  } args;

static const char kSubTestSeparator[] = "  **************************";


static const int kMemoryAllocSize = 4096;

void MemoryAtomic::MemoryAtomicTest(hsa_agent_t cpuAgent,
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

  // Find a memory pool that supports kernel arguments.
  hsa_amd_memory_pool_t kernarg_pool;
  err = hsa_amd_agent_iterate_memory_pools(cpuAgent,
                                            rocrtst::GetKernArgMemoryPool,
                                            &kernarg_pool);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  // Allocate the host side buffers
  // (refSysdata,oldValues,oldrefdata,kernArg) on system memory

  // this is ref sys data on which atomics operation need to done
  int *refSysdata = NULL;
  // This is oldrefdata which will be required  to compare the returned old values after atomics operation
  int *oldrefdata = NULL;
  // This is returned old values
  int *oldValues = NULL;
  // This is expected data set
  int *expecteddata = NULL;
  // Array size for the data
  int arraySize = kMemoryAllocSize/sizeof(int);

  // Get System Memory Pool on the cpuAgent to allocate host side buffers
  hsa_amd_memory_pool_t global_pool;
  err = hsa_amd_agent_iterate_memory_pools(cpuAgent,
                                            rocrtst::GetGlobalMemoryPool,
                                            &global_pool);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  err = hsa_amd_memory_pool_allocate(global_pool,
                                    kMemoryAllocSize, 0,
                                    reinterpret_cast<void **>(&oldValues));
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  err = hsa_amd_memory_pool_allocate(global_pool,
                                    kMemoryAllocSize, 0,
                                    reinterpret_cast<void **>(&refSysdata));
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  err = hsa_amd_memory_pool_allocate(global_pool,
                                    kMemoryAllocSize, 0,
                                    reinterpret_cast<void **>(&oldrefdata));
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  err = hsa_amd_memory_pool_allocate(global_pool,
                                    kMemoryAllocSize, 0,
                                    reinterpret_cast<void **>(&expecteddata));
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);


  // Allocate the kernel argument buffer from the kernarg_pool.
  args *kernArguments = NULL;
  err = hsa_amd_memory_pool_allocate(kernarg_pool, sizeof(args_t), 0,
                                     reinterpret_cast<void **>(&kernArguments));
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);


  memset(oldValues, 0, kMemoryAllocSize);
  memset(expecteddata, 0, kMemoryAllocSize);
  // this signal will be used for copying the data memory from To and fro from GPU
  // on Non-largebar system
  hsa_signal_t copy_signal;

  // for the dGPU, we have coarse grained local memory,
  // so allocate memory for it on the GPU's GLOBAL segment .

  // Get local memory of GPU to allocate device side buffers on which atomics operation need to done
  int *gpuRefData = NULL;

  // On non-Large bar system acess to GPU pool not allowed to directly so pinned memory
  // g_gpuRefData is pointer to GPU Memory allocated on non-large bar where
  // gpuRefData would be pointer to  host allocated memory on non-large bar
  int *g_gpuRefData = NULL;
  //  Pointer to the location where to store the new address
  int *device_ptr = NULL;

  if (access != HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
    err = hsa_amd_memory_pool_allocate(gpu_pool, kMemoryAllocSize, 0,
                                       reinterpret_cast<void **>(&gpuRefData));
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    // Allow cpuAgent access to all allocated GPU memory.
    err = hsa_amd_agents_allow_access(1, &cpuAgent, NULL, gpuRefData);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
    memset(gpuRefData, 0, kMemoryAllocSize);
  } else {
    err = hsa_signal_create(1, 0, NULL, &copy_signal);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
    // Alocate the System Memory and get pointer gpuRefData
    err = hsa_amd_memory_pool_allocate(global_pool, kMemoryAllocSize, 0,
                                        reinterpret_cast<void **>(&gpuRefData));
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
    memset(gpuRefData, 0, kMemoryAllocSize);
    // Alocate the GPU Memory and get pointer g_gpuRefData
    err = hsa_amd_memory_pool_allocate(gpu_pool, kMemoryAllocSize, 0,
                                        reinterpret_cast<void **>(&g_gpuRefData));
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
    // Map the Host memory and get the pointer to new adress which is accesible to GPU agent
    err = hsa_amd_agents_allow_access(1, &gpuAgent, NULL, gpuRefData);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
    device_ptr = gpuRefData;
  }



  // initialize the host buffers & gpuRefData buffer
  for (int i = 0; i < arraySize; ++i) {
    unsigned int seed = time(NULL);
    refSysdata[i] = 6 + rand_r(&seed) % 1;
    gpuRefData[i] = 6 + rand_r(&seed) % 1;
    oldrefdata[i] = refSysdata[i];
  }

  // Sync the data from system memory to GPU memory on non-largebar
  if (access == HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
    hsa_signal_store_relaxed(copy_signal, 1);
    err = hsa_amd_memory_async_copy(g_gpuRefData, gpuAgent, device_ptr,
                                    gpuAgent, kMemoryAllocSize, 0, NULL, copy_signal);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
    while (hsa_signal_wait_acquire(copy_signal, HSA_SIGNAL_CONDITION_LT, 1, (uint64_t)(-1), HSA_WAIT_STATE_ACTIVE)) {}
  }


  // Allow gpuAgent access to all allocated system memory.
  err = hsa_amd_agents_allow_access(1, &gpuAgent, NULL, oldValues);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  err = hsa_amd_agents_allow_access(1, &gpuAgent, NULL, refSysdata);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  err = hsa_amd_agents_allow_access(1, &gpuAgent, NULL, oldrefdata);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  err = hsa_amd_agents_allow_access(1, &gpuAgent, NULL, kernArguments);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  kernArguments->a = refSysdata;
  if (access != HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
    kernArguments->b = gpuRefData;
  } else {
    kernArguments->b = g_gpuRefData;
  }
  kernArguments->c = oldValues;

  if (testtype_ != INC && testtype_ != DEC) {
    kernArguments->d = kValue;
  }

  // Create the executable, get symbol by name and load the code object
  set_kernel_file_name("atomicOperations_kernels.hsaco");

  if (testtype_ == ADD) {
    set_kernel_name("test_atomic_add");
    // set the expected data result set from kernel
    for (int i = 0; i < arraySize; ++i) {
      expecteddata[i] = oldrefdata[i] + kValue;
    }
  } else if (testtype_ == SUB) {
    set_kernel_name("test_atomic_sub");
    // set the expected data result set from kernel
    for (int i = 0; i < arraySize; ++i) {
      expecteddata[i] = oldrefdata[i] - kValue;
    }
  } else if (testtype_ == AND) {
    set_kernel_name("test_atomic_and");
    // set the expected data result set from kernel
    for (int i = 0; i < arraySize; ++i) {
      expecteddata[i] = oldrefdata[i] & kValue;
    }
  } else if (testtype_ == OR) {
    set_kernel_name("test_atomic_or");
    // set the expected data result set from kernel
    for (int i = 0; i < arraySize; ++i) {
      expecteddata[i] = oldrefdata[i] | kValue;
    }
  } else if (testtype_ == XOR) {
    set_kernel_name("test_atomic_xor");
    // set the expected data result set from kernel
    for (int i = 0; i < arraySize; ++i) {
      expecteddata[i] = oldrefdata[i] ^ kValue;
    }
  } else if (testtype_ == MIN) {
    set_kernel_name("test_atomic_min");
    // set the expected data result set from kernel
    for (int i = 0; i < arraySize; ++i) {
      expecteddata[i] = std::min(oldrefdata[i], kValue);
    }
  } else if (testtype_ == MAX) {
    set_kernel_name("test_atomic_max");
    // set the expected data result set from kernel
    for (int i = 0; i < arraySize; ++i) {
      expecteddata[i] = std::max(oldrefdata[i], kValue);
    }
  } else if (testtype_ == INC) {
    set_kernel_name("test_atomic_inc");
    // set the expected data result set from kernel
    for (int i = 0; i < arraySize; ++i) {
      expecteddata[i] = oldrefdata[i] + 4;
    }
  } else if (testtype_ == DEC) {
    set_kernel_name("test_atomic_dec");
    // set the expected data result set from kernel
    for (int i = 0; i < arraySize; ++i) {
      expecteddata[i] = oldrefdata[i] - 4;
    }
  } else if (testtype_ == XCHG) {
    set_kernel_name("test_atomic_xchg");
    // set the expected data result set from kernel
    for (int i = 0; i < arraySize; ++i) {
      expecteddata[i] = kValue;
    }
  } else {
    if (verbosity() > 0) {
      std::cout<< "No test specified" <<std::endl;
    }
  }

  err = rocrtst::LoadKernelFromObjFile(this, &gpuAgent);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  // Fill up the kernel packet except header
  err = rocrtst::InitializeAQLPacket(this, &aql());
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  aql().workgroup_size_x = 256;
  aql().workgroup_size_y = 1;
  aql().workgroup_size_z = 1;
  aql().grid_size_x = arraySize;
  aql().kernarg_address = kernArguments;
  aql().kernel_object = kernel_object();

  const uint32_t queue_mask = queue->size - 1;

  // Load index for writing header later to command queue at same index
  uint64_t index = hsa_queue_load_write_index_relaxed(queue);
  hsa_queue_store_write_index_relaxed(queue, index + 1);

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

  // Sync the data from GPU memory to system memory on non-largebar
  if (access == HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
    hsa_signal_store_relaxed(copy_signal, 1);
    err = hsa_amd_memory_async_copy(device_ptr, gpuAgent, g_gpuRefData,
                                    gpuAgent, kMemoryAllocSize, 0, NULL, copy_signal);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
    while (hsa_signal_wait_acquire(copy_signal, HSA_SIGNAL_CONDITION_LT, 1, (uint64_t)(-1), HSA_WAIT_STATE_ACTIVE)) { }
  }

  // compare results with expected results
  for (int i = 0; i < arraySize; ++i) {
    ASSERT_EQ(refSysdata[i], expecteddata[i]);
    ASSERT_EQ(gpuRefData[i], expecteddata[i]);
    ASSERT_EQ(oldValues[i], oldrefdata[i]);
  }

  if (refSysdata) {
    err = hsa_memory_free(refSysdata);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  }
  if (oldrefdata) {
    err = hsa_memory_free(oldrefdata);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  }
  if (oldValues) {
    err = hsa_memory_free(oldValues);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  }
  if (access == HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
    err = hsa_amd_memory_unlock(gpuRefData);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
    // Destroy the copy signal
    err = hsa_signal_destroy(copy_signal);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
    if (g_gpuRefData) {
      err = hsa_memory_free(g_gpuRefData);
      ASSERT_EQ(err, HSA_STATUS_SUCCESS);
    }
  }
  if (gpuRefData) {
    err = hsa_memory_free(gpuRefData);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  }
  if (kernArguments) {
    err = hsa_memory_free(kernArguments);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  }
  if (queue) {
    err = hsa_queue_destroy(queue);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  }
}

void MemoryAtomic::MemoryAtomicTest(void) {
  hsa_status_t err;
  // find all cpu agents
  std::vector<hsa_agent_t> cpus;
  err = hsa_iterate_agents(rocrtst::IterateCPUAgents, &cpus);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  // find all gpu agents
  std::vector<hsa_agent_t> gpus;
  err = hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  for (unsigned int i = 0 ; i< gpus.size(); ++i) {
    MemoryAtomicTest(cpus[0], gpus[i]);
  }
}

