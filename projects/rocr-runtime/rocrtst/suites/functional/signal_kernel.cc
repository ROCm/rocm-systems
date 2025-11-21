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
 *         AMD Research and AMD ROC Software Development
 *
 *         Advanced Micro Devices, Inc.
 *
 *         www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *  this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *  notice, this list of conditions and the following disclaimers in
 *  the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *  nor the names of its contributors may be used to endorse or promote
 *  products derived from this Software without specific prior written
 *  permission.
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

#include <inttypes.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <vector>
#include "suites/functional/signal_kernel.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "common/hsatimer.h"
#include "common/concurrent_utils.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"

static unsigned int NumOfKernels = 1;

#define ASSERT_MSG(C, err) { \
  if (C == 1) { \
  std::cout << err << std::endl; \
  } \
}

SignalKernelTest::SignalKernelTest(SignalKernelType type_) : TestBase() {
  set_num_iteration(10);  // Number of iterations to execute of the main test;
            // This is a default value which can be overridden
            // on the command line.
  if (type_ == SET) {
  set_title("RocR Signal Kernel Set Test");
  set_description("This test verifies that the signal is set from kernel");
  } else if (type_ == WAIT) {
  set_title("RocR Signal Wait Test");
  set_description("This test verifies that the signal is re-set from system side");
  } else if (type_ == MULTISET) {
  set_title("RocR Signal Kernel Multi Set Test");
  set_description("This test verifies that the signal is set on multiple work-items");
  } else if (type_ ==  MULTIWAIT) {
  set_title("RocR Signal Kernel Multi Set Test");
  set_description("This tset verifies that re-set signal from system side, multiple work-items");
  }
}

SignalKernelTest::~SignalKernelTest(void) {
}

void SignalKernelTest::SetUp(void) {
  hsa_status_t err;

  TestBase::SetUp();

  err = rocrtst::SetDefaultAgents(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  err = rocrtst::SetPoolsTypical(this);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  return;
}


void SignalKernelTest::Run(void) {
// Compare required profile for this test case with what we're actually
// running on
  if (!rocrtst::CheckProfile(this)) {
  return;
  }

  TestBase::Run();
}

void SignalKernelTest::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void SignalKernelTest::DisplayResults(void) const {
  return;
}

void SignalKernelTest::Close() {
  // This will close handles opened within rocrtst utility calls and call
  // hsa_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}



// The kernarg data structure
typedef struct __attribute__ ((aligned(16))) signal_args_s {
  void *signal_values;
} signal_args_t;
signal_args_t signal_args;

void SignalKernelTest::KernelSetFunction(SignalKernelType type_) {
  hsa_status_t status;

  // Get the GPU agents into a vector
  std::vector<hsa_agent_t> agent_list;
  status = hsa_iterate_agents(rocrtst::IterateGPUAgents, &agent_list);
  ASSERT_EQ(status, HSA_STATUS_SUCCESS);

  // Get CPU agent to get the kern_arg pool
  std::vector<hsa_agent_t> cpu_agent;
  status = hsa_iterate_agents(rocrtst::IterateCPUAgents, &cpu_agent);
  ASSERT_EQ(status, HSA_STATUS_SUCCESS);

  // Repeat the test for each agent
  unsigned int ii;
  for (ii = 0; ii < agent_list.size(); ++ii) {
    // Check if the queue supports dispatch
    uint32_t features = 0;
    status = hsa_agent_get_info(agent_list[ii], HSA_AGENT_INFO_FEATURE, &features);
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);
    if (0 == (features & HSA_AGENT_FEATURE_KERNEL_DISPATCH)) {
      continue;
    }

    // Find a memory pool that supports fine grained memory
    hsa_amd_memory_pool_t global_pool;
    global_pool.handle = (uint64_t)-1;
    status = hsa_amd_agent_iterate_memory_pools(agent_list[ii], rocrtst::GetGlobalMemoryPool, &global_pool);
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);

    // Obtain the agent's machine model
    hsa_machine_model_t machine_model;
    status = hsa_agent_get_info(agent_list[ii], HSA_AGENT_INFO_MACHINE_MODEL, &machine_model);
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);

    // Find a memory pool that supports kernel arguments
    hsa_amd_memory_pool_t kernarg_pool;
    kernarg_pool.handle = (uint64_t)-1;
    status = hsa_amd_agent_iterate_memory_pools(cpu_agent[0], rocrtst::GetKernArgMemoryPool, &kernarg_pool);
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);
    // Create a queue
    hsa_queue_t* queue;
    status = hsa_queue_create(agent_list[ii], 1024, HSA_QUEUE_TYPE_SINGLE, NULL, NULL, UINT32_MAX, UINT32_MAX, &queue);
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);

    set_kernel_file_name("signal_operations_kernels.hsaco");
    if (type_ == SET) {
      set_kernel_name("signal_st_rlx_kernel");
    } else if (type_ == MULTISET) {
      set_kernel_name("signal_st_rlx_kernel_multi");
      NumOfKernels = 16;
    } else if (type_ == WAIT) {
      set_kernel_name("signal_wait_kernel");
    } else if (type_ == MULTIWAIT) {
      set_kernel_name("signal_wait_kernel_multi");
      NumOfKernels = 16;
    }

    status = rocrtst::LoadKernelFromObjFile(this, &agent_list[ii]);
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);


    // Allocate the kernel argument buffer from the correct pool
    signal_args_t* kernarg_buffer = NULL;
    status = hsa_amd_memory_pool_allocate(kernarg_pool,
                 sizeof(signal_args_t), 0,
                   reinterpret_cast<void**>(&kernarg_buffer));
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);

    status = hsa_amd_agents_allow_access(1, &agent_list[ii], NULL, kernarg_buffer);
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);
    // Create the completion signal
    hsa_signal_t completion_signal;
    status = hsa_signal_create(1, 0, NULL, &completion_signal);
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);

    hsa_amd_memory_pool_access_t access;
    status = hsa_amd_agent_memory_pool_get_info(cpu_agent[0],
                                              global_pool,
                                              HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS,
                                              &access);
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);

    hsa_signal_t* kernel_signal;

    if (access != HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
     // Create the kernel signal
      status = hsa_amd_memory_pool_allocate(global_pool,
                                          NumOfKernels*sizeof(hsa_signal_t), 0,
                                          reinterpret_cast<void**>(&kernel_signal));
      ASSERT_EQ(status, HSA_STATUS_SUCCESS);
      status = hsa_amd_agents_allow_access(1, &cpu_agent[0], NULL, kernel_signal);
      ASSERT_EQ(status, HSA_STATUS_SUCCESS);
      for (unsigned int k = 0; k < NumOfKernels; ++k) {
        status = hsa_signal_create(1, 0, NULL, kernel_signal);
        ASSERT_EQ(status, HSA_STATUS_SUCCESS);
      }
      // Set the signal_args with kernel_signal, will be accessed from Kernel side
      signal_args.signal_values = reinterpret_cast<void*>(kernel_signal);
    }

    memcpy(kernarg_buffer, &signal_args, sizeof(signal_args_t));

    // Setup the dispatch packet
    hsa_kernel_dispatch_packet_t dispatch_packet;
    memset(&dispatch_packet, 0, sizeof(hsa_kernel_dispatch_packet_t));
    dispatch_packet.workgroup_size_x = NumOfKernels;
    dispatch_packet.workgroup_size_y = 1;
    dispatch_packet.workgroup_size_z = 1;
    dispatch_packet.grid_size_x = NumOfKernels;
    dispatch_packet.grid_size_y = 1;
    dispatch_packet.grid_size_z = 1;
    dispatch_packet.kernel_object = kernel_object();
    dispatch_packet.group_segment_size = group_segment_size();
    dispatch_packet.private_segment_size = private_segment_size();
    dispatch_packet.kernarg_address = kernarg_buffer;
    dispatch_packet.completion_signal = completion_signal;

    // const uint32_t queue_size = queue->size;
    const uint32_t queue_mask = queue->size - 1;

    // write to command queue
    uint64_t index = hsa_queue_load_write_index_relaxed(queue);
    hsa_queue_store_write_index_relaxed(queue, index + 1);

    rocrtst::WriteAQLToQueueLoc(queue, index, &dispatch_packet);


    dispatch_packet.header |= HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
    dispatch_packet.header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
    dispatch_packet.header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
    dispatch_packet.header |= 1 << HSA_PACKET_HEADER_BARRIER;
    dispatch_packet.setup |= 1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;

    void* q_base = queue->base_address;
    // Set the Aql packet header
    rocrtst::AtomicSetPacketHeader(dispatch_packet.header, dispatch_packet.setup,
                        &(reinterpret_cast<hsa_kernel_dispatch_packet_t*>
                            (q_base))[index & queue_mask]);

    // ringdoor bell
    hsa_signal_store_relaxed(queue->doorbell_signal, index);

    if (type_ == WAIT) {
      for (unsigned int k = 0; k < NumOfKernels; ++k) {
        // setting the kernel_signal to 0 from system side.
        kernel_signal[k].handle = 0;
      }
    }

    // Wait on the completion signal
    hsa_signal_wait_relaxed(completion_signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

    // Check kernel signal
    for (unsigned int k = 0; k < NumOfKernels; ++k) {
      ASSERT_EQ(0, (int)(kernel_signal[k].handle));
    }

    status = hsa_signal_destroy(completion_signal);
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);


    if (access != HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
      status = hsa_amd_memory_pool_free(kernel_signal);
      ASSERT_EQ(status, HSA_STATUS_SUCCESS);
    }

    status = hsa_amd_memory_pool_free(kernarg_buffer);
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);

    // Destroy the queue
    status = hsa_queue_destroy(queue);
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);
  }
}
void SignalKernelTest::TestSignalKernelSet(void) {
  KernelSetFunction(SET);
}


void SignalKernelTest::TestSignalKernelMultiSet(void) {
  KernelSetFunction(MULTISET);
}


void SignalKernelTest::TestSignalKernelWait(void) {
  KernelSetFunction(WAIT);
}


void SignalKernelTest::TestSignalKernelMultiWait(void) {
  hsa_status_t status;

  // Get the GPU agents into a vector
  std::vector<hsa_agent_t> agent_list;
  status = hsa_iterate_agents(rocrtst::IterateGPUAgents, &agent_list);
  ASSERT_EQ(status, HSA_STATUS_SUCCESS);


  // Get CPU agent to get the kern_arg pool
  std::vector<hsa_agent_t> cpu_agent;
  status = hsa_iterate_agents(rocrtst::IterateCPUAgents, &cpu_agent);
  ASSERT_EQ(status, HSA_STATUS_SUCCESS);

  // Repeat the test for each agent
  unsigned int ii;
  for (ii = 0; ii < agent_list.size(); ++ii) {
    // Check if the queue supports dispatch
    uint32_t features = 0;
    status = hsa_agent_get_info(agent_list[ii], HSA_AGENT_INFO_FEATURE, &features);
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);
    if (0 == (features & HSA_AGENT_FEATURE_KERNEL_DISPATCH)) {
      continue;
    }

    // Find a memory pool that supports fine grained memory
    hsa_amd_memory_pool_t global_pool;
    global_pool.handle = (uint64_t)-1;
    status = hsa_amd_agent_iterate_memory_pools(agent_list[ii], rocrtst::GetGlobalMemoryPool, &global_pool);
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);

    // Obtain the agent's machine model
    hsa_machine_model_t machine_model;
    status = hsa_agent_get_info(agent_list[ii], HSA_AGENT_INFO_MACHINE_MODEL, &machine_model);
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);


    // Find a memory pool that supports kernel arguments
    hsa_amd_memory_pool_t kernarg_pool;
    kernarg_pool.handle = (uint64_t)-1;
    status = hsa_amd_agent_iterate_memory_pools(cpu_agent[0], rocrtst::GetKernArgMemoryPool, &kernarg_pool);
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);

    // Create a queue
    hsa_queue_t* queue;
    status = hsa_queue_create(agent_list[ii], 1024, HSA_QUEUE_TYPE_SINGLE, NULL, NULL, UINT32_MAX, UINT32_MAX, &queue);
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);


    set_kernel_file_name("signal_operations_kernels.hsaco");
    set_kernel_name("signal_wait_kernel_multi");
    status = rocrtst::LoadKernelFromObjFile(this, &agent_list[ii]);
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);

    // Allocate the kernel argument buffer from the correct pool
    signal_args_t* kernarg_buffer = NULL;
    status = hsa_amd_memory_pool_allocate(kernarg_pool,
           sizeof(signal_args_t), 0,
           reinterpret_cast<void**>(&kernarg_buffer));
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);

    status = hsa_amd_agents_allow_access(1, &agent_list[ii], NULL, kernarg_buffer);
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);
    // Create the completion signal
    hsa_signal_t completion_signal;
    status = hsa_signal_create(1, 0, NULL, &completion_signal);
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);

    hsa_amd_memory_pool_access_t access;
    status = hsa_amd_agent_memory_pool_get_info(cpu_agent[0],
                                              global_pool,
                                              HSA_AMD_AGENT_MEMORY_POOL_INFO_ACCESS,
                                              &access);
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);

    hsa_signal_t* kernel_signal;

    if (access != HSA_AMD_MEMORY_POOL_ACCESS_NEVER_ALLOWED) {
      // Create the kernel signal
      status = hsa_amd_memory_pool_allocate(global_pool,
                                          NumOfKernels*sizeof(hsa_signal_t), 0,
                                          reinterpret_cast<void**>(&kernel_signal));
      ASSERT_EQ(status, HSA_STATUS_SUCCESS);
      status = hsa_amd_agents_allow_access(1, &cpu_agent[0], NULL, kernel_signal);
      ASSERT_EQ(status, HSA_STATUS_SUCCESS);

      for (unsigned int k = 0; k < NumOfKernels; ++k) {
       status = hsa_signal_create(1, 0, NULL, kernel_signal + k);
       ASSERT_EQ(status, HSA_STATUS_SUCCESS);
      }

      // Set the signal_args with kernel_signal, will be accessed from Kernel side
      signal_args.signal_values = reinterpret_cast<void*>(kernel_signal);
    }

    memcpy(kernarg_buffer, &signal_args, sizeof(signal_args_t));

    // Setup the dispatch packet
    hsa_kernel_dispatch_packet_t dispatch_packet;
    memset(&dispatch_packet, 0, sizeof(hsa_kernel_dispatch_packet_t));

    dispatch_packet.workgroup_size_x = NumOfKernels;
    dispatch_packet.workgroup_size_y = 1;
    dispatch_packet.workgroup_size_z = 1;
    dispatch_packet.grid_size_x = NumOfKernels;
    dispatch_packet.grid_size_y = 1;
    dispatch_packet.grid_size_z = 1;
    dispatch_packet.kernel_object = kernel_object();
    dispatch_packet.group_segment_size = group_segment_size();
    dispatch_packet.private_segment_size = private_segment_size();
    dispatch_packet.kernarg_address = kernarg_buffer;
    dispatch_packet.completion_signal = completion_signal;

    // const uint32_t queue_size = queue->size;
    const uint32_t queue_mask = queue->size - 1;
    // write to command queue
    uint64_t index = hsa_queue_load_write_index_relaxed(queue);
    hsa_queue_store_write_index_relaxed(queue, index + 1);

    rocrtst::WriteAQLToQueueLoc(queue, index, &dispatch_packet);


    dispatch_packet.header |= HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE;
    dispatch_packet.header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
    dispatch_packet.header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;
    dispatch_packet.header |= 1 << HSA_PACKET_HEADER_BARRIER;
    dispatch_packet.setup |= 1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;

    void* q_base = queue->base_address;
    // Set the Aql packet header
    rocrtst::AtomicSetPacketHeader(dispatch_packet.header, dispatch_packet.setup,
                        &(reinterpret_cast<hsa_kernel_dispatch_packet_t*>
                            (q_base))[index & queue_mask]);


    // ringdoor bell
    hsa_signal_store_relaxed(queue->doorbell_signal, index);

    // setting the kernel_signal to 0 from system side.
    for (unsigned int k = 0; k < NumOfKernels; ++k) {
      kernel_signal[k].handle = 0;
    }
    // Wait on the completion signal
    hsa_signal_wait_relaxed(completion_signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

    // Check kernel signal
    ASSERT_EQ(0, (int)kernel_signal->handle);

    // destroy the signal created
    status = hsa_signal_destroy(completion_signal);
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);

    status = hsa_amd_memory_pool_free(kernarg_buffer);
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);

    // Destroy the queue
    status = hsa_queue_destroy(queue);
    ASSERT_EQ(status, HSA_STATUS_SUCCESS);
  }
}

