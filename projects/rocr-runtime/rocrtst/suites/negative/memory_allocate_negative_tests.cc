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

#include "suites/negative/memory_allocate_negative_tests.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "common/hsatimer.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"

static const uint32_t kNumBufferElements = 256;





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



MemoryAllocateNegativeTest::MemoryAllocateNegativeTest(void) :
    TestBase() {
  set_num_iteration(10);  // Number of iterations to execute of the main test;
                          // This is a default value which can be overridden
                          // on the command line.

  set_title("RocR Memory Allocate Negative Test");
  set_description("This series of tests are Negative tests "
    "that do check memory allocation on GPU and CPU, "
    "i.e. requesting an allocation of more than max "
    "pool size or 0 size.");
}

MemoryAllocateNegativeTest::~MemoryAllocateNegativeTest(void) {
}

// Any 1-time setup involving member variables used in the rest of the test
// should be done here.
void MemoryAllocateNegativeTest::SetUp(void) {
  hsa_status_t err;

  TestBase::SetUp();

  err = rocrtst::SetDefaultAgents(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  err = rocrtst::SetPoolsTypical(this);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  return;
}

void MemoryAllocateNegativeTest::Run(void) {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  TestBase::Run();
}

void MemoryAllocateNegativeTest::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void MemoryAllocateNegativeTest::DisplayResults(void) const {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  return;
}

void MemoryAllocateNegativeTest::Close() {
  // This will close handles opened within rocrtst utility calls and call
  // hsa_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}




static const char kSubTestSeparator[] = "  **************************";

static void PrintMemorySubtestHeader(const char *header) {
  std::cout << "  *** Memory Subtest: " << header << " ***" << std::endl;
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

static const int kMemoryAllocSize = 1024;

// This test verify that hsa_memory_allocate can't allocate
// memory more than HSA_AMD_MEMORY_POOL_INFO_ALLOC_MAX_SIZE
void MemoryAllocateNegativeTest::MaxMemoryAllocateTest(hsa_agent_t agent,
                                               hsa_amd_memory_pool_t pool) {
  hsa_status_t err;

  rocrtst::pool_info_t pool_i;
  err = rocrtst::AcquirePoolInfo(pool, &pool_i);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  if (verbosity() > 0) {
    PrintAgentNameAndType(agent);
  }

  // Determine if allocation is allowed in this pool
  if (!pool_i.alloc_allowed || pool_i.alloc_granule == 0) {
    if (verbosity() > 0) {
      std::cout << "  Test not applicable. Skipping." << std::endl;
      std::cout << kSubTestSeparator << std::endl;
    }
    return;
  }

    char *memoryPtr;
  auto gran_sz = pool_i.alloc_granule;
  size_t max_size = pool_i.aggregate_alloc_max;
  err = hsa_amd_memory_pool_allocate(pool, (max_size + gran_sz), 0,
                                       reinterpret_cast<void**>(&memoryPtr));
    ASSERT_EQ(err, HSA_STATUS_ERROR_INVALID_ALLOCATION);
  return;
}




// This test verify that requesting an allocation
// of 0 size is valid on memory pool or not
void MemoryAllocateNegativeTest::ZeroMemoryAllocateTest(hsa_agent_t agent,
                                                hsa_amd_memory_pool_t pool) {
  hsa_status_t err;

  rocrtst::pool_info_t pool_i;
  err = rocrtst::AcquirePoolInfo(pool, &pool_i);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  if (verbosity() > 0) {
    PrintAgentNameAndType(agent);
  }

  // Determine if allocation is allowed in this pool
  bool alloc = false;
  err = hsa_amd_memory_pool_get_info(pool,
                   HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED, &alloc);

  if (alloc) {
    char *memoryPtr;
    err = hsa_amd_memory_pool_allocate(pool, 0, 0,
                                       reinterpret_cast<void**>(&memoryPtr));
    ASSERT_EQ(err, HSA_STATUS_ERROR_INVALID_ARGUMENT);
  }
  return;
}


void MemoryAllocateNegativeTest::MaxMemoryAllocateTest(void) {
  hsa_status_t err;
  std::vector<std::shared_ptr<rocrtst::agent_pools_t>> agent_pools;

  PrintMemorySubtestHeader("MaxMemoryAllocateTest in Memory Pools");

  err = rocrtst::GetAgentPools(&agent_pools);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  auto pool_idx = 0;
  for (auto a : agent_pools) {
    for (auto p : a->pools) {
      std::cout << "  Pool " << pool_idx++ << ":" << std::endl;
      MaxMemoryAllocateTest(a->agent, p);
    }
  }

  if (verbosity() > 0) {
    std::cout << "subtest Passed" << std::endl;
    std::cout << kSubTestSeparator << std::endl;
  }
}

void MemoryAllocateNegativeTest::ZeroMemoryAllocateTest(void) {
  hsa_status_t err;
  std::vector<std::shared_ptr<rocrtst::agent_pools_t>> agent_pools;

  PrintMemorySubtestHeader("ZeroMemoryAllocateTest in Memory Pools");

  err = rocrtst::GetAgentPools(&agent_pools);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  auto pool_idx = 0;
  for (auto a : agent_pools) {
    for (auto p : a->pools) {
      std::cout << "  Pool " << pool_idx++ << ":" << std::endl;
      ZeroMemoryAllocateTest(a->agent, p);
    }
  }

  if (verbosity() > 0) {
    std::cout << "subtest Passed" << std::endl;
    std::cout << kSubTestSeparator << std::endl;
  }
}

static const uint32_t kMaxQueueSizeForAgent = 1024;
static const uint32_t kMaxQueue = 64;

typedef struct test_validation_data_t {
  bool cb_triggered;
  uint64_t expected_address;
} test_validation_data;

hsa_status_t CallbackSystemErrorHandling(const hsa_amd_event_t* event, void* data) {
  test_validation_data* user_data = reinterpret_cast<test_validation_data*>(data);

  if (event->event_type != HSA_AMD_GPU_MEMORY_ERROR_EVENT) {
    std::cout << "ERROR: Invalid error type" << std::endl;
    return HSA_STATUS_SUCCESS;
  }

  const hsa_amd_gpu_memory_error_info_t& error_info =
      reinterpret_cast<const hsa_amd_gpu_memory_error_info_t&>(event->memory_error);

  if (error_info.virtual_address != user_data->expected_address) {
    std::cout << "ERROR: Invalid virtual address" << std::endl;
    return HSA_STATUS_SUCCESS;
  }

  if (!(error_info.error_reason_mask & HSA_AMD_MEMORY_ERROR_MEMORY_IN_USE)) {
    std::cout << "ERROR: HSA_AMD_MEMORY_ERROR_MEMORY_IN_USE flag not set" << std::endl;
    return HSA_STATUS_SUCCESS;
  }

  user_data->cb_triggered = true;

  return HSA_STATUS_SUCCESS;
}


void MemoryAllocateNegativeTest::FreeQueueRingBufferTest(void) {
  hsa_status_t err;

  memset(&aql(), 0, sizeof(hsa_kernel_dispatch_packet_t));
  set_kernel_file_name("dispatch_time_kernels.hsaco");
  set_kernel_name("empty_kernel");

  if (verbosity() > 0) {
    PrintMemorySubtestHeader("RingBufferFree");
  }

  // find all cpu agents
  std::vector<hsa_agent_t> cpus;
  err = hsa_iterate_agents(rocrtst::IterateCPUAgents, &cpus);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  // find all gpu agents
  std::vector<hsa_agent_t> gpus;
  err = hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  for (unsigned int i = 0; i < gpus.size(); ++i) {
    FreeQueueRingBufferTest(gpus[i]);
  }

  if (verbosity() > 0) {
    std::cout << "subtest Passed" << std::endl;
    std::cout << kSubTestSeparator << std::endl;
  }
}

void MemoryAllocateNegativeTest::FreeQueueRingBufferTest(hsa_agent_t gpuAgent) {
  hsa_status_t err;

  auto enqueue_dispatch = [&](hsa_queue_t* queue) {
    hsa_signal_store_relaxed(aql().completion_signal, 1);

    aql().setup |= 1 << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
    aql().workgroup_size_x = 1;
    aql().workgroup_size_y = 1;
    aql().workgroup_size_z = 1;

    aql().kernel_object = kernel_object();

    const uint32_t queue_mask = queue->size - 1;

    // Load index for writing header later to command queue at same index
    uint64_t index = hsa_queue_load_write_index_relaxed(queue);
    hsa_queue_store_write_index_relaxed(queue, index + 1);

    rocrtst::WriteAQLToQueueLoc(queue, index, &aql());
    aql().header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
    aql().header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
    aql().header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;

    // Set the Aql packet header
    rocrtst::AtomicSetPacketHeader(aql().header, aql().setup,
                                   &(reinterpret_cast<hsa_kernel_dispatch_packet_t*>(
                                       queue->base_address))[index & queue_mask]);


    // ringdoor bell
    hsa_signal_store_relaxed(queue->doorbell_signal, index);

    // wait for the signal long enough for the queue error handling callback to happen
    hsa_signal_value_t completion;
    completion = hsa_signal_wait_scacquire(aql().completion_signal, HSA_SIGNAL_CONDITION_LT, 1,
                                           0xffffff, HSA_WAIT_STATE_ACTIVE);
    // completion signal should be 0.
    return completion;
  };

  // Create the executable, get symbol by name and load the code object
  ASSERT_SUCCESS(rocrtst::LoadKernelFromObjFile(this, &gpuAgent));

  // Fill up the kernel packet except header
  ASSERT_SUCCESS(rocrtst::InitializeAQLPacket(this, &aql()));

  // get queue size
  uint32_t queue_max = 0;
  ASSERT_SUCCESS(hsa_agent_get_info(gpuAgent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_max));

  // Adjust the size to the max of 1024
  queue_max = (queue_max < kMaxQueueSizeForAgent) ? queue_max : kMaxQueueSizeForAgent;

  hsa_queue_t* queue[kMaxQueue];  // command queue
  uint32_t i;
  test_validation_data user_data = {};
  ASSERT_SUCCESS( hsa_amd_register_system_event_handler(CallbackSystemErrorHandling, &user_data));
  for (i = 0; i < kMaxQueue; ++i) {
    // create queue
    ASSERT_SUCCESS(hsa_queue_create(gpuAgent, kMaxQueueSizeForAgent, HSA_QUEUE_TYPE_SINGLE, NULL,
                                    NULL, 0, 0, &queue[i]));

    user_data.cb_triggered = false;
    user_data.expected_address = reinterpret_cast<uint64_t>(queue[i]->base_address);

    // Enqueue a dispatch and make sure completion signal is 0.
    ASSERT_EQ(enqueue_dispatch(queue[i]), 0);

    // Try to delete the Queue ring buffer, this should return error.
    // Note: This will leave the hsa-runtime internal allocation table in an inconsistent state
    // because hsa-runtime clean's up its internal allocation table before calling libhsakmt to try
    // to do the actual free. So when compiled in debug mode, this will trigger a "Can't find
    // address in allocation map" warning when hsa_queue_destroy is called afterwards. This is the
    // expected behavior because trying to re-organise hsa-runtime hsa_memory_free function to
    // handle this negative use-case is not worth it and the caller is expected to call abort in
    // their system error handler.

    ASSERT_NE(hsa_memory_free(queue[i]->base_address), HSA_STATUS_SUCCESS);

    // Make sure queue is still in a working state. Enqueue a second dispatch and make sure
    // completion signal is 0.
    ASSERT_EQ(enqueue_dispatch(queue[i]), 0);

    // Make sure CallbackSystemErrorHandling was called and memory event has valid info
    ASSERT_TRUE(user_data.cb_triggered);

    if (queue[i]) hsa_queue_destroy(queue[i]);
  }

  clear_code_object();
}

#undef RET_IF_HSA_ERR
