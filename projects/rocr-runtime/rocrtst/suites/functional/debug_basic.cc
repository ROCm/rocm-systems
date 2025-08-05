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

#include "suites/functional/debug_basic.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "common/hsatimer.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"

#define M_ORDER 64
#define M_GET(M, I, J) M[I * M_ORDER + J]
#define M_SET(M, I, J, V) M[I * M_ORDER + J] = V

static const uint32_t kNumBufferElements = 256;
typedef struct test_debug_data_t {
  bool trap_triggered;
  hsa_queue_t** queue_pointer;
} test_debug_data;

static void TestDebugTrap(hsa_status_t status, hsa_queue_t *source, void *data);

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

DebugBasicTest::DebugBasicTest(void) :
    TestBase() {
  set_num_iteration(10);  // Number of iterations to execute of the main test;
                          // This is a default value which can be overridden
                          // on the command line.

  set_title("RocR Debug Function Tests");
  set_description("This series of tests check debug related functions.");
  set_kernel_file_name("vector_add_debug_trap_kernels.hsaco");
  set_kernel_name("vector_add_debug_trap");
}

DebugBasicTest::~DebugBasicTest(void) {
}

// Any 1-time setup involving member variables used in the rest of the test
// should be done here.
void DebugBasicTest::SetUp(void) {
  hsa_status_t err;

  TestBase::SetUp();

  err = rocrtst::SetDefaultAgents(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  err = rocrtst::SetPoolsTypical(this);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  return;
}

void DebugBasicTest::Run(void) {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  TestBase::Run();
}

void DebugBasicTest::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void DebugBasicTest::DisplayResults(void) const {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  return;
}

void DebugBasicTest::Close() {
  // This will close handles opened within rocrtst utility calls and call
  // hsa_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

typedef struct __attribute__((aligned(16))) arguments_t {
  const int *a;
  const int *b;
  const int *c;
  int *d;
  int *e;
} arguments;

arguments *vectorAddKernArgs = NULL;

static const char kSubTestSeparator[] = "  **************************";

static void PrintDebugSubtestHeader(const char *header) {
  std::cout << "  *** Debug Basic Subtest: " << header << " ***" << std::endl;
}

void DebugBasicTest::VectorAddDebugTrapTest(hsa_agent_t cpuAgent,
                                            hsa_agent_t gpuAgent) {
  hsa_status_t err;
  hsa_queue_t *queue = NULL;  // command queue
  hsa_signal_t signal = {0};  // completion signal

  int *M_IN0 = NULL;
  int *M_IN1 = NULL;
  int *M_RESULT_DEVICE = NULL;
  int M_RESULT_HOST[M_ORDER * M_ORDER];

  // get queue size
  uint32_t queue_size = 0;
  err = hsa_agent_get_info(gpuAgent,
                           HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  test_debug_data user_data{.trap_triggered = false,
                            .queue_pointer = &queue};

  // create queue
  err = hsa_queue_create(gpuAgent,
                         queue_size, HSA_QUEUE_TYPE_MULTI,
                         TestDebugTrap, &user_data, 0, 0, &queue);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  // Find a memory pool that supports kernel arguments.
  hsa_amd_memory_pool_t kernarg_pool;
  err = hsa_amd_agent_iterate_memory_pools(cpuAgent,
                                           rocrtst::GetKernArgMemoryPool,
                                           &kernarg_pool);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  // Get System Memory Pool on the cpuAgent to allocate host side buffers
  hsa_amd_memory_pool_t global_pool;
  err = hsa_amd_agent_iterate_memory_pools(cpuAgent,
                                           rocrtst::GetGlobalMemoryPool,
                                           &global_pool);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  // allocate input and output kernel arguments
  err = hsa_amd_memory_pool_allocate(global_pool,
                                     M_ORDER * M_ORDER * sizeof(int), 0,
                                     reinterpret_cast<void**>(&M_IN0));
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  err = hsa_amd_memory_pool_allocate(global_pool,
                                     M_ORDER * M_ORDER * sizeof(int), 0,
                                     reinterpret_cast<void**>(&M_IN1));
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  err = hsa_amd_memory_pool_allocate(global_pool,
                                     M_ORDER * M_ORDER * sizeof(int), 0,
                                     reinterpret_cast<void**>(&M_RESULT_DEVICE));
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  // create kernel arguments
  err = hsa_amd_memory_pool_allocate(kernarg_pool,
                                     sizeof(arguments), 0,
                                     reinterpret_cast<void**>(&vectorAddKernArgs));
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  // Allow gpuAgent access to all allocated system memory.
  err = hsa_amd_agents_allow_access(1, &gpuAgent, NULL, M_IN0);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  err = hsa_amd_agents_allow_access(1, &gpuAgent, NULL, M_IN1);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  err = hsa_amd_agents_allow_access(1, &gpuAgent, NULL, M_RESULT_DEVICE);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  err = hsa_amd_agents_allow_access(1, &gpuAgent, NULL, vectorAddKernArgs);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  memset(M_RESULT_HOST, 0, M_ORDER * M_ORDER * sizeof(int));
  memset(M_RESULT_DEVICE, 0, M_ORDER * M_ORDER * sizeof(int));

  vectorAddKernArgs->a = M_IN0;
  vectorAddKernArgs->b = M_IN1;
  vectorAddKernArgs->c = M_RESULT_DEVICE;

  // initialize input and run on host
  srand(time(NULL));
  for (int i = 0; i < M_ORDER; ++i) {
    for (int j = 0; j < M_ORDER; ++j) {
      M_SET(M_IN0, i, j, (1 + rand() % 10));
      M_SET(M_IN1, i, j, (1 + rand() % 10));
    }
  }

  for (int i = 0; i < M_ORDER; ++i) {
    for (int j = 0; j < M_ORDER; ++j) {
      int s = M_GET(M_IN0, i, j) + M_GET(M_IN1, i, j);
      M_SET(M_RESULT_HOST, i, j, s);
    }
  }

  // Create the executable, get symbol by name and load the code object
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
  aql.header = 0;
  aql.setup = 1;
  aql.workgroup_size_x = 64;
  aql.workgroup_size_y = 1;
  aql.workgroup_size_z = 1;
  aql.grid_size_x = M_ORDER * M_ORDER;
  aql.grid_size_y = 1;
  aql.grid_size_z = 1;
  aql.private_segment_size = 0;
  aql.group_segment_size = 0;
  aql.kernel_object = kernel_object();  // kernel_code;
  aql.kernarg_address = vectorAddKernArgs;
  aql.completion_signal = signal;

  // const uint32_t queue_size = queue->size;
  const uint32_t queue_mask = queue->size - 1;

  // write to command queue
  uint64_t index = hsa_queue_load_write_index_relaxed(queue);

  hsa_queue_store_write_index_relaxed(queue, index + 1);

  rocrtst::WriteAQLToQueueLoc(queue, index, &aql);

  uint32_t aql_header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
  aql_header |= HSA_FENCE_SCOPE_SYSTEM <<
                HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
  aql_header |= HSA_FENCE_SCOPE_SYSTEM <<
                HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;

  void* q_base = queue->base_address;
  rocrtst::AtomicSetPacketHeader(aql_header, aql.setup,
                        &(reinterpret_cast<hsa_kernel_dispatch_packet_t*>
                            (q_base))[index & queue_mask]);

  // ringdoor bell
  hsa_signal_store_relaxed(queue->doorbell_signal, index);

  // wait for the signal long enough for the debug trap event to happen
  hsa_signal_value_t completion;
  completion = hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, 1,
                                         0xffffff, HSA_WAIT_STATE_ACTIVE);

  // completion signal should not be changed.
  ASSERT_EQ(completion, 1);

  // trap should be triggered
  ASSERT_EQ(user_data.trap_triggered, true);

  hsa_signal_store_relaxed(signal, 1);

  if (M_IN0) { hsa_memory_free(M_IN0); }
  if (M_IN1) { hsa_memory_free(M_IN1); }
  if (M_RESULT_DEVICE) {hsa_memory_free(M_RESULT_DEVICE); }
  if (vectorAddKernArgs) { hsa_memory_free(vectorAddKernArgs); }
  if (signal.handle) { hsa_signal_destroy(signal); }
  if (queue) { hsa_queue_destroy(queue); }
  std::cout << kSubTestSeparator << std::endl;
}

void DebugBasicTest::VectorAddDebugTrapTest(void) {
  hsa_status_t err;

  PrintDebugSubtestHeader("VectorAddDebugTrapTest");

  // find all cpu agents
  std::vector<hsa_agent_t> cpus;
  err = hsa_iterate_agents(rocrtst::IterateCPUAgents, &cpus);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  // find all gpu agents
  std::vector<hsa_agent_t> gpus;
  err = hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  for (unsigned int i = 0 ; i< gpus.size(); ++i) {
    VectorAddDebugTrapTest(cpus[0], gpus[i]);
  }

  if (verbosity() > 0) {
    std::cout << "subtest Passed" << std::endl;
    std::cout << kSubTestSeparator << std::endl;
  }
}

void TestDebugTrap(hsa_status_t status, hsa_queue_t *source, void *data) {
  std::cout<< "runtime catched trap instruction successfully"<< std::endl;
  ASSERT_NE(source, nullptr);
  ASSERT_NE(data, nullptr);

  test_debug_data *debug_data = reinterpret_cast<test_debug_data*>(data);
  hsa_queue_t * queue  = *(debug_data->queue_pointer);
  debug_data->trap_triggered = true;
  // check the status
  ASSERT_EQ(status, HSA_STATUS_ERROR_EXCEPTION);

  // check the queue id and user data
  ASSERT_EQ(source->id, queue->id);
  std::cout<< "custom queue error handler completed successfully"<< std::endl;
}

#undef RET_IF_HSA_ERR
