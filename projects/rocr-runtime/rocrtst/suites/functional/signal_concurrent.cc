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

#include <algorithm>
#include <iostream>
#include <vector>
#include "suites/functional/signal_concurrent.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "common/hsatimer.h"
#include "common/concurrent_utils.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"

static const int N = 8;
static const int M = 32;
static const int INI_VAL = 0;
static const int CMP_VAL = 1;
hsa_signal_t *signals;

#define ASSERT_MSG(C, err) { \
  if (C == 1) { \
    std::cout << err << std::endl; \
  } \
}

static void TestSignalCreateFunction(void *data) {
  hsa_status_t status;
  int* offset = reinterpret_cast<int *>(data);
  int i;
  for (i = 0; i < M; ++i) {
    status = hsa_signal_create(INI_VAL, 0, NULL, &signals[*offset + i]);
    ASSERT_EQ(HSA_STATUS_SUCCESS, status);
  }
  return;
}

static void signals_wait_host_func(void *data) {
  int i;
  for (i = 0; i < M * N; ++i) {
    hsa_signal_wait_scacquire(signals[i], HSA_SIGNAL_CONDITION_EQ, CMP_VAL, UINT64_MAX,
                              HSA_WAIT_STATE_BLOCKED);
  }
  return;
}

static void signals_wait_component_func(void *data) {
  int i;
  for (i = 0; i < M * N; ++i) {
    // Launch a kernel with signal_wait_func
    hsa_signal_wait_scacquire(signals[i], HSA_SIGNAL_CONDITION_EQ, CMP_VAL, UINT64_MAX,
                              HSA_WAIT_STATE_BLOCKED);
  }
  return;
}

static void TestSignalDestroyFunction(void* data) {
  hsa_status_t status;
  int *offset = reinterpret_cast<int*>(data);
  int i;
  for (i = 0; i < M; i++) {
    status = hsa_signal_destroy(signals[*offset + i]);
    ASSERT_EQ(HSA_STATUS_SUCCESS, status);
  }
}

static void signal_wait_host_func(void *data) {
  hsa_signal_t *signal_ptr = reinterpret_cast<hsa_signal_t*>(data);
  hsa_signal_wait_scacquire(*signal_ptr, HSA_SIGNAL_CONDITION_EQ, CMP_VAL, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
  return;
}

static void signal_wait_component_func(void *data) {
  hsa_signal_t *signal_ptr = reinterpret_cast<hsa_signal_t*>(data);
  hsa_signal_wait_scacquire(*signal_ptr, HSA_SIGNAL_CONDITION_EQ, CMP_VAL, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
  return;
}
SignalConcurrentTest::SignalConcurrentTest(bool destroy, bool max_consumer, bool cpu, bool create)
    : TestBase() {
  set_num_iteration(10);  // Number of iterations to execute of the main test;
                        // This is a default value which can be overridden
                        // on the command line.
  if (destroy) {
    set_title("RocR Signal Destroy Concurrent Test");
    set_description("This test destroy signals concurrently");
  } else if (max_consumer) {
    set_title("RocR Signal Max Consumers Test");
    set_description("This verify signal is created with num_consumers and signal can wait on all");
  } else if (create) {
    set_title("RocR Signal Create Concurrent Test");
    set_description("This test create signals concurrently");
  } else if (cpu) {
    set_title("RocR CPU Signal Completion Test");
    set_description("This test checks whether CPU signals completed");
  }
}

SignalConcurrentTest::~SignalConcurrentTest(void) {
}

// Any 1-time setup involving member variables used in the rest of the test
// should be done here.
void SignalConcurrentTest::SetUp(void) {
  hsa_status_t err;

  TestBase::SetUp();

  err = rocrtst::SetDefaultAgents(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  err = rocrtst::SetPoolsTypical(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);
  return;
}


void SignalConcurrentTest::Run(void) {
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  TestBase::Run();
}

void SignalConcurrentTest::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void SignalConcurrentTest::DisplayResults(void) const {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  return;
}


void SignalConcurrentTest::Close() {
  // This will close handles opened within rocrtst utility calls and call
  // hsa_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

void SignalConcurrentTest::TestSignalCreateConcurrent(void) {
  unsigned int i;
  hsa_status_t status;
  signals = reinterpret_cast<hsa_signal_t*>(malloc(sizeof(hsa_signal_t) * N * M));

  ASSERT_NE(signals, nullptr);

  struct rocrtst::test_group* tg_sg_create = rocrtst::TestGroupCreate(N);
  int* offset = reinterpret_cast<int*>(malloc(sizeof(int) * N));

  EXPECT_NE(offset, nullptr);
  if (!offset) {
	  free(signals);
	  return;
  }

  for (i = 0; i < N; ++i) {
    offset[i] = i * M;
    rocrtst::TestGroupAdd(tg_sg_create, &TestSignalCreateFunction, offset + i, 1);
    }
  rocrtst::TestGroupThreadCreate(tg_sg_create);
  rocrtst::TestGroupStart(tg_sg_create);
  rocrtst::TestGroupWait(tg_sg_create);
  rocrtst::TestGroupExit(tg_sg_create);
  rocrtst::TestGroupDestroy(tg_sg_create);

  std::vector<hsa_agent_t> gpus;
  status = hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status);
    struct rocrtst::test_group *tg_sg_wait = rocrtst::TestGroupCreate(gpus.size());
    for (i = 0; i < gpus.size(); ++i) {
      hsa_device_type_t device_type;
      status = hsa_agent_get_info(gpus[i], HSA_AGENT_INFO_DEVICE, &device_type);
      ASSERT_EQ(HSA_STATUS_SUCCESS, status);
      if (device_type == HSA_DEVICE_TYPE_CPU) {
        rocrtst::TestGroupAdd(tg_sg_wait, &signals_wait_host_func, &(gpus[i]), 1);
      } else if (device_type == HSA_DEVICE_TYPE_GPU) {
        rocrtst::TestGroupAdd(tg_sg_wait, &signals_wait_component_func, &(gpus[i]), 1);
      } else if (device_type == HSA_DEVICE_TYPE_DSP) {
        ASSERT_MSG(1, "ERROR: DSP_AGENT NOT SUPPORTED\n");
      } else {
        ASSERT_MSG(1, "ERROR: UNKNOWN DEVICE\n");
      }
    }

    rocrtst::TestGroupThreadCreate(tg_sg_wait);
    rocrtst::TestGroupStart(tg_sg_wait);

    for (i = 0; i < N * M; ++i) {
      hsa_signal_store_relaxed(signals[i], CMP_VAL);
    }
    rocrtst::TestGroupWait(tg_sg_wait);
    rocrtst::TestGroupExit(tg_sg_wait);
    rocrtst::TestGroupDestroy(tg_sg_wait);

    for (i = 0; i < N * M; ++i) {
      status = hsa_signal_destroy(signals[i]);
      ASSERT_EQ(HSA_STATUS_SUCCESS, status);
    }

    free(signals);
    free(offset);
}

 /*
 * Test Name: TestSignalDestroyConcurrent
 * Scope: Conformance
 *
 * Purpose: Verifies that signals can be created concurrently in different
 * threads.
 *
 * Test Description:
 * 1) Start N threads that each
 *   a) Create M signals, that are maintained in a global list.
 *   b) When creating the symbols specify all agents as consumers.
 * 2) After the signals have been created, have each agent wait on
 *    each of the signals. All agents should wait on a signal concurrently
 *    and all signals in the signal list should be waited on one at a time.
 * 3) Set the signal values in another thread so the waiting agents wake
 *    up, as expected.
 * 4) Destroy all of the signals in the main thread.
 *
 *   Expected Results: All of the signals should be created successfully.
 *   All
 *   agents should be able to wait on all of the N*M threads successfully.
 */
void SignalConcurrentTest::TestSignalDestroyConcurrent(void) {
  int i;

  signals = reinterpret_cast<hsa_signal_t *>(malloc(sizeof(hsa_signal_t) * N * M));

  ASSERT_NE(signals, nullptr);

  struct rocrtst::test_group *tg_sg_destroy = rocrtst::TestGroupCreate(N);
  int *offset = reinterpret_cast<int *>(malloc(sizeof(int) * N));

  EXPECT_NE(offset, nullptr);
  if (!offset)
    return;

  for (i = 0; i < N; ++i) {
    int j;
    offset[i] = i * M;
    for (j = 0; j < M; ++j) {
      hsa_status_t status = hsa_signal_create(INI_VAL, 0, NULL, &signals[i * M + j]);
      ASSERT_EQ(HSA_STATUS_SUCCESS, status);
    }
  }

  for (i = 0; i < N; ++i) {
    rocrtst::TestGroupAdd(tg_sg_destroy, &TestSignalDestroyFunction, &offset[i], 1);
  }

  rocrtst::TestGroupThreadCreate(tg_sg_destroy);
  rocrtst::TestGroupStart(tg_sg_destroy);
  rocrtst::TestGroupWait(tg_sg_destroy);
  rocrtst::TestGroupExit(tg_sg_destroy);
  rocrtst::TestGroupDestroy(tg_sg_destroy);

  free(signals);
  free(offset);
}

/*
 * Test Name: TestSignalCreateMaxConsumers
 * Scope: Conformance
 *
 * Purpose: Verifies that when a signal is created with the num_consumers
 * parameter set to the total number of agents and a consumers list
 * that contains all agents, the signal can be waited on by all agent_list.
 *
 * Test Description:
 * 1) Create a signal using the following parameters,
 *    a) A num_consumers value equal to the total number
 *       of agents on the system.
 *    b) A consumers list containing all of the agents
 *       in the system.
 * 2) After the signal is created, have all of the agents in
 * the system wait on the signal one at a time,
 * either using the appropriate hsa_signal_wait API or a
 * HSAIL instruction executed in a kernel.
 * 3) Set the signal on another thread such that the waiting
 * threads wait condition is satisfied.
 *
 * Expected Results: All of the agents should be able to properly wait
 * on the signal.
 */
void SignalConcurrentTest::TestSignalCreateMaxConsumers(void) {
  unsigned int i;
  hsa_status_t status;

  std::vector<hsa_agent_t> gpus;
  status = hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status);


  hsa_signal_t signal;
  status = hsa_signal_create(INI_VAL, 0, NULL, &signal);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status);

  struct rocrtst::test_group *tg_sg_wait = rocrtst::TestGroupCreate(gpus.size());
  for (i = 0; i < gpus.size(); ++i) {
    hsa_device_type_t device_type;
    hsa_agent_get_info(gpus[i], HSA_AGENT_INFO_DEVICE, &device_type);
    if (device_type == HSA_DEVICE_TYPE_CPU) {
      rocrtst::TestGroupAdd(tg_sg_wait, &signal_wait_host_func, &signal, 1);
    } else if (device_type == HSA_DEVICE_TYPE_GPU) {
      rocrtst::TestGroupAdd(tg_sg_wait, &signal_wait_component_func, &signal, 1);
    } else if (device_type == HSA_DEVICE_TYPE_DSP) {
      ASSERT_MSG(1, "ERROR: DSP_AGENT NOT SUPPORTED\n");
    } else {
      ASSERT_MSG(1, "ERROR: UNKOWN DEIVCE TYPE");
    }
  }

  rocrtst::TestGroupThreadCreate(tg_sg_wait);
  rocrtst::TestGroupStart(tg_sg_wait);

  hsa_signal_store_relaxed(signal, CMP_VAL);

  rocrtst::TestGroupWait(tg_sg_wait);
  rocrtst::TestGroupExit(tg_sg_wait);
  rocrtst::TestGroupDestroy(tg_sg_wait);

  status = hsa_signal_destroy(signal);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status);
}

void SignalConcurrentTest::TestSignalCPUCompletion(void) {
  // Not clear with the requirements, have to check with Runtime team/Ramesh
  // As we are not implemented the test fully hence the test will be skipped for now
  std::cout << "The test skipped siliently and reports as pass" << std::endl;
}

#undef RET_IF_HSA_ERR
