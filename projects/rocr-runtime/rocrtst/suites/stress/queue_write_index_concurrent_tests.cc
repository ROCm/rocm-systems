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

#include "suites/stress/queue_write_index_concurrent_tests.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "common/hsatimer.h"
#include "common/concurrent_utils.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"

enum memoryOrdering {
  SCACQ_SCREL,
  SCACQUIRE,
  RELAXED,
  SCRELEASE,
  MEM_ORDERING_END};

static const uint32_t kNumThreadsForAdd = 10;

static const uint32_t kNumOfAddAtomic = 1*1024*1024;

typedef struct write_index_add_atomic_data_s {
    hsa_queue_t* queue;
    int memory_ordering_type;
} write_index_add_atomic_data_t;


static void thread_proc_write_index_add_atomic(void* data) {
  write_index_add_atomic_data_t* thread_data = reinterpret_cast<write_index_add_atomic_data_t*> (data);
  uint64_t ii;
  for (ii = 0; ii < kNumOfAddAtomic; ++ii) {
    switch (thread_data->memory_ordering_type) {
      case SCACQ_SCREL:
        hsa_queue_add_write_index_scacq_screl(thread_data->queue, 1);
        break;
      case SCACQUIRE:
        hsa_queue_add_write_index_scacquire(thread_data->queue, 1);
        break;
      case RELAXED:
        hsa_queue_add_write_index_relaxed(thread_data->queue, 1);
        break;
      case SCRELEASE:
        hsa_queue_add_write_index_screlease(thread_data->queue, 1);
        break;
      default:
        break;
    }
  }
}

static const uint32_t kNumThreadsForCas = 4;
static const uint32_t kNumOfCasAtomic = 1*1024*1024;
typedef struct write_index_cas_thread_data_s {
    hsa_queue_t* queue;
    int thread_index;
    int num_threads;
    uint64_t termination_value;
    int memory_ordering_type;
} write_index_cas_thread_data_t;

static void thread_proc_write_index_cas_atomic(void* data) {
  write_index_cas_thread_data_t* thread_data = reinterpret_cast<write_index_cas_thread_data_t*>(data);

  uint64_t ii;
  for (ii = thread_data->thread_index; ii < thread_data->termination_value; ii += thread_data->num_threads) {
    switch (thread_data->memory_ordering_type) {
      case SCACQ_SCREL:
        while ((uint64_t)ii !=
          hsa_queue_cas_write_index_scacq_screl(thread_data->queue, ii, ii + 1)) {}
          break;
     case SCACQUIRE:
        while ((uint64_t)ii !=
          hsa_queue_cas_write_index_scacquire(thread_data->queue, ii, ii + 1)) {}
          break;
     case RELAXED:
        while ((uint64_t)ii !=
          hsa_queue_cas_write_index_relaxed(thread_data->queue, ii, ii + 1)) {}
          break;
     case SCRELEASE:
        while ((uint64_t)ii !=
          hsa_queue_cas_write_index_screlease(thread_data->queue, ii, ii + 1)) {}
          break;
        }
    }
}

static const uint32_t kNumOfLoadStoreAtomic = 1*1024*1024;
// Use a 64-bit value to test the atomicity
static uint64_t kStoreValue = UINT64_MAX;

typedef struct write_index_load_atomic_thread_data_s {
  hsa_queue_t* queue;
  uint64_t num_iterations;
  int memory_ordering_type;
} write_index_load_atomic_thread_data_t;

typedef struct write_index_store_atomic_thread_data_s {
  hsa_queue_t* queue;
  uint64_t kStoreValue;
  uint64_t num_iterations;
  int memory_ordering_type;
} write_index_store_atomic_thread_data_t;

static uint64_t const WRITE_INDEX_FAILURE = 2;
void thread_proc_write_index_load_atomic(void* data) {
  write_index_load_atomic_thread_data_t* thread_data =
              reinterpret_cast<write_index_load_atomic_thread_data_t*>(data);
  uint32_t ii;
  for (ii = 0; ii < thread_data->num_iterations; ++ii) {
    uint64_t write_index = WRITE_INDEX_FAILURE;  // initalized with value other than kStoreValue
    if (SCRELEASE == thread_data->memory_ordering_type) {
      write_index = hsa_queue_load_write_index_scacquire(thread_data->queue);
    } else if (RELAXED == thread_data->memory_ordering_type) {
      write_index = hsa_queue_load_write_index_relaxed(thread_data->queue);
    }
    // The only two possible values
    EXPECT_TRUE(0 == write_index || kStoreValue == write_index);
  }
}

void thread_proc_write_index_store_atomic(void* data) {
  write_index_store_atomic_thread_data_t* thread_data =
              reinterpret_cast<write_index_store_atomic_thread_data_t*>(data);
  uint32_t ii;
  for (ii = 0; ii < thread_data->num_iterations; ++ii) {
    if (SCRELEASE == thread_data->memory_ordering_type) {
      hsa_queue_store_write_index_screlease(thread_data->queue, thread_data->kStoreValue);
    } else if (RELAXED == thread_data->memory_ordering_type) {
      hsa_queue_store_write_index_relaxed(thread_data->queue, thread_data->kStoreValue);
    }
  }
}



QueueWriteIndexConcurrentTest::QueueWriteIndexConcurrentTest(bool launch_Concurrent_AddWriteIndex,
                      bool launch_Concurrent_CasWriteIndex ,
                      bool launch_Concurrent_LoadStoreWriteIndex) :TestBase() {
  set_num_iteration(10);  // Number of iterations to execute of the main test;
                          // This is a default value which can be overridden
                          // on the command line.

  std::string name;
  std::string desc;

  name = "RocR Queue write Index Tests";
  desc = "These series of tests are Stress tests which contains different subtests ";

  if (launch_Concurrent_AddWriteIndex) {
    name += " AddWriteIndex";
    desc += " This test Verifies that the hsa_queue_write_index_add operations is atomic"
            " and 'torn' adds do not occur when this API is executed concurrently.";
  } else if (launch_Concurrent_CasWriteIndex) {
    name += " CasWriteIndex";
    desc += " This test Verifies that the hsa_queue_cas_write_index operations is atomic,"
            " and 'torn' compare and swaps do not occur when this API is executed"
            " concurrently.";
  } else if (launch_Concurrent_LoadStoreWriteIndex) {
    name += " LoadStoreWriteIndex";
    desc += " This test Verifies that the hsa_queue_write_index_load and store operations"
            " are atomic, and 'torn' loads or stores do not occur when these APIs are executed"
            " concurrently.";
  }
  set_title(name);
  set_description(desc);
}

QueueWriteIndexConcurrentTest::~QueueWriteIndexConcurrentTest(void) {
}

// Any 1-time setup involving member variables used in the rest of the test
// should be done here.
void QueueWriteIndexConcurrentTest::SetUp(void) {
  hsa_status_t err;

  TestBase::SetUp();

  err = rocrtst::SetDefaultAgents(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  err = rocrtst::SetPoolsTypical(this);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  return;
}

void QueueWriteIndexConcurrentTest::Run(void) {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  TestBase::Run();
}

void QueueWriteIndexConcurrentTest::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void QueueWriteIndexConcurrentTest::DisplayResults(void) const {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  return;
}

void QueueWriteIndexConcurrentTest::Close() {
  // This will close handles opened within rocrtst utility calls and call
  // hsa_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}




static const char kSubTestSeparator[] = "  **************************";

static void PrintDebugSubtestHeader(const char *header) {
  std::cout << "  *** QueueWriteIndexConcurrent Subtest: " << header << " ***" << std::endl;
}



// This test verify check  memory can be
// concurrently allocated from pool on ROCR agents
void QueueWriteIndexConcurrentTest::QueueAddWriteIndexAtomic(hsa_agent_t cpuAgent,
                                    hsa_agent_t gpuAgent) {
  hsa_status_t err;

  // check if the gpuAgent supports kernel dispatch
  uint32_t features = 0;
  err = hsa_agent_get_info(gpuAgent, HSA_AGENT_INFO_FEATURE, &features);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  if (0 == (features & HSA_AGENT_FEATURE_KERNEL_DISPATCH)) {
    return;
  }


  // Get max number of queues
  uint32_t queue_size;
  err = hsa_agent_get_info(gpuAgent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  // Create a queue
  hsa_queue_t* queue;
  err = hsa_queue_create(gpuAgent, queue_size, HSA_QUEUE_TYPE_SINGLE, NULL, NULL, UINT32_MAX, UINT32_MAX, &queue);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  int memory_ordering_type;
  for (memory_ordering_type = SCACQ_SCREL; memory_ordering_type < MEM_ORDERING_END; ++memory_ordering_type) {
    // Thread data
    write_index_add_atomic_data_t thread_data;
    thread_data.queue = queue;
    thread_data.memory_ordering_type = memory_ordering_type;

    // Create a test group
    rocrtst::test_group* tg_concurrent = rocrtst::TestGroupCreate(kNumThreadsForAdd);

    uint32_t kk;
    for (kk = 0; kk < kNumThreadsForAdd; kk++) {
      rocrtst::TestGroupAdd(tg_concurrent, &thread_proc_write_index_add_atomic, &thread_data, 1);
    }

    // Create threads for each test
    rocrtst::TestGroupThreadCreate(tg_concurrent);

    // Start to run tests
    rocrtst::TestGroupStart(tg_concurrent);

    // Wait all tests finish
    rocrtst::TestGroupWait(tg_concurrent);

    // Exit all tests
    rocrtst::TestGroupExit(tg_concurrent);

    // Destroy thread group and cleanup resources
    rocrtst::TestGroupDestroy(tg_concurrent);

    // Verify the write_index
    uint64_t write_index = hsa_queue_load_write_index_relaxed(queue);
    uint64_t expected = (uint64_t)(kNumOfAddAtomic * kNumThreadsForAdd);
    ASSERT_EQ(write_index, expected);

    // Restore the write_index of the queue
    hsa_queue_store_write_index_screlease(queue, 0);
  }

  // Destroy queue
  err = hsa_queue_destroy(queue);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
}




// This test verify check  memory can be
// concurrently allocated from pool on ROCR agents
void QueueWriteIndexConcurrentTest::QueueCasWriteIndexAtomic(hsa_agent_t cpuAgent, hsa_agent_t gpuAgent) {
  hsa_status_t err;

  // check if the gpuAgent supports kernel dispatch
  uint32_t features = 0;
  err = hsa_agent_get_info(gpuAgent, HSA_AGENT_INFO_FEATURE, &features);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  if (0 == (features & HSA_AGENT_FEATURE_KERNEL_DISPATCH)) {
    return;
  }


  // Get max number of queues
  uint32_t queue_size;
  err = hsa_agent_get_info(gpuAgent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  // Create a queue
  hsa_queue_t* queue;
  err = hsa_queue_create(gpuAgent, queue_size, HSA_QUEUE_TYPE_SINGLE, NULL, NULL, UINT32_MAX, UINT32_MAX, &queue);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  int memory_ordering_type;
  for (memory_ordering_type = SCACQ_SCREL; memory_ordering_type < MEM_ORDERING_END; ++memory_ordering_type) {
    // Thread data
    write_index_cas_thread_data_t thread_data[kNumThreadsForCas];

    // Create a test group
    rocrtst::test_group* tg_concurrent = rocrtst::TestGroupCreate(kNumThreadsForCas);

    uint32_t kk;
    for (kk = 0; kk < kNumThreadsForCas; ++kk) {
      thread_data[kk].queue = queue;
      thread_data[kk].thread_index = kk;
      thread_data[kk].num_threads = kNumThreadsForCas;
      thread_data[kk].memory_ordering_type = memory_ordering_type;
      thread_data[kk].termination_value = kNumOfCasAtomic;
      rocrtst::TestGroupAdd(tg_concurrent, &thread_proc_write_index_cas_atomic, thread_data + kk, 1);
    }

    // Create threads for each test
    rocrtst::TestGroupThreadCreate(tg_concurrent);

    // Start to run tests
    rocrtst::TestGroupStart(tg_concurrent);

    // Wait all tests finish
    rocrtst::TestGroupWait(tg_concurrent);

    // Exit all tests
    rocrtst::TestGroupExit(tg_concurrent);

    // Destroy thread group and cleanup resources
    rocrtst::TestGroupDestroy(tg_concurrent);

    // Verify the write_index
    uint64_t write_index = hsa_queue_load_write_index_relaxed(queue);
    uint64_t expected = (uint64_t)(kNumOfCasAtomic);
    ASSERT_EQ(write_index, expected);

    // Restore the write_index of the queue
    hsa_queue_store_write_index_screlease(queue, 0);
  }

  // Destroy queue
  err = hsa_queue_destroy(queue);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
}


// This test verify if each Agent pool's attribute information
// is consistent across multiple thread.
void QueueWriteIndexConcurrentTest::QueueLoadStoreWriteIndexAtomic(hsa_agent_t cpuAgent, hsa_agent_t gpuAgent) {
  hsa_status_t err;

  // check if the gpuAgent supports kernel dispatch
  uint32_t features = 0;
  err = hsa_agent_get_info(gpuAgent, HSA_AGENT_INFO_FEATURE, &features);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  if (0 == (features & HSA_AGENT_FEATURE_KERNEL_DISPATCH)) {
    return;
  }


  // Get max number of queues
  uint32_t queue_size;
  err = hsa_agent_get_info(gpuAgent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  // Create a queue
  hsa_queue_t* queue;
  err = hsa_queue_create(gpuAgent, queue_size, HSA_QUEUE_TYPE_SINGLE, NULL, NULL, UINT32_MAX, UINT32_MAX, &queue);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  // Use a 64-bit value to test the atomicity
  kStoreValue = UINT64_MAX;

  int memory_ordering_type;
  for (memory_ordering_type = RELAXED; memory_ordering_type < MEM_ORDERING_END; ++memory_ordering_type) {
    // Thread data
    write_index_load_atomic_thread_data_t  load_thread_data[2];
    write_index_store_atomic_thread_data_t store_thread_data[2];
    load_thread_data[0].queue = queue;
    load_thread_data[0].num_iterations = kNumOfLoadStoreAtomic;
    load_thread_data[0].memory_ordering_type = memory_ordering_type;
    load_thread_data[1].queue = queue;
    load_thread_data[1].num_iterations = kNumOfLoadStoreAtomic;
    load_thread_data[1].memory_ordering_type = memory_ordering_type;

    store_thread_data[0].queue = queue;
    store_thread_data[0].kStoreValue = 0;
    store_thread_data[0].num_iterations = kNumOfLoadStoreAtomic;
    store_thread_data[0].memory_ordering_type = memory_ordering_type;
    store_thread_data[1].queue = queue;
    store_thread_data[1].kStoreValue = kStoreValue;
    store_thread_data[1].num_iterations = kNumOfLoadStoreAtomic;
    store_thread_data[1].memory_ordering_type = memory_ordering_type;
    // Create a test group
    rocrtst::test_group* tg_concurrent = rocrtst::TestGroupCreate(4);
    rocrtst::TestGroupAdd(tg_concurrent, &thread_proc_write_index_load_atomic, load_thread_data, 1);
    rocrtst::TestGroupAdd(tg_concurrent, &thread_proc_write_index_load_atomic, load_thread_data  + 1, 1);
    rocrtst::TestGroupAdd(tg_concurrent,  &thread_proc_write_index_store_atomic, store_thread_data, 1);
    rocrtst::TestGroupAdd(tg_concurrent, &thread_proc_write_index_store_atomic, store_thread_data + 1, 1);


    // Create threads for each test
    rocrtst::TestGroupThreadCreate(tg_concurrent);

    // Start to run tests
    rocrtst::TestGroupStart(tg_concurrent);

    // Wait all tests finish
    rocrtst::TestGroupWait(tg_concurrent);

    // Exit all tests
    rocrtst::TestGroupExit(tg_concurrent);

    // Destroy thread group and cleanup resources
    rocrtst::TestGroupDestroy(tg_concurrent);
  }

  // Destroy queue
  err = hsa_queue_destroy(queue);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
}


void QueueWriteIndexConcurrentTest::QueueAddWriteIndexAtomic(void) {
  hsa_status_t err;

  if (verbosity() > 0) {
    PrintDebugSubtestHeader("QueueAddWriteIndexAtomic");
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
    QueueAddWriteIndexAtomic(cpus[0], gpus[i]);
  }

  if (verbosity() > 0) {
    std::cout << "subtest Passed" << std::endl;
    std::cout << kSubTestSeparator << std::endl;
  }
}

void QueueWriteIndexConcurrentTest::QueueCasWriteIndexAtomic(void) {
  hsa_status_t err;

  if (verbosity() > 0) {
    PrintDebugSubtestHeader("QueueCasWriteIndexAtomic");
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
    QueueCasWriteIndexAtomic(cpus[0], gpus[i]);
  }

  if (verbosity() > 0) {
    std::cout << "subtest Passed" << std::endl;
    std::cout << kSubTestSeparator << std::endl;
  }
}

void QueueWriteIndexConcurrentTest::QueueLoadStoreWriteIndexAtomic(void) {
  hsa_status_t err;

  if (verbosity() > 0) {
    PrintDebugSubtestHeader("QueueLoadStoreWriteIndexAtomic");
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
    QueueLoadStoreWriteIndexAtomic(cpus[0], gpus[i]);
  }

  if (verbosity() > 0) {
    std::cout << "subtest Passed" << std::endl;
    std::cout << kSubTestSeparator << std::endl;
  }
}

