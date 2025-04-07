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

#include "suites/negative/queue_validation.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "common/hsatimer.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"

static const uint32_t kMaxQueueSizeForAgent = 1024;
static const uint32_t kMaxQueue = 64;

typedef struct test_validation_data_t {
  bool cb_triggered;
  hsa_queue_t** queue_pointer;
  hsa_status_t  expected_status;
} test_validation_data;

static void CallbackQueueErrorHandling(hsa_status_t status, hsa_queue_t *source, void *data);

QueueValidation::QueueValidation(bool launch_InvalidDimension,
                                 bool launch_InvalidGroupMemory,
                                 bool launch_InvalidKernelObject,
                                 bool launch_InvalidPacket,
                                 bool launch_InvalidWorkGroupSize) :TestBase() {
  set_num_iteration(10);  // Number of iterations to execute of the main test;
                          // This is a default value which can be overridden
                          // on the command line.
  std::string name;
  std::string desc;

  name = "RocR Queue Validation";
  desc = "This series of tests submit different negative aql packet into the queue"
         " and verifies that queue error handling callback called with proper exception.";

  if (launch_InvalidDimension) {
    name += " For InvalidDimension";
    desc += " This test verifies that if an aql packet specifies a dimension "
            " value above 3, the queue's error handling callback will trigger";
  } else if (launch_InvalidGroupMemory) {
    name += " For InvalidGroupMemory";
    desc += " This test verifies that if an aql packet specifies an invalid group"
            " memory size, the queue's error handling.";
  } else if (launch_InvalidKernelObject) {
    name += " ForInvalidKernelObject";
    desc += " This test verifies that if an aql packet specifies an invalid"
            " kernel object, the queue's error handling callback will trigger.";
  } else if (launch_InvalidPacket) {
    name += " For InvalidPacket";
    desc += " This test verifies that if an aql packet is invalid (bad packet type),"
            " the queue's error handling callback will trigger.";
  } else if (launch_InvalidWorkGroupSize) {
    name += " For InvalidWorkGroupSize";
    desc += " This test verifies that if an aql packet specifies an invalid"
            " workgroup size, the queue's error handling callback will trigger.";
  }
  set_title(name);
  set_description(desc);

  memset(&aql(), 0, sizeof(hsa_kernel_dispatch_packet_t));
  set_kernel_file_name("dispatch_time_kernels.hsaco");
  set_kernel_name("empty_kernel");
}

QueueValidation::~QueueValidation(void) {
}

// Any 1-time setup involving member variables used in the rest of the test
// should be done here.
void QueueValidation::SetUp(void) {
  hsa_status_t err;

  TestBase::SetUp();

  /* The queue exceptions will trigger a coredump. Set the limit to 0 to disable  */
  if (getrlimit(RLIMIT_CORE, &rlimit_)) {
    perror("Could not get system rlimit\n");
  } else {
    struct rlimit rlimit_set;

    rlimit_set.rlim_cur = 0;
    rlimit_set.rlim_max = 0;

    /* Do not error if system does not allow disabling limit */
    if (setrlimit(RLIMIT_CORE, &rlimit_set))
      perror("Could not set core file size\n");
  }

  err = rocrtst::SetDefaultAgents(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  err = rocrtst::SetPoolsTypical(this);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  // Fill up the kernel packet except header
  err = rocrtst::InitializeAQLPacket(this, &aql());
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);
  return;
}

void QueueValidation::Run(void) {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  TestBase::Run();
}

void QueueValidation::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void QueueValidation::DisplayResults(void) const {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  return;
}

void QueueValidation::Close() {
  /* Restore rlimit to initial value before test - do not error if fails */
  if (setrlimit(RLIMIT_CORE, &rlimit_))
      perror("Could not set core file size\n");

  // This will close handles opened within rocrtst utility calls and call
  // hsa_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}


static const char kSubTestSeparator[] = "  **************************";

static void PrintDebugSubtestHeader(const char *header) {
  std::cout << "  *** QueueValidation Subtest: " << header << " ***" << std::endl;
}

void QueueValidation::QueueValidationForInvalidDimension(hsa_agent_t cpuAgent,
                                            hsa_agent_t gpuAgent) {
  hsa_status_t err;

  // Create the executable, get symbol by name and load the code object
  err = rocrtst::LoadKernelFromObjFile(this, &gpuAgent);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  // get queue size
  uint32_t queue_max = 0;
  err = hsa_agent_get_info(gpuAgent,
                           HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_max);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  // Adjust the size to the max of 1024
  queue_max = (queue_max < kMaxQueueSizeForAgent) ? queue_max: kMaxQueueSizeForAgent;

  hsa_queue_t *queue[kMaxQueue];  // command queue
  uint32_t ii;
  test_validation_data user_data[kMaxQueue];
  for (ii = 0; ii < kMaxQueue; ++ii) {
    // set callback flag to false if callback called then it will change to true
    user_data[ii].cb_triggered = false;
    // set the queue pointer
    user_data[ii].queue_pointer = &queue[ii];
    // set the expected status in queue error calback handling
    user_data[ii].expected_status = HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS;

    // create queue
    err = hsa_queue_create(gpuAgent,
                       queue_max, HSA_QUEUE_TYPE_SINGLE,
                       CallbackQueueErrorHandling, &user_data[ii], 0, 0, &queue[ii]);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    // setting the dimesion more than 3
    aql().setup = 4;
    aql().kernel_object = kernel_object();
    const uint32_t queue_mask = queue[ii]->size - 1;

    // Load index for writing header later to command queue at same index
    uint64_t index = hsa_queue_load_write_index_relaxed(queue[ii]);
    hsa_queue_store_write_index_relaxed(queue[ii], index + 1);

    rocrtst::WriteAQLToQueueLoc(queue[ii], index, &aql());

    aql().header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
    aql().header |= HSA_FENCE_SCOPE_SYSTEM <<
                 HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
    aql().header |= HSA_FENCE_SCOPE_SYSTEM <<
                 HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;

    void* q_base = queue[ii]->base_address;
    // Set the Aql packet header
    rocrtst::AtomicSetPacketHeader(aql().header, aql().setup,
                        &(reinterpret_cast<hsa_kernel_dispatch_packet_t*>
                            (q_base))[index & queue_mask]);


    // ringdoor bell
    hsa_signal_store_relaxed(queue[ii]->doorbell_signal, index);

    // wait for the signal long enough for the queue error handling callback to happen
    hsa_signal_value_t completion;
    completion = hsa_signal_wait_scacquire(aql().completion_signal, HSA_SIGNAL_CONDITION_LT, 1,
                                           0xffffff, HSA_WAIT_STATE_ACTIVE);
    // completion signal should not be changed.
    ASSERT_EQ(completion, 1);

    hsa_signal_store_relaxed(aql().completion_signal, 1);
  }
  sleep(1);
  for (ii = 0; ii < kMaxQueue; ++ii) {
    // queue error handling callback  should be triggered
    ASSERT_EQ(user_data[ii].cb_triggered, true);
    if (queue[ii]) { hsa_queue_destroy(queue[ii]); }
  }

  clear_code_object();
}


void QueueValidation::QueueValidationInvalidGroupMemory(hsa_agent_t cpuAgent,
                                            hsa_agent_t gpuAgent) {
  hsa_status_t err;

  // Create the executable, get symbol by name and load the code object
  err = rocrtst::LoadKernelFromObjFile(this, &gpuAgent);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  // Fill up the kernel packet except header
  err = rocrtst::InitializeAQLPacket(this, &aql());
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  // get queue size
  uint32_t queue_max = 0;
  err = hsa_agent_get_info(gpuAgent,
                           HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_max);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  // Adjust the size to the max of 1024
  queue_max = (queue_max < kMaxQueueSizeForAgent) ? queue_max: kMaxQueueSizeForAgent;

  hsa_queue_t *queue[kMaxQueue];  // command queue
  test_validation_data user_data[kMaxQueue];

  uint32_t ii;
  for (ii = 0; ii < kMaxQueue; ++ii) {
    // set callback flag to false if callback called then it will change to true
    user_data[ii].cb_triggered = false;
    // set the queue pointer
    user_data[ii].queue_pointer = &queue[ii];
    // set the expected status in queue error calback handling
    user_data[ii].expected_status = HSA_STATUS_ERROR_INVALID_ALLOCATION;

    // create queue
    err = hsa_queue_create(gpuAgent,
                       queue_max, HSA_QUEUE_TYPE_SINGLE,
                       CallbackQueueErrorHandling, &user_data[ii], 0, 0, &queue[ii]);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    aql().kernel_object = kernel_object();
    // Request a large group memory segment size
    aql().group_segment_size = (uint32_t)-1;

    const uint32_t queue_mask = queue[ii]->size - 1;

    // Load index for writing header later to command queue at same index
    uint64_t index = hsa_queue_load_write_index_relaxed(queue[ii]);
    hsa_queue_store_write_index_relaxed(queue[ii], index + 1);

    rocrtst::WriteAQLToQueueLoc(queue[ii], index, &aql());

    aql().header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
    aql().header |= HSA_FENCE_SCOPE_SYSTEM <<
                 HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
    aql().header |= HSA_FENCE_SCOPE_SYSTEM <<
                 HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;

    void* q_base = queue[ii]->base_address;
    // Set the Aql packet header
    rocrtst::AtomicSetPacketHeader(aql().header, aql().setup,
                        &(reinterpret_cast<hsa_kernel_dispatch_packet_t*>
                            (q_base))[index & queue_mask]);


    // ringdoor bell
    hsa_signal_store_relaxed(queue[ii]->doorbell_signal, index);

    // wait for the signal long enough for the queue error handling callback to happen
    hsa_signal_value_t completion;
    completion = hsa_signal_wait_scacquire(aql().completion_signal, HSA_SIGNAL_CONDITION_LT, 1,
                                           0xffffff, HSA_WAIT_STATE_ACTIVE);
    // completion signal should not be changed.
    ASSERT_EQ(completion, 1);

    hsa_signal_store_relaxed(aql().completion_signal, 1);
  }
  sleep(1);
  for (ii = 0; ii < kMaxQueue; ++ii) {
    // queue error handling callback  should be triggered
    ASSERT_EQ(user_data[ii].cb_triggered, true);
    if (queue[ii]) { hsa_queue_destroy(queue[ii]); }
  }

  clear_code_object();
}

void QueueValidation::QueueValidationForInvalidKernelObject(hsa_agent_t cpuAgent,
                                            hsa_agent_t gpuAgent) {
  hsa_status_t err;

  // Create the executable, get symbol by name and load the code object
  err = rocrtst::LoadKernelFromObjFile(this, &gpuAgent);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);


  // Fill up the kernel packet except header
  err = rocrtst::InitializeAQLPacket(this, &aql());
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  // get queue size
  uint32_t queue_max = 0;
  err = hsa_agent_get_info(gpuAgent,
                           HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_max);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  // Adjust the size to the max of 1024
  queue_max = (queue_max < kMaxQueueSizeForAgent) ? queue_max: kMaxQueueSizeForAgent;

  hsa_queue_t *queue[kMaxQueue];  // command queue
  test_validation_data user_data[kMaxQueue];
  uint32_t ii;
  for (ii = 0; ii < kMaxQueue; ++ii) {
    // set callback flag to false if callback called then it will change to true
    user_data[ii].cb_triggered = false;
    // set the queue pointer
    user_data[ii].queue_pointer = &queue[ii];
    // set the expected status in queue error calback handling
    user_data[ii].expected_status = HSA_STATUS_ERROR_INVALID_CODE_OBJECT;

    // create queue
    err = hsa_queue_create(gpuAgent,
                           kMaxQueueSizeForAgent, HSA_QUEUE_TYPE_SINGLE,
                           CallbackQueueErrorHandling, &user_data[ii], 0, 0, &queue[ii]);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    // setting the null code object
    aql().kernel_object = 0;

    const uint32_t queue_mask = queue[ii]->size - 1;

    // Load index for writing header later to command queue at same index
    uint64_t index = hsa_queue_load_write_index_relaxed(queue[ii]);
    hsa_queue_store_write_index_relaxed(queue[ii], index + 1);

    rocrtst::WriteAQLToQueueLoc(queue[ii], index, &aql());

    aql().header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
    aql().header |= HSA_FENCE_SCOPE_SYSTEM <<
                 HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
    aql().header |= HSA_FENCE_SCOPE_SYSTEM <<
                 HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;

    void* q_base = queue[ii]->base_address;
    // Set the Aql packet header
    rocrtst::AtomicSetPacketHeader(aql().header, aql().setup,
                        &(reinterpret_cast<hsa_kernel_dispatch_packet_t*>
                            (q_base))[index & queue_mask]);


    // ringdoor bell
    hsa_signal_store_relaxed(queue[ii]->doorbell_signal, index);

    // wait for the signal long enough for the queue error handling callback to happen
    hsa_signal_value_t completion;
    completion = hsa_signal_wait_scacquire(aql().completion_signal, HSA_SIGNAL_CONDITION_LT, 1,
                                           0xffffff, HSA_WAIT_STATE_ACTIVE);
    // completion signal should not be changed.
    ASSERT_EQ(completion, 1);

    hsa_signal_store_relaxed(aql().completion_signal, 1);
  }
  sleep(1);
  for (ii = 0; ii < kMaxQueue; ++ii) {
    // queue error handling callback  should be triggered
    ASSERT_EQ(user_data[ii].cb_triggered, true);
    if (queue[ii]) { hsa_queue_destroy(queue[ii]); }
  }

  clear_code_object();
}

void QueueValidation::QueueValidationForInvalidPacket(hsa_agent_t cpuAgent,
                                            hsa_agent_t gpuAgent) {
  hsa_status_t err;

  // Create the executable, get symbol by name and load the code object
  err = rocrtst::LoadKernelFromObjFile(this, &gpuAgent);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  // Fill up the kernel packet except header
  err = rocrtst::InitializeAQLPacket(this, &aql());
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  // get queue size
  uint32_t queue_max = 0;
  err = hsa_agent_get_info(gpuAgent,
                           HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_max);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  // Adjust the size to the max of 1024
  queue_max = (queue_max < kMaxQueueSizeForAgent) ? queue_max: kMaxQueueSizeForAgent;

  hsa_queue_t *queue[kMaxQueue];  // command queue
  uint32_t ii;
  test_validation_data user_data[kMaxQueue];
  for (ii = 0; ii < kMaxQueue; ++ii) {
    // set callback flag to false if callback called then it will change to true
    user_data[ii].cb_triggered = false;
    // set the queue pointer
    user_data[ii].queue_pointer = &queue[ii];
    // set the expected status in queue error calback handling
    user_data[ii].expected_status = HSA_STATUS_ERROR_INVALID_PACKET_FORMAT;

    // create queue
    err = hsa_queue_create(gpuAgent,
                       queue_max, HSA_QUEUE_TYPE_SINGLE,
                       CallbackQueueErrorHandling, &user_data[ii], 0, 0, &queue[ii]);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    const uint32_t queue_mask = queue[ii]->size - 1;

    // Load index for writing header later to command queue at same index
    uint64_t index = hsa_queue_load_write_index_relaxed(queue[ii]);
    hsa_queue_store_write_index_relaxed(queue[ii], index + 1);

    rocrtst::WriteAQLToQueueLoc(queue[ii], index, &aql());
    // setting the invalid packet type
    aql().header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
    aql().header |=  0xFFFF << HSA_PACKET_HEADER_TYPE;
    aql().kernel_object = kernel_object();

    void* q_base = queue[ii]->base_address;
    // Set the Aql packet header
    rocrtst::AtomicSetPacketHeader(aql().header, aql().setup,
                        &(reinterpret_cast<hsa_kernel_dispatch_packet_t*>
                            (q_base))[index & queue_mask]);


    // ringdoor bell
    hsa_signal_store_relaxed(queue[ii]->doorbell_signal, index);

    // wait for the signal long enough for the queue error handling callback to happen
    hsa_signal_value_t completion;
    completion = hsa_signal_wait_scacquire(aql().completion_signal, HSA_SIGNAL_CONDITION_LT, 1,
                                           0xffffff, HSA_WAIT_STATE_ACTIVE);
    // completion signal should not be changed.
    ASSERT_EQ(completion, 1);

    hsa_signal_store_relaxed(aql().completion_signal, 1);
  }
  sleep(1);
  for (ii = 0; ii < kMaxQueue; ++ii) {
    // queue error handling callback  should be triggered
    ASSERT_EQ(user_data[ii].cb_triggered, true);
    if (queue[ii]) { hsa_queue_destroy(queue[ii]); }
  }

  clear_code_object();
}

void QueueValidation::QueueValidationForInvalidWorkGroupSize(hsa_agent_t cpuAgent,
                                            hsa_agent_t gpuAgent) {
  hsa_status_t err;

  // Create the executable, get symbol by name and load the code object
  err = rocrtst::LoadKernelFromObjFile(this, &gpuAgent);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  // Fill up the kernel packet except header
  err = rocrtst::InitializeAQLPacket(this, &aql());
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  // get queue size
  uint32_t queue_max = 0;
  err = hsa_agent_get_info(gpuAgent,
                           HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_max);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  // Adjust the size to the max of 1024
  queue_max = (queue_max < kMaxQueueSizeForAgent) ? queue_max: kMaxQueueSizeForAgent;

  hsa_queue_t *queue[kMaxQueue];  // command queue
  test_validation_data user_data[kMaxQueue][3];
  uint32_t ii;
  for (ii = 0; ii < kMaxQueue; ++ii) {
    uint32_t jj;
    for (jj = 1; jj <= 3; ++jj) {
      // set callback flag to false if callback called then it will change to true
      user_data[ii][jj - 1].cb_triggered = false;
      // set the queue pointer
      user_data[ii][jj - 1].queue_pointer = &queue[ii];
      // set the expected status in queue error calback handling
      user_data[ii][jj - 1].expected_status = HSA_STATUS_ERROR_INVALID_ARGUMENT;

      // create queue
      err = hsa_queue_create(gpuAgent,
              kMaxQueueSizeForAgent, HSA_QUEUE_TYPE_SINGLE,
              CallbackQueueErrorHandling, &user_data[ii][jj - 1], 0, 0, &queue[ii]);
      ASSERT_EQ(err, HSA_STATUS_SUCCESS);

      aql().setup |= jj << HSA_KERNEL_DISPATCH_PACKET_SETUP_DIMENSIONS;
      aql().workgroup_size_x = (jj == 1) ? (uint16_t)-1 : 1;
      aql().workgroup_size_y = (jj == 2) ? (uint16_t)-1 : 1;
      aql().workgroup_size_z = (jj == 3) ? (uint16_t)-1 : 1;

      aql().kernel_object = kernel_object();

      const uint32_t queue_mask = queue[ii]->size - 1;

      // Load index for writing header later to command queue at same index
      uint64_t index = hsa_queue_load_write_index_relaxed(queue[ii]);
      hsa_queue_store_write_index_relaxed(queue[ii], index + 1);

      rocrtst::WriteAQLToQueueLoc(queue[ii], index, &aql());
      aql().header = HSA_PACKET_TYPE_KERNEL_DISPATCH;
      aql().header |= HSA_FENCE_SCOPE_SYSTEM <<
                    HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE;
      aql().header |= HSA_FENCE_SCOPE_SYSTEM <<
                    HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE;

      void* q_base = queue[ii]->base_address;
      // Set the Aql packet header
      rocrtst::AtomicSetPacketHeader(aql().header, aql().setup,
                          &(reinterpret_cast<hsa_kernel_dispatch_packet_t*>
                          (q_base))[index & queue_mask]);


      // ringdoor bell
      hsa_signal_store_relaxed(queue[ii]->doorbell_signal, index);

      // wait for the signal long enough for the queue error handling callback to happen
      hsa_signal_value_t completion;
      completion = hsa_signal_wait_scacquire(aql().completion_signal, HSA_SIGNAL_CONDITION_LT, 1,
                                             0xffffff, HSA_WAIT_STATE_ACTIVE);
      // completion signal should not be changed.
      ASSERT_EQ(completion, 1);

      hsa_signal_store_relaxed(aql().completion_signal, 1);
      if (queue[ii]) { hsa_queue_destroy(queue[ii]); }
    }
  }
  sleep(1);
  for (uint32_t ii = 0; ii < kMaxQueue; ++ii) {
    for (uint32_t jj = 0; jj < 3; ++jj) {
      // queue error handling callback  should be triggered
      ASSERT_EQ(user_data[ii][jj].cb_triggered, true);
    }
  }

  clear_code_object();
}


void QueueValidation::QueueValidationForInvalidDimension(void) {
  hsa_status_t err;
  if (verbosity() > 0) {
    PrintDebugSubtestHeader("InvalidDimensionTest");
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
    QueueValidationForInvalidDimension(cpus[0], gpus[i]);
  }

  if (verbosity() > 0) {
    std::cout << "subtest Passed" << std::endl;
    std::cout << kSubTestSeparator << std::endl;
  }
}

void QueueValidation::QueueValidationInvalidGroupMemory(void) {
  hsa_status_t err;

  if (verbosity() > 0) {
    PrintDebugSubtestHeader("InvalidGroupMemory");
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
    QueueValidationInvalidGroupMemory(cpus[0], gpus[i]);
  }

  if (verbosity() > 0) {
    std::cout << "subtest Passed" << std::endl;
    std::cout << kSubTestSeparator << std::endl;
  }
}

void QueueValidation::QueueValidationForInvalidKernelObject(void) {
  hsa_status_t err;

  if (verbosity() > 0) {
    PrintDebugSubtestHeader("InvalidKernelObject");
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
    QueueValidationForInvalidKernelObject(cpus[0], gpus[i]);
  }

  if (verbosity() > 0) {
    std::cout << "subtest Passed" << std::endl;
    std::cout << kSubTestSeparator << std::endl;
  }
}

void QueueValidation::QueueValidationForInvalidPacket(void) {
  hsa_status_t err;

  if (verbosity() > 0) {
    PrintDebugSubtestHeader("InvalidPacket");
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
    QueueValidationForInvalidPacket(cpus[0], gpus[i]);
  }

  if (verbosity() > 0) {
    std::cout << "subtest Passed" << std::endl;
    std::cout << kSubTestSeparator << std::endl;
  }
}

void QueueValidation::QueueValidationForInvalidWorkGroupSize(void) {
  hsa_status_t err;

  if (verbosity() > 0) {
    PrintDebugSubtestHeader("InvalidWorkGroupSize");
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
    QueueValidationForInvalidWorkGroupSize(cpus[0], gpus[i]);
  }

  if (verbosity() > 0) {
    std::cout << "subtest Passed" << std::endl;
    std::cout << kSubTestSeparator << std::endl;
  }
}


void CallbackQueueErrorHandling(hsa_status_t status, hsa_queue_t* source, void* data) {
  ASSERT_NE(source, nullptr);
  ASSERT_NE(data, nullptr);

  test_validation_data *debug_data = reinterpret_cast<test_validation_data*>(data);
  hsa_queue_t * queue  = *(debug_data->queue_pointer);
  debug_data->cb_triggered = true;
  // check the status
  ASSERT_EQ(status, debug_data->expected_status);
  // check the queue id and user data
  ASSERT_EQ(source->id, queue->id);
  return;
}

