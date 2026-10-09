/*
* Copyright © Advanced Micro Devices, Inc., or its affiliates.
*
* SPDX-License-Identifier: MIT
*/

#include <algorithm>
#include <iostream>
#include <vector>
#include <memory>
#include <sys/sysinfo.h>

#include "suites/functional/time_stamp.h"
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


TimeStamp::TimeStamp(void) :
    TestBase() {
  set_num_iteration(10);  // Number of iterations to execute of the main test;
                          // This is a default value which can be overridden
                          // on the command line.
  set_title("RocR verify clock counter");
  set_description("This series of tests captures various sets of  timestamps from KFD,"
    " to get snapshots of CPU vs GPU tickets/clock counters.");
}

TimeStamp::~TimeStamp(void) {
  // If Close() wasn't called (ASSERT failure), clean up here
  if (!resources_free) {
    FreeResources();
    // Note: HSA is still initialized, so we can safely destroy resources
    // TestBase::Close() will be called by gtest's TearDown if it exists,
    // or HSA will leak but at least our signals/queue won't
  }
}

// Any 1-time setup involving member variables used in the rest of the test
// should be done here.
void TimeStamp::SetUp(void) {
  hsa_status_t err;

  TestBase::SetUp();
  if (test_skipped_) return;

  err = rocrtst::SetDefaultAgents(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  err = rocrtst::SetPoolsTypical(this);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  return;
}

void TimeStamp::Run(void) {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  TestBase::Run();
}

void TimeStamp::DisplayTestInfo(void) {
  TestBase::DisplayTestInfo();
}

void TimeStamp::DisplayResults(void) const {
  // Compare required profile for this test case with what we're actually
  // running on
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  return;
}

void TimeStamp::FreeResources() {
  // Clean up queue and signals
  if (queue) {
    hsa_queue_destroy(queue);
    queue = nullptr;
  }

  for (uint32_t i = 0; i < NUM_BARRIERS; ++i) {
    if (completion_signals[i].handle) {
      hsa_signal_destroy(completion_signals[i]);
      completion_signals[i].handle = 0;
    }
  }
  resources_free = true;
}

void TimeStamp::Close() {
  // Clean up HSA resources BEFORE calling hsa_shut_down()
  FreeResources();

  // This will close handles opened within rocrtst utility calls and call
  // hsa_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

static const char kSubTestSeparator[] = "  **************************";

static void PrinTimeStampSubtestHeader(const char *header) {
  std::cout << "  *** TimeStamp Subtest: " << header << " ***" << std::endl;
}

void TimeStamp::TimeStampTest (void) {
  const int NUM_COUNTERS = 50;
  hsa_status_t err;
  std::vector<std::shared_ptr<rocrtst::agent_pools_t>> agent_pools;
  char ag_name[64];
  hsa_device_type_t ag_type;
  PrinTimeStampSubtestHeader("TimeStampTest: verifing clock counter IOCTLs");

  err = rocrtst::GetAgentPools(&agent_pools);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  std::cout<<"Total number of agent_pools: "<<agent_pools.size()<<std::endl;
  for (auto a : agent_pools) {
      err = hsa_agent_get_info(a->agent, HSA_AGENT_INFO_NAME, ag_name);
      ASSERT_EQ(err, HSA_STATUS_SUCCESS);

      err = hsa_agent_get_info(a->agent, HSA_AGENT_INFO_DEVICE, &ag_type);
      ASSERT_EQ(err, HSA_STATUS_SUCCESS);
      if (verbosity() > 0){
          std::cout << std::endl<<"  Agent: " << ag_name;
      }
      switch (ag_type) {
          case HSA_DEVICE_TYPE_GPU:{
              hsa_amd_clock_counters_t counter[NUM_COUNTERS];
              if (verbosity() > 0){
                std::cout << "(GPU)"<<std::endl;
              }
              for(int i = 0; i < NUM_COUNTERS ;i++){
                ASSERT_EQ(hsa_agent_get_info(a->agent, (hsa_agent_info_t) HSA_AMD_AGENT_INFO_CLOCK_COUNTERS,
                        &counter[i]), HSA_STATUS_SUCCESS);
                if(i > 0){
                    ASSERT_GE(counter[i].gpu_clock_counter, counter[i-1].gpu_clock_counter);
                    ASSERT_GE(counter[i].cpu_clock_counter, counter[i-1].cpu_clock_counter);
                    ASSERT_GE(counter[i].system_clock_counter, counter[i-1].system_clock_counter);
                 }
                if (verbosity() > 0){
                    std::cout<<" gpu_clock_counter:    "<<counter[i].gpu_clock_counter <<std::endl;
                    std::cout<<" cpu_clock_counter:    "<<counter[i].cpu_clock_counter <<std::endl;
                    std::cout<<" system_clock_counter: "<<counter[i].system_clock_counter <<std::endl;
                    std::cout<<std::endl;
                 }
              }
            }
            break;
          default:
            std::cout<<std::endl;
            break;
        } //switch ends.
    } //for loop iterating over agent_pools ends
}

void TimeStamp::BarrierPacketTimestampValidationTest(void) {

  PrinTimeStampSubtestHeader("BarrierPacketTimestampValidationTest:test to verify timestamps in AQL packets");
  hsa_status_t err;
  // find all gpu agents
  std::vector<hsa_agent_t> gpus;
  err = hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);

  // - allocate queue
  uint32_t queue_size = 0;
  ASSERT_GT(gpus.size(), (uint32_t)0);
  ASSERT_SUCCESS(hsa_agent_get_info(gpus[0], HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size));
  

  err = hsa_queue_create(gpus[0], queue_size, HSA_QUEUE_TYPE_MULTI, nullptr, nullptr, 0, 0, &queue);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  // Enable profiling on the queue to collect timestamps
  err = hsa_amd_profiling_set_profiler_enabled(queue, 1);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  std::cout << "Profiling enabled on queue" << std::endl;

  // - Create new signals
  // - Create a completion signal for the barrier packet
  for (uint32_t i = 0; i < NUM_BARRIERS; ++i) {
    err = hsa_signal_create(1, 0, NULL, &completion_signals[i]);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  }

  std::cout << "Submitting " << NUM_BARRIERS << " barrier packets..." << std::endl;

  // Submit 100 barrier packets
  for (uint32_t i = 0; i < NUM_BARRIERS; ++i) {
    // Create a Barrier-AND packet
    // - Place a Barrier-Value packet into the queue
    hsa_barrier_and_packet_t barrier_pkt;
    memset(&barrier_pkt, 0, sizeof(barrier_pkt));
    barrier_pkt.header = HSA_PACKET_TYPE_BARRIER_AND | (1 << HSA_PACKET_HEADER_BARRIER) |
                        (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
                        (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    barrier_pkt.completion_signal = completion_signals[i];

    // Get the write index and reserve a slot in the queue
    uint64_t index = hsa_queue_load_write_index_relaxed(queue);
    hsa_queue_store_write_index_relaxed(queue, index + 1);
    reinterpret_cast<hsa_barrier_and_packet_t*>(queue->base_address)[index % queue->size] = barrier_pkt;

    hsa_signal_store_relaxed(queue->doorbell_signal, index);
  }
  std::cout << "Waiting for barrier packets to complete..." << std::endl;

  hsa_amd_profiling_dispatch_time_t dispatch_times[NUM_BARRIERS];
  // - Wait for the completion-signals to change from 1 to 0.
  // Wait for the last barriers to complete and collect timing data
  while (hsa_signal_wait_scacquire(completion_signals[NUM_BARRIERS - 1],
      HSA_SIGNAL_CONDITION_LT, 1, (uint64_t)-1, HSA_WAIT_STATE_ACTIVE)) {}
  for (uint32_t i = 0; i < NUM_BARRIERS; ++i) {
    // Get the dispatch time for this barrier packet
    err = hsa_amd_profiling_get_dispatch_time(gpus[0], completion_signals[i], &dispatch_times[i]);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    // Assert failures for Incorrect timestamps.
    ASSERT_GE(dispatch_times[i].end, dispatch_times[i].start);

    if(i > 0) {
      ASSERT_GE(dispatch_times[i].start, dispatch_times[i - 1].end);
    }
    if (verbosity() > 0) {
      std::cout << "Barrier " << i << " - Start: " << dispatch_times[i].start
              << ", End: " << dispatch_times[i].end
              << ", Duration: " << (dispatch_times[i].end - dispatch_times[i].start)
              << " ticks" << std::endl;
    }
  }
}

#undef RET_IF_HSA_ERR
