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
#include <unistd.h>

#include "suites/functional/time_stamp.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "common/hsatimer.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

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

// The runtime maps GPU timestamps onto the system clock from correlated clock samples that it
// refreshes about 16 times per second while translation is going on. A refresh must not change
// the translation of any tick that was already translated: otherwise the end of one packet and
// the start of the next, which are the same instant on the GPU, can be reported out of order
// when they are translated on opposite sides of a refresh. Barrier packets are submitted in
// batches spread over more than a second so that many refreshes happen in between, each batch
// is translated right after it completes (as a profiler does), and at the end every raw tick is
// translated again and compared with the value handed out the first time.
void TimeStamp::TimestampTranslationStabilityTest(void) {
  PrinTimeStampSubtestHeader(
      "TimestampTranslationStabilityTest: translation stays stable across clock refreshes");
  hsa_status_t err;
  std::vector<hsa_agent_t> gpus;
  err = hsa_iterate_agents(rocrtst::IterateGPUAgents, &gpus);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  ASSERT_GT(gpus.size(), (uint32_t)0);
  uint32_t queue_size = 0;
  ASSERT_SUCCESS(hsa_agent_get_info(gpus[0], HSA_AGENT_INFO_QUEUE_MAX_SIZE, &queue_size));
  err = hsa_queue_create(gpus[0], queue_size, HSA_QUEUE_TYPE_MULTI, nullptr, nullptr, 0, 0, &queue);
  ASSERT_EQ(err, HSA_STATUS_SUCCESS);
  ASSERT_SUCCESS(hsa_amd_profiling_set_profiler_enabled(queue, 1));
  for (uint32_t i = 0; i < NUM_BARRIERS; ++i) {
    ASSERT_SUCCESS(hsa_signal_create(1, 0, NULL, &completion_signals[i]));
  }

  struct Record {
    uint64_t raw_start, raw_end;  // GPU ticks written by the command processor
    uint64_t start, end;          // their first translation to system time
  };
  const uint32_t kBatches = 40;  // 40 x 40 ms covers tens of calibration refreshes
  std::vector<Record> records;
  records.reserve(kBatches * NUM_BARRIERS);
  for (uint32_t b = 0; b < kBatches; ++b) {
    for (uint32_t i = 0; i < NUM_BARRIERS; ++i) {
      hsa_signal_store_relaxed(completion_signals[i], 1);
      hsa_barrier_and_packet_t barrier_pkt;
      memset(&barrier_pkt, 0, sizeof(barrier_pkt));
      barrier_pkt.header = HSA_PACKET_TYPE_BARRIER_AND | (1 << HSA_PACKET_HEADER_BARRIER) |
                           (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
                           (HSA_FENCE_SCOPE_NONE << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
      barrier_pkt.completion_signal = completion_signals[i];
      uint64_t index = hsa_queue_load_write_index_relaxed(queue);
      hsa_queue_store_write_index_relaxed(queue, index + 1);
      reinterpret_cast<hsa_barrier_and_packet_t*>(queue->base_address)[index % queue->size] =
          barrier_pkt;
      hsa_signal_store_relaxed(queue->doorbell_signal, index);
    }
    for (uint32_t i = 0; i < NUM_BARRIERS; ++i) {
      while (hsa_signal_wait_scacquire(completion_signals[i], HSA_SIGNAL_CONDITION_LT, 1,
                                       (uint64_t)-1, HSA_WAIT_STATE_ACTIVE) >= 1) {
      }
      const amd_signal_t* raw = reinterpret_cast<const amd_signal_t*>(completion_signals[i].handle);
      hsa_amd_profiling_dispatch_time_t time;
      ASSERT_SUCCESS(hsa_amd_profiling_get_dispatch_time(gpus[0], completion_signals[i], &time));
      records.push_back(Record{raw->start_ts, raw->end_ts, time.start, time.end});
    }
    usleep(40000);
  }

  // Consecutive barrier packets execute in order, so wherever the GPU timestamps are ordered
  // the translated timestamps must be ordered too.
  uint32_t raw_disorder = 0;
  for (size_t i = 1; i < records.size(); ++i) {
    if (records[i - 1].raw_end > records[i].raw_start) {
      ++raw_disorder;
      continue;
    }
    ASSERT_LE(records[i - 1].end, records[i].start)
        << "translated end of packet " << i - 1 << " lands after the start of packet " << i;
  }
  EXPECT_EQ(raw_disorder, 0u) << "GPU timestamps of consecutive barrier packets out of order";

  // Translating the same ticks again after all those refreshes must reproduce the values that
  // were handed out the first time.
  for (size_t i = 0; i < records.size(); ++i) {
    uint64_t start = 0, end = 0;
    ASSERT_SUCCESS(
        hsa_amd_profiling_convert_tick_to_system_domain(gpus[0], records[i].raw_start, &start));
    ASSERT_SUCCESS(
        hsa_amd_profiling_convert_tick_to_system_domain(gpus[0], records[i].raw_end, &end));
    ASSERT_EQ(start, records[i].start) << "start of packet " << i << " was translated differently";
    ASSERT_EQ(end, records[i].end) << "end of packet " << i << " was translated differently";
  }
  if (verbosity() > 0) {
    std::cout << "Translated " << records.size() << " barrier packets over " << kBatches
              << " batches; all re-translations identical" << std::endl;
  }
}
#undef RET_IF_HSA_ERR
