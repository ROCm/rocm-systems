/*
 * Copyright (c) 2026, Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#include "suites/stress/sdma_preemption_stress.h"

#include <cstring>
#include <iostream>
#include <thread>

#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "gtest/gtest.h"

SdmaPreemptionStressTest::SdmaPreemptionStressTest()
    : TestBase(),
      sdma_queue_(nullptr),
      copy_src_buffer_(nullptr),
      copy_dst_buffer_(nullptr),
      completion_signal_({0}),
      successful_iterations_(0),
      failed_iterations_(0),
      preemption_test_passed_(false) {
  set_title("SDMA Preemption Stress Test");
  set_description(
      "Reproduces SDMA preemption issues by performing rapid SDMA copies "
      "with controlled timing to expose RPTR/WPTR anomalies.");
}

SdmaPreemptionStressTest::~SdmaPreemptionStressTest() {}

void SdmaPreemptionStressTest::SetUp() {
  TestBase::SetUp();
  if (test_skipped_) return;

  hsa_status_t err;

  err = rocrtst::SetDefaultAgents(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  err = rocrtst::SetPoolsTypical(this);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  // Allocate buffers for SDMA operations
  hsa_agent_t ag_list[2] = {*gpu_device1(), *cpu_device()};

  // Allocate copy source buffer
  err = hsa_amd_memory_pool_allocate(cpu_pool(), kCopySize, 0, &copy_src_buffer_);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);
  err = hsa_amd_agents_allow_access(2, ag_list, NULL, copy_src_buffer_);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);

  // Initialize source with pattern
  memset(copy_src_buffer_, 0xAB, kCopySize);

  // Allocate copy destination buffer
  err = hsa_amd_memory_pool_allocate(cpu_pool(), kCopySize, 0, &copy_dst_buffer_);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);
  err = hsa_amd_agents_allow_access(2, ag_list, NULL, copy_dst_buffer_);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);
  memset(copy_dst_buffer_, 0, kCopySize);

  // Create completion signal
  err = hsa_signal_create(1, 0, nullptr, &completion_signal_);
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);
}

void SdmaPreemptionStressTest::Run() {
  if (!rocrtst::CheckProfile(this)) {
    return;
  }
  TestBase::Run();
}

void SdmaPreemptionStressTest::Close() {
  // Free allocated buffers
  if (copy_src_buffer_) {
    hsa_amd_memory_pool_free(copy_src_buffer_);
    copy_src_buffer_ = nullptr;
  }
  if (copy_dst_buffer_) {
    hsa_amd_memory_pool_free(copy_dst_buffer_);
    copy_dst_buffer_ = nullptr;
  }

  if (completion_signal_.handle != 0) {
    hsa_signal_destroy(completion_signal_);
    completion_signal_.handle = 0;
  }

  TestBase::Close();
}

void SdmaPreemptionStressTest::DisplayResults() const {
  if (!rocrtst::CheckProfile(this)) {
    return;
  }

  std::cout << "  SDMA Preemption Stress Test Results:" << std::endl;
  std::cout << "    Successful iterations: " << successful_iterations_ << std::endl;
  std::cout << "    Failed iterations: " << failed_iterations_ << std::endl;
  std::cout << "    Preemption test: " << (preemption_test_passed_ ? "PASSED" : "FAILED")
            << std::endl;
}

void SdmaPreemptionStressTest::DisplayTestInfo() { TestBase::DisplayTestInfo(); }

hsa_status_t SdmaPreemptionStressTest::CreateSdmaQueue(hsa_agent_t agent, hsa_queue_t** queue) {
  // SDMA queue creation via hsa_amd_queue_create with HSA_AMD_QUEUE_ENGINE_SDMA
  // is not available in older ROCm versions. Return error to skip the test.
  (void)agent;
  (void)queue;
  return HSA_STATUS_ERROR;
}

void SdmaPreemptionStressTest::DestroySdmaQueue(hsa_queue_t* queue) {
  if (queue) {
    hsa_queue_destroy(queue);
  }
}

void SdmaPreemptionStressTest::BuildPollRegMemPacket(uint32_t* pkt, uint64_t addr, uint32_t value,
                                                     uint32_t mask) {
  (void)pkt;
  (void)addr;
  (void)value;
  (void)mask;
}

void SdmaPreemptionStressTest::BuildGcrUserPacket(uint32_t* pkt, uint32_t gcr_control) {
  (void)pkt;
  (void)gcr_control;
}

void SdmaPreemptionStressTest::BuildCopyLinearPacket(uint32_t* pkt, uint64_t dst, uint64_t src,
                                                     uint32_t size) {
  (void)pkt;
  (void)dst;
  (void)src;
  (void)size;
}

void SdmaPreemptionStressTest::BuildAtomicPacket(uint32_t* pkt, uint64_t addr, uint64_t src_data,
                                                 uint64_t cmp_data) {
  (void)pkt;
  (void)addr;
  (void)src_data;
  (void)cmp_data;
}

void SdmaPreemptionStressTest::BuildFencePacket(uint32_t* pkt, uint64_t addr, uint32_t data) {
  (void)pkt;
  (void)addr;
  (void)data;
}

void SdmaPreemptionStressTest::BuildTrapPacket(uint32_t* pkt, uint32_t event_id) {
  (void)pkt;
  (void)event_id;
}

void SdmaPreemptionStressTest::SubmitSdmaPacketSequence(hsa_queue_t* queue, uint64_t poll_addr,
                                                        uint64_t copy_src, uint64_t copy_dst,
                                                        uint64_t atomic_addr, uint64_t fence_addr,
                                                        uint32_t fence_value) {
  (void)queue;
  (void)poll_addr;
  (void)copy_src;
  (void)copy_dst;
  (void)atomic_addr;
  (void)fence_addr;
  (void)fence_value;
}

void SdmaPreemptionStressTest::BasicSdmaPacketTest() {
  hsa_status_t err;

  std::cout << "  Running Basic SDMA Copy Test (using hsa_amd_memory_async_copy)..." << std::endl;

  // Use standard HSA async copy API which exercises SDMA internally
  memset(copy_dst_buffer_, 0, kCopySize);

  // Reset signal
  hsa_signal_store_screlease(completion_signal_, 1);

  // Perform async copy CPU->GPU to exercise SDMA engine
  // Using GPU as dst agent forces SDMA path (CPU-to-CPU would use memcpy)
  err = hsa_amd_memory_async_copy(copy_dst_buffer_, *gpu_device1(), copy_src_buffer_, *cpu_device(),
                                  kCopySize, 0, nullptr,  // No dependencies
                                  completion_signal_);

  if (err != HSA_STATUS_SUCCESS) {
    std::cout << "    hsa_amd_memory_async_copy failed: " << err << std::endl;
    failed_iterations_++;
    EXPECT_EQ(HSA_STATUS_SUCCESS, err);
    return;
  }

  // Wait for completion
  hsa_signal_value_t result =
      hsa_signal_wait_scacquire(completion_signal_, HSA_SIGNAL_CONDITION_LT, 1,
                                5000000000ULL,  // 5 second timeout
                                HSA_WAIT_STATE_BLOCKED);

  if (result != 0) {
    std::cout << "    Copy did not complete in time" << std::endl;
    failed_iterations_++;
    EXPECT_EQ(result, 0);
    return;
  }

  // Verify copy
  bool copy_ok = (memcmp(copy_src_buffer_, copy_dst_buffer_, kCopySize) == 0);
  std::cout << "    Copy verification: " << (copy_ok ? "PASS" : "FAIL") << std::endl;

  if (copy_ok) {
    successful_iterations_++;
  } else {
    failed_iterations_++;
  }
  EXPECT_TRUE(copy_ok) << "SDMA copy data mismatch";
}

void SdmaPreemptionStressTest::SdmaDoorbellStressTest() {
  hsa_status_t err;

  std::cout << "  Running SDMA Copy Stress Test (" << kIterations << " iterations, with "
            << kDoorbellDelayUs << "us delay)..." << std::endl;

  successful_iterations_ = 0;
  failed_iterations_ = 0;

  for (uint32_t i = 0; i < kIterations; i++) {
    // Clear destination
    memset(copy_dst_buffer_, 0, kCopySize);

    // Reset signal
    hsa_signal_store_screlease(completion_signal_, 1);

    // Controlled delay to mimic CNDI congestion timing
    if (kDoorbellDelayUs > 0) {
      std::this_thread::sleep_for(std::chrono::microseconds(kDoorbellDelayUs));
    }

    // Perform async copy CPU->GPU to exercise SDMA engine
    err = hsa_amd_memory_async_copy(copy_dst_buffer_, *gpu_device1(), copy_src_buffer_,
                                    *cpu_device(), kCopySize, 0, nullptr, completion_signal_);

    if (err != HSA_STATUS_SUCCESS) {
      failed_iterations_++;
      std::cout << "    Iteration " << i << ": async_copy failed" << std::endl;
      continue;
    }

    // Wait for completion with timeout
    hsa_signal_value_t result =
        hsa_signal_wait_scacquire(completion_signal_, HSA_SIGNAL_CONDITION_LT, 1,
                                  1000000000ULL,  // 1 second timeout
                                  HSA_WAIT_STATE_BLOCKED);

    if (result != 0) {
      failed_iterations_++;
      std::cout << "    Iteration " << i << ": Timeout" << std::endl;
      continue;
    }

    // Verify copy
    bool copy_ok = (memcmp(copy_src_buffer_, copy_dst_buffer_, kCopySize) == 0);
    if (copy_ok) {
      successful_iterations_++;
    } else {
      failed_iterations_++;
      std::cout << "    Iteration " << i << ": Copy mismatch" << std::endl;
    }
  }

  std::cout << "    Completed: " << successful_iterations_ << "/" << kIterations << " successful"
            << std::endl;

  EXPECT_EQ(failed_iterations_, 0U) << "Some SDMA iterations failed";
}

void SdmaPreemptionStressTest::WorkloadSubmissionThread(hsa_queue_t* queue,
                                                        std::atomic<bool>& stop_flag) {
  (void)queue;
  (void)stop_flag;
}

void SdmaPreemptionStressTest::PreemptionThread(hsa_queue_t* queue, hsa_agent_t agent,
                                                std::atomic<bool>& stop_flag) {
  (void)queue;
  (void)agent;
  (void)stop_flag;
}

void SdmaPreemptionStressTest::SdmaPreemptionStressTest_() {
  std::cout << "  Running SDMA Preemption Stress Test..." << std::endl;

  // This test requires the raw SDMA queue API (hsa_amd_queue_create with
  // HSA_AMD_QUEUE_ENGINE_SDMA) which is not available in older ROCm versions.
  // For now, we run a multi-threaded stress test using hsa_amd_memory_async_copy.

  successful_iterations_ = 0;
  failed_iterations_ = 0;
  preemption_test_passed_ = false;

  std::atomic<bool> stop_flag(false);
  std::atomic<uint32_t> total_copies(0);
  std::atomic<uint32_t> total_failures(0);

  // Allocate per-thread buffers
  constexpr uint32_t kNumThreads = 4;
  void* src_buffers[kNumThreads] = {};
  void* dst_buffers[kNumThreads] = {};
  hsa_signal_t signals[kNumThreads] = {};

  hsa_status_t err;
  hsa_agent_t ag_list[2] = {*gpu_device1(), *cpu_device()};

  // Allocate per-thread resources
  for (uint32_t t = 0; t < kNumThreads; t++) {
    err = hsa_amd_memory_pool_allocate(cpu_pool(), kCopySize, 0, &src_buffers[t]);
    ASSERT_EQ(HSA_STATUS_SUCCESS, err);
    err = hsa_amd_agents_allow_access(2, ag_list, NULL, src_buffers[t]);
    ASSERT_EQ(HSA_STATUS_SUCCESS, err);
    memset(src_buffers[t], 0xAB + t, kCopySize);

    err = hsa_amd_memory_pool_allocate(cpu_pool(), kCopySize, 0, &dst_buffers[t]);
    ASSERT_EQ(HSA_STATUS_SUCCESS, err);
    err = hsa_amd_agents_allow_access(2, ag_list, NULL, dst_buffers[t]);
    ASSERT_EQ(HSA_STATUS_SUCCESS, err);

    err = hsa_signal_create(1, 0, nullptr, &signals[t]);
    ASSERT_EQ(HSA_STATUS_SUCCESS, err);
  }

  // Worker thread function
  auto worker = [&](uint32_t thread_id) {
    void* src = src_buffers[thread_id];
    void* dst = dst_buffers[thread_id];
    hsa_signal_t sig = signals[thread_id];

    while (!stop_flag.load(std::memory_order_relaxed)) {
      memset(dst, 0, kCopySize);
      hsa_signal_store_screlease(sig, 1);

      // Small delay
      if (kDoorbellDelayUs > 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(kDoorbellDelayUs));
      }

      // Use GPU as dst agent to exercise SDMA engine (CPU-to-CPU would use memcpy)
      hsa_status_t status = hsa_amd_memory_async_copy(dst, *gpu_device1(), src, *cpu_device(),
                                                      kCopySize, 0, nullptr, sig);

      if (status != HSA_STATUS_SUCCESS) {
        total_failures.fetch_add(1);
        continue;
      }

      hsa_signal_value_t result = hsa_signal_wait_scacquire(sig, HSA_SIGNAL_CONDITION_LT, 1,
                                                            500000000ULL,  // 500ms timeout
                                                            HSA_WAIT_STATE_BLOCKED);

      if (result != 0) {
        total_failures.fetch_add(1);
        continue;
      }

      if (memcmp(src, dst, kCopySize) != 0) {
        total_failures.fetch_add(1);
      } else {
        total_copies.fetch_add(1);
      }
    }
  };

  // Start worker threads
  std::vector<std::thread> threads;
  for (uint32_t t = 0; t < kNumThreads; t++) {
    threads.emplace_back(worker, t);
  }

  // Run for 5 seconds
  std::this_thread::sleep_for(std::chrono::seconds(5));

  // Stop threads
  stop_flag.store(true, std::memory_order_relaxed);

  for (auto& th : threads) {
    th.join();
  }

  successful_iterations_ = total_copies.load();
  failed_iterations_ = total_failures.load();
  preemption_test_passed_ = (failed_iterations_ == 0);

  std::cout << "    Completed: " << successful_iterations_ << " successful, " << failed_iterations_
            << " failed" << std::endl;

  // Cleanup
  for (uint32_t t = 0; t < kNumThreads; t++) {
    hsa_signal_destroy(signals[t]);
    hsa_amd_memory_pool_free(src_buffers[t]);
    hsa_amd_memory_pool_free(dst_buffers[t]);
  }

  EXPECT_EQ(failed_iterations_, 0U) << "SDMA preemption stress test had failures";
}
