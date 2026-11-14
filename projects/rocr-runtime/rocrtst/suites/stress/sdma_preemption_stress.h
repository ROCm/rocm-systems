/*
 * Copyright (c) 2026, Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef ROCRTST_SUITES_STRESS_SDMA_PREEMPTION_STRESS_H_
#define ROCRTST_SUITES_STRESS_SDMA_PREEMPTION_STRESS_H_

#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include "suites/test_common/test_base.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

/**
 * @brief SDMA Preemption Stress Test
 *
 * This test reproduces SDMA queue preemption issues by:
 * 1. Performing rapid SDMA copy operations using hsa_amd_memory_async_copy
 * 2. Adding controlled delays between operations to mimic CNDI congestion
 * 3. Running multiple threads to stress SDMA queues concurrently
 *
 * Based on SDMA team recommended testing approach for exposing RPTR>WPTR anomaly.
 */
class SdmaPreemptionStressTest : public TestBase {
 public:
  SdmaPreemptionStressTest();
  virtual ~SdmaPreemptionStressTest();

  virtual void SetUp() override;
  virtual void Run() override;
  virtual void Close() override;
  virtual void DisplayResults() const override;
  virtual void DisplayTestInfo() override;

  /**
   * @brief Basic SDMA copy test using hsa_amd_memory_async_copy
   * Performs a single copy and verifies completion
   */
  void BasicSdmaPacketTest();

  /**
   * @brief SDMA stress test with periodic copy operations
   * Performs many copies with controlled timing delays
   */
  void SdmaDoorbellStressTest();

  /**
   * @brief SDMA stress test with multiple threads
   * Multiple threads perform concurrent copies to stress SDMA subsystem
   */
  void SdmaPreemptionStressTest_();

 private:
  // Test configuration
  static constexpr uint32_t kSdmaQueueSizeBytes = 4096;
  static constexpr uint32_t kCopySize = 4096;  // 4KB copy size
  static constexpr uint32_t kIterations = 100;
  static constexpr uint32_t kDoorbellDelayUs = 10;   // Delay in microseconds
  static constexpr uint32_t kPreemptionDelayMs = 5;  // Preemption delay in milliseconds

  // Helper functions (stubs for raw packet API - not used in current version)
  hsa_status_t CreateSdmaQueue(hsa_agent_t agent, hsa_queue_t** queue);
  void DestroySdmaQueue(hsa_queue_t* queue);

  // Get queue ring buffer pointer and write index
  uint32_t* GetQueueRingBuffer(hsa_queue_t* queue);
  uint64_t GetQueueWritePtr(hsa_queue_t* queue);
  void SetQueueWritePtr(hsa_queue_t* queue, uint64_t write_ptr);
  void RingDoorbell(hsa_queue_t* queue, uint64_t write_ptr_bytes);

  // Build SDMA packets (stubs - not used in current version)
  void BuildPollRegMemPacket(uint32_t* pkt, uint64_t addr, uint32_t value, uint32_t mask);
  void BuildGcrUserPacket(uint32_t* pkt, uint32_t gcr_control);
  void BuildCopyLinearPacket(uint32_t* pkt, uint64_t dst, uint64_t src, uint32_t size);
  void BuildAtomicPacket(uint32_t* pkt, uint64_t addr, uint64_t src_data, uint64_t cmp_data);
  void BuildFencePacket(uint32_t* pkt, uint64_t addr, uint32_t data);
  void BuildTrapPacket(uint32_t* pkt, uint32_t event_id);

  // Submit packet sequence to SDMA queue (stub - not used in current version)
  void SubmitSdmaPacketSequence(hsa_queue_t* queue, uint64_t poll_addr, uint64_t copy_src,
                                uint64_t copy_dst, uint64_t atomic_addr, uint64_t fence_addr,
                                uint32_t fence_value);

  // Thread functions
  void WorkloadSubmissionThread(hsa_queue_t* queue, std::atomic<bool>& stop_flag);
  void PreemptionThread(hsa_queue_t* queue, hsa_agent_t agent, std::atomic<bool>& stop_flag);

  // Test state
  hsa_queue_t* sdma_queue_;
  void* copy_src_buffer_;
  void* copy_dst_buffer_;
  hsa_signal_t completion_signal_;

  // Results
  uint32_t successful_iterations_;
  uint32_t failed_iterations_;
  bool preemption_test_passed_;
};

#endif  // ROCRTST_SUITES_STRESS_SDMA_PREEMPTION_STRESS_H_
