// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// ROCrtst Level 4 Tests: Small Multi-Destination (2-10)
// Purpose: Scale from 2-10 destinations, validate multi-dest logic

#include <gtest/gtest.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include "../../common/broadcast_copy_utils.h"

class BroadcastCopyL4 : public ::testing::Test {
 protected:
  void SetUp() override {}
};

//
// TC-L4-001: Two Destinations (Minimal Multi-Dest)
//

TEST_F(BroadcastCopyL4, TwoDests_4KB) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  const size_t SIZE = 4096;
  const int NUM_DESTS = 2;

  void* src = ctx.AllocateGPUBuffer(SIZE);
  std::vector<void*> dsts(NUM_DESTS);
  for (auto& dst : dsts) {
    dst = ctx.AllocateGPUBuffer(SIZE);
    memset(dst, 0xFF, SIZE);  // Fill with garbage
  }

  BroadcastTestUtils::FillPattern(src, SIZE, BroadcastTestUtils::RANDOM);

  std::vector<hsa_agent_t> dst_agents(NUM_DESTS, ctx.gpu_agent);
  hsa_signal_t signal = BroadcastTestUtils::CreateSignal(1);

  std::cout << "[TC-L4-001] Broadcasting to " << NUM_DESTS << " destinations:" << std::endl;
  std::cout << "  src=" << src << std::endl;
  for (int i = 0; i < NUM_DESTS; i++) {
    std::cout << "  dst[" << i << "]=" << dsts[i] << std::endl;
  }

  auto start = std::chrono::high_resolution_clock::now();

  hsa_status_t status =
      hsa_amd_memory_broadcast_copy(src, ctx.gpu_agent, dsts.data(), dst_agents.data(), NUM_DESTS,
                                    SIZE, 0, nullptr, signal, HSA_AMD_SDMA_ENGINE_0, false);

  ASSERT_EQ(HSA_STATUS_SUCCESS, status);
  BroadcastTestUtils::WaitSignal(signal);

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  std::cout << "  Completed in " << duration.count() << " µs" << std::endl;

  // Verify all destinations
  int pass_count = 0;
  for (int i = 0; i < NUM_DESTS; i++) {
    bool valid = BroadcastTestUtils::CompareBuffers(src, dsts[i], SIZE);
    std::cout << "  dst[" << i << "]: " << (valid ? "✓ PASS" : "❌ FAIL") << std::endl;
    if (valid) pass_count++;
  }

  ASSERT_EQ(NUM_DESTS, pass_count) << "Some destinations have corrupted data";

  BroadcastTestUtils::DestroySignal(signal);
  ctx.Free(src);
  for (auto dst : dsts) ctx.Free(dst);
}

//
// TC-L4-002: Ten Destinations with Performance Baseline
//

TEST_F(BroadcastCopyL4, TenDests_1MB_Performance) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  const size_t SIZE = 1048576;  // 1MB
  const int NUM_DESTS = 10;

  void* src = ctx.AllocateGPUBuffer(SIZE);
  std::vector<void*> dsts(NUM_DESTS);
  for (auto& dst : dsts) {
    dst = ctx.AllocateGPUBuffer(SIZE);
  }

  BroadcastTestUtils::FillPattern(src, SIZE, BroadcastTestUtils::SEQUENTIAL);
  std::vector<hsa_agent_t> dst_agents(NUM_DESTS, ctx.gpu_agent);
  hsa_signal_t signal = BroadcastTestUtils::CreateSignal(1);

  std::cout << "[TC-L4-002] 10-destination broadcast (" << SIZE << " bytes):" << std::endl;

  // Broadcast copy
  auto start = std::chrono::high_resolution_clock::now();

  hsa_status_t status =
      hsa_amd_memory_broadcast_copy(src, ctx.gpu_agent, dsts.data(), dst_agents.data(), NUM_DESTS,
                                    SIZE, 0, nullptr, signal, HSA_AMD_SDMA_ENGINE_0, false);

  ASSERT_EQ(HSA_STATUS_SUCCESS, status);
  BroadcastTestUtils::WaitSignal(signal);

  auto end = std::chrono::high_resolution_clock::now();
  auto broadcast_time_us =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  // Baseline: sequential copies (for comparison)
  hsa_signal_store_screlease(signal, NUM_DESTS);
  start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < NUM_DESTS; i++) {
    hsa_amd_memory_async_copy(dsts[i], ctx.gpu_agent, src, ctx.gpu_agent, SIZE, 0, nullptr, signal);
  }
  BroadcastTestUtils::WaitSignal(signal, HSA_SIGNAL_CONDITION_EQ, 0);
  end = std::chrono::high_resolution_clock::now();
  auto sequential_time_us =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  double speedup = sequential_time_us > 0 ? (double)sequential_time_us / broadcast_time_us : 0.0;

  std::cout << "  Broadcast time:  " << broadcast_time_us << " µs" << std::endl;
  std::cout << "  Sequential time: " << sequential_time_us << " µs" << std::endl;
  std::cout << "  Speedup:         " << std::fixed << std::setprecision(2) << speedup << "x"
            << std::endl;

  // Verify correctness
  int pass_count = 0;
  for (int i = 0; i < NUM_DESTS; i++) {
    if (BroadcastTestUtils::CompareBuffers(src, dsts[i], SIZE)) {
      pass_count++;
    }
  }

  std::cout << "  Verified:        " << pass_count << "/" << NUM_DESTS << " destinations"
            << std::endl;
  ASSERT_EQ(NUM_DESTS, pass_count) << "Data corruption in some destinations";

  if (speedup >= 1.2) {
    std::cout << "  ✓ Performance acceptable (≥1.2x speedup)" << std::endl;
  } else {
    std::cout << "  ℹ️  Performance below target (expected ≥1.2x for 10 dests)" << std::endl;
  }

  BroadcastTestUtils::DestroySignal(signal);
  ctx.Free(src);
  for (auto dst : dsts) ctx.Free(dst);
}

//
// TC-L4-003: Progressive Destination Counts (2, 4, 6, 8, 10)
//

TEST_F(BroadcastCopyL4, ProgressiveDestCounts) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  const size_t SIZE = 65536;  // 64KB
  std::vector<int> dest_counts = {2, 4, 6, 8, 10};

  std::cout << "[TC-L4-003] Progressive destination counts:" << std::endl;
  std::cout << std::setw(10) << "Num Dests"
            << " | " << std::setw(12) << "Time (µs)"
            << " | "
            << "Result" << std::endl;
  std::cout << std::string(50, '-') << std::endl;

  for (int NUM_DESTS : dest_counts) {
    void* src = ctx.AllocateGPUBuffer(SIZE);
    std::vector<void*> dsts(NUM_DESTS);
    for (auto& dst : dsts) dst = ctx.AllocateGPUBuffer(SIZE);

    BroadcastTestUtils::FillPattern(src, SIZE, BroadcastTestUtils::RANDOM, 0x12345678 + NUM_DESTS);
    std::vector<hsa_agent_t> dst_agents(NUM_DESTS, ctx.gpu_agent);
    hsa_signal_t signal = BroadcastTestUtils::CreateSignal(1);

    auto start = std::chrono::high_resolution_clock::now();
    hsa_status_t status =
        hsa_amd_memory_broadcast_copy(src, ctx.gpu_agent, dsts.data(), dst_agents.data(), NUM_DESTS,
                                      SIZE, 0, nullptr, signal, HSA_AMD_SDMA_ENGINE_0, false);
    BroadcastTestUtils::WaitSignal(signal);
    auto end = std::chrono::high_resolution_clock::now();
    auto time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    int verified = 0;
    for (int i = 0; i < NUM_DESTS; i++) {
      if (BroadcastTestUtils::CompareBuffers(src, dsts[i], SIZE)) verified++;
    }

    bool pass = (status == HSA_STATUS_SUCCESS && verified == NUM_DESTS);

    std::cout << std::setw(10) << NUM_DESTS << " | " << std::setw(12) << time_us << " | "
              << (pass ? "✓ PASS" : "❌ FAIL") << " (" << verified << "/" << NUM_DESTS
              << " verified)" << std::endl;

    ASSERT_TRUE(pass);

    BroadcastTestUtils::DestroySignal(signal);
    ctx.Free(src);
    for (auto dst : dsts) ctx.Free(dst);
  }
}

//
// TC-L4-004: Multiple Sequential Broadcast Operations
//

TEST_F(BroadcastCopyL4, MultipleSequentialBroadcasts) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  const size_t SIZE = 16384;
  const int NUM_DESTS = 4;
  const int NUM_ITERATIONS = 10;

  std::cout << "[TC-L4-004] Multiple sequential broadcast operations:" << std::endl;
  std::cout << "  Iterations: " << NUM_ITERATIONS << ", Dests per iteration: " << NUM_DESTS
            << std::endl;

  int total_pass = 0;

  for (int iter = 0; iter < NUM_ITERATIONS; iter++) {
    void* src = ctx.AllocateGPUBuffer(SIZE);
    std::vector<void*> dsts(NUM_DESTS);
    for (auto& dst : dsts) dst = ctx.AllocateGPUBuffer(SIZE);

    BroadcastTestUtils::FillPattern(src, SIZE, BroadcastTestUtils::SEQUENTIAL, iter);
    std::vector<hsa_agent_t> dst_agents(NUM_DESTS, ctx.gpu_agent);
    hsa_signal_t signal = BroadcastTestUtils::CreateSignal(1);

    hsa_status_t status =
        hsa_amd_memory_broadcast_copy(src, ctx.gpu_agent, dsts.data(), dst_agents.data(), NUM_DESTS,
                                      SIZE, 0, nullptr, signal, HSA_AMD_SDMA_ENGINE_0, false);

    ASSERT_EQ(HSA_STATUS_SUCCESS, status);
    BroadcastTestUtils::WaitSignal(signal);

    bool all_valid = true;
    for (int i = 0; i < NUM_DESTS; i++) {
      if (!BroadcastTestUtils::VerifyPattern(dsts[i], SIZE, BroadcastTestUtils::SEQUENTIAL, iter)) {
        all_valid = false;
        break;
      }
    }

    if (all_valid) total_pass++;

    BroadcastTestUtils::DestroySignal(signal);
    ctx.Free(src);
    for (auto dst : dsts) ctx.Free(dst);
  }

  std::cout << "  Passed: " << total_pass << "/" << NUM_ITERATIONS << " iterations" << std::endl;
  ASSERT_EQ(NUM_ITERATIONS, total_pass) << "Some iterations failed";
  std::cout << "  ✓ All iterations passed" << std::endl;
}
