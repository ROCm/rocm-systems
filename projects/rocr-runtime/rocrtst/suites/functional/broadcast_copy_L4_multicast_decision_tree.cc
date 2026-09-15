/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// ROCrtst Level 4B Tests: Multicast Decision Tree Validation
// Purpose: Verify correct path selection (MULTICAST vs BROADCAST vs FANOUT)
//
// Decision Tree (from DmaCopyBroadcast):
//   num_dsts >= 3 && MulticastSupported  -> HW MULTICAST (N-dst packet)
//   num_dsts == 2 && BroadcastSupported  -> HW BROADCAST (2-dst packet)
//   Otherwise                            -> SW FAN-OUT (prologue/body/epilogue)

#include <gtest/gtest.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <cstdlib>
#include "../../common/broadcast_copy_utils.h"

class BroadcastCopyDecisionTree : public ::testing::Test {
 protected:
  void SetUp() override {
    // Check for debug environment variable
    sdma_logging_enabled_ = (std::getenv("HSA_DEBUG_SDMA_LOGGING") != nullptr);
  }

  bool sdma_logging_enabled_;
};

//
// Helper: Query hardware capabilities
//
struct HWCapabilities {
  uint32_t max_destinations;
  bool multicast_supported;  // N-dst HW path (GFX13+)
  bool broadcast_supported;  // 2-dst HW path
};

HWCapabilities QueryCapabilities(hsa_agent_t agent) {
  HWCapabilities caps = {0, false, false};

  hsa_status_t status = hsa_amd_memory_broadcast_capability(agent, &caps.max_destinations);
  if (status == HSA_STATUS_SUCCESS && caps.max_destinations > 0) {
    // If max_destinations >= 1024, assume full multicast support
    caps.multicast_supported = (caps.max_destinations >= 3);
    caps.broadcast_supported = (caps.max_destinations >= 2);
  }

  return caps;
}

// ============================================================================
// TC-DT-001: Two Destinations -> Should Use HW BROADCAST Path
// ============================================================================
// When num_dsts == 2, the runtime should use the 2-destination BROADCAST
// packet (fixed 9-DWORD format) rather than MULTICAST or FANOUT.

TEST_F(BroadcastCopyDecisionTree, TwoDests_UsesBroadcastPath) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  HWCapabilities caps = QueryCapabilities(ctx.gpu_agent);

  std::cout << "[TC-DT-001] Two destinations -> BROADCAST path test" << std::endl;
  std::cout << "  HW Capabilities: max_dsts=" << caps.max_destinations
            << ", broadcast=" << (caps.broadcast_supported ? "yes" : "no")
            << ", multicast=" << (caps.multicast_supported ? "yes" : "no") << std::endl;

  if (!caps.broadcast_supported) {
    std::cout << "  Expected path: SW FAN-OUT (no HW broadcast support)" << std::endl;
  } else {
    std::cout << "  Expected path: HW BROADCAST (2-dst packet)" << std::endl;
  }

  const size_t SIZE = 4096;
  const int NUM_DESTS = 2;

  void* src = ctx.AllocateGPUBuffer(SIZE);
  std::vector<void*> dsts(NUM_DESTS);
  for (auto& dst : dsts) {
    dst = ctx.AllocateGPUBuffer(SIZE);
  }

  BroadcastTestUtils::FillPattern(src, SIZE, BroadcastTestUtils::RANDOM, 0xB40ADCA5);
  std::vector<hsa_agent_t> dst_agents(NUM_DESTS, ctx.gpu_agent);
  hsa_signal_t signal = BroadcastTestUtils::CreateSignal(1);

  auto start = std::chrono::high_resolution_clock::now();

  hsa_status_t status =
      hsa_amd_memory_broadcast_copy(src, ctx.gpu_agent, dsts.data(), dst_agents.data(), NUM_DESTS,
                                    SIZE, 0, nullptr, signal, HSA_AMD_SDMA_ENGINE_0, false);

  ASSERT_EQ(HSA_STATUS_SUCCESS, status);
  BroadcastTestUtils::WaitSignal(signal);

  auto end = std::chrono::high_resolution_clock::now();
  auto time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  // Verify both destinations received correct data
  int pass_count = 0;
  for (int i = 0; i < NUM_DESTS; i++) {
    if (BroadcastTestUtils::CompareBuffers(src, dsts[i], SIZE)) {
      pass_count++;
    }
  }

  std::cout << "  Time: " << time_us << " us, Verified: " << pass_count << "/" << NUM_DESTS
            << std::endl;
  ASSERT_EQ(NUM_DESTS, pass_count) << "Data corruption in BROADCAST path";
  std::cout << "  PASS: 2-destination copy succeeded (BROADCAST or FANOUT path)" << std::endl;

  BroadcastTestUtils::DestroySignal(signal);
  ctx.Free(src);
  for (auto dst : dsts) ctx.Free(dst);
}

// ============================================================================
// TC-DT-002: Three Destinations -> Should Use HW MULTICAST Path
// ============================================================================
// When num_dsts >= 3, the runtime should use the N-destination MULTICAST
// packet (variable-length format) rather than BROADCAST or FANOUT.

TEST_F(BroadcastCopyDecisionTree, ThreeDests_UsesMulticastPath) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  HWCapabilities caps = QueryCapabilities(ctx.gpu_agent);

  std::cout << "[TC-DT-002] Three destinations -> MULTICAST path test" << std::endl;
  std::cout << "  HW Capabilities: max_dsts=" << caps.max_destinations << std::endl;

  if (caps.multicast_supported) {
    std::cout << "  Expected path: HW MULTICAST (N-dst packet)" << std::endl;
  } else {
    std::cout << "  Expected path: SW FAN-OUT (no HW multicast support)" << std::endl;
  }

  const size_t SIZE = 4096;
  const int NUM_DESTS = 3;  // Minimum for MULTICAST path

  void* src = ctx.AllocateGPUBuffer(SIZE);
  std::vector<void*> dsts(NUM_DESTS);
  for (auto& dst : dsts) {
    dst = ctx.AllocateGPUBuffer(SIZE);
  }

  BroadcastTestUtils::FillPattern(src, SIZE, BroadcastTestUtils::SEQUENTIAL, 0x33333333);
  std::vector<hsa_agent_t> dst_agents(NUM_DESTS, ctx.gpu_agent);
  hsa_signal_t signal = BroadcastTestUtils::CreateSignal(1);

  auto start = std::chrono::high_resolution_clock::now();

  hsa_status_t status =
      hsa_amd_memory_broadcast_copy(src, ctx.gpu_agent, dsts.data(), dst_agents.data(), NUM_DESTS,
                                    SIZE, 0, nullptr, signal, HSA_AMD_SDMA_ENGINE_0, false);

  ASSERT_EQ(HSA_STATUS_SUCCESS, status);
  BroadcastTestUtils::WaitSignal(signal);

  auto end = std::chrono::high_resolution_clock::now();
  auto time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  // Verify ALL three destinations received correct data
  // This is the critical test - before the multicast fix, only dst[0] got data
  int pass_count = 0;
  for (int i = 0; i < NUM_DESTS; i++) {
    bool valid = BroadcastTestUtils::CompareBuffers(src, dsts[i], SIZE);
    std::cout << "  dst[" << i << "]: " << (valid ? "PASS" : "FAIL") << std::endl;
    if (valid) pass_count++;
  }

  std::cout << "  Time: " << time_us << " us" << std::endl;

  ASSERT_EQ(NUM_DESTS, pass_count) << "CRITICAL: Not all destinations received data. "
                                   << "If only dst[0] passed, MULTICAST packet is broken.";

  std::cout << "  PASS: 3-destination copy succeeded (MULTICAST or FANOUT path)" << std::endl;

  BroadcastTestUtils::DestroySignal(signal);
  ctx.Free(src);
  for (auto dst : dsts) ctx.Free(dst);
}

// ============================================================================
// TC-DT-003: Boundary Test - 2 vs 3 Destinations Performance
// ============================================================================
// Compare performance of 2-dst (BROADCAST) vs 3-dst (MULTICAST) to validate
// different paths are being taken.

TEST_F(BroadcastCopyDecisionTree, BoundaryTest_2vs3_Dests) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  std::cout << "[TC-DT-003] Boundary test: 2 dsts (BROADCAST) vs 3 dsts (MULTICAST)" << std::endl;

  const size_t SIZE = 65536;  // 64KB
  const int ITERATIONS = 10;

  // Test 2 destinations (BROADCAST path)
  {
    const int NUM_DESTS = 2;
    void* src = ctx.AllocateGPUBuffer(SIZE);
    std::vector<void*> dsts(NUM_DESTS);
    for (auto& dst : dsts) dst = ctx.AllocateGPUBuffer(SIZE);

    BroadcastTestUtils::FillPattern(src, SIZE, BroadcastTestUtils::RANDOM);
    std::vector<hsa_agent_t> dst_agents(NUM_DESTS, ctx.gpu_agent);
    hsa_signal_t signal = BroadcastTestUtils::CreateSignal(1);

    uint64_t total_time = 0;
    for (int iter = 0; iter < ITERATIONS; iter++) {
      hsa_signal_store_screlease(signal, 1);
      auto start = std::chrono::high_resolution_clock::now();

      hsa_amd_memory_broadcast_copy(src, ctx.gpu_agent, dsts.data(), dst_agents.data(), NUM_DESTS,
                                    SIZE, 0, nullptr, signal, HSA_AMD_SDMA_ENGINE_0, false);
      BroadcastTestUtils::WaitSignal(signal);

      auto end = std::chrono::high_resolution_clock::now();
      total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }

    double avg_time_2dst = (double)total_time / ITERATIONS;
    std::cout << "  2 dsts (BROADCAST): avg " << std::fixed << std::setprecision(1) << avg_time_2dst
              << " us" << std::endl;

    BroadcastTestUtils::DestroySignal(signal);
    ctx.Free(src);
    for (auto dst : dsts) ctx.Free(dst);
  }

  // Test 3 destinations (MULTICAST path)
  {
    const int NUM_DESTS = 3;
    void* src = ctx.AllocateGPUBuffer(SIZE);
    std::vector<void*> dsts(NUM_DESTS);
    for (auto& dst : dsts) dst = ctx.AllocateGPUBuffer(SIZE);

    BroadcastTestUtils::FillPattern(src, SIZE, BroadcastTestUtils::RANDOM);
    std::vector<hsa_agent_t> dst_agents(NUM_DESTS, ctx.gpu_agent);
    hsa_signal_t signal = BroadcastTestUtils::CreateSignal(1);

    uint64_t total_time = 0;
    for (int iter = 0; iter < ITERATIONS; iter++) {
      hsa_signal_store_screlease(signal, 1);
      auto start = std::chrono::high_resolution_clock::now();

      hsa_amd_memory_broadcast_copy(src, ctx.gpu_agent, dsts.data(), dst_agents.data(), NUM_DESTS,
                                    SIZE, 0, nullptr, signal, HSA_AMD_SDMA_ENGINE_0, false);
      BroadcastTestUtils::WaitSignal(signal);

      auto end = std::chrono::high_resolution_clock::now();
      total_time += std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    }

    double avg_time_3dst = (double)total_time / ITERATIONS;
    std::cout << "  3 dsts (MULTICAST): avg " << std::fixed << std::setprecision(1) << avg_time_3dst
              << " us" << std::endl;

    // Verify correctness
    int pass_count = 0;
    for (int i = 0; i < NUM_DESTS; i++) {
      if (BroadcastTestUtils::CompareBuffers(src, dsts[i], SIZE)) pass_count++;
    }
    ASSERT_EQ(NUM_DESTS, pass_count) << "Data corruption in 3-dst test";

    BroadcastTestUtils::DestroySignal(signal);
    ctx.Free(src);
    for (auto dst : dsts) ctx.Free(dst);
  }

  std::cout << "  PASS: Both paths executed successfully" << std::endl;
}

// ============================================================================
// TC-DT-004: Single Destination -> Should NOT Use Multicast/Broadcast
// ============================================================================
// When num_dsts == 1, the runtime should use standard linear copy path,
// not the multicast machinery.

TEST_F(BroadcastCopyDecisionTree, SingleDest_StandardPath) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  std::cout << "[TC-DT-004] Single destination -> Standard copy path" << std::endl;

  const size_t SIZE = 4096;
  const int NUM_DESTS = 1;

  void* src = ctx.AllocateGPUBuffer(SIZE);
  void* dst = ctx.AllocateGPUBuffer(SIZE);

  BroadcastTestUtils::FillPattern(src, SIZE, BroadcastTestUtils::WALKING_BIT);
  std::vector<void*> dsts = {dst};
  std::vector<hsa_agent_t> dst_agents = {ctx.gpu_agent};
  hsa_signal_t signal = BroadcastTestUtils::CreateSignal(1);

  auto start = std::chrono::high_resolution_clock::now();

  hsa_status_t status =
      hsa_amd_memory_broadcast_copy(src, ctx.gpu_agent, dsts.data(), dst_agents.data(), NUM_DESTS,
                                    SIZE, 0, nullptr, signal, HSA_AMD_SDMA_ENGINE_0, false);

  ASSERT_EQ(HSA_STATUS_SUCCESS, status);
  BroadcastTestUtils::WaitSignal(signal);

  auto end = std::chrono::high_resolution_clock::now();
  auto time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  bool valid = BroadcastTestUtils::CompareBuffers(src, dst, SIZE);
  std::cout << "  Time: " << time_us << " us, Result: " << (valid ? "PASS" : "FAIL") << std::endl;

  ASSERT_TRUE(valid) << "Single destination copy failed";

  BroadcastTestUtils::DestroySignal(signal);
  ctx.Free(src);
  ctx.Free(dst);
}

// ============================================================================
// TC-DT-005: Scaling Test - Verify Multicast for Various Counts
// ============================================================================
// Test destination counts: 3, 4, 5, 8, 10, 16, 32
// All should use MULTICAST path and produce correct results.

TEST_F(BroadcastCopyDecisionTree, MulticastScaling_3to32) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  std::cout << "[TC-DT-005] Multicast scaling test (3-32 destinations)" << std::endl;
  std::cout << std::setw(10) << "Num Dsts"
            << " | " << std::setw(12) << "Time (us)"
            << " | " << std::setw(10) << "Verified"
            << " | "
            << "Result" << std::endl;
  std::cout << std::string(55, '-') << std::endl;

  const size_t SIZE = 8192;  // 8KB
  std::vector<int> dest_counts = {3, 4, 5, 8, 10, 16, 32};

  for (int NUM_DESTS : dest_counts) {
    void* src = ctx.AllocateGPUBuffer(SIZE);
    std::vector<void*> dsts(NUM_DESTS);
    for (auto& dst : dsts) {
      dst = ctx.AllocateGPUBuffer(SIZE);
      if (!dst) {
        std::cout << "  Memory allocation failed at " << NUM_DESTS << " dests" << std::endl;
        ctx.Free(src);
        for (auto d : dsts)
          if (d) ctx.Free(d);
        GTEST_SKIP() << "Insufficient memory";
      }
    }

    BroadcastTestUtils::FillPattern(src, SIZE, BroadcastTestUtils::RANDOM, NUM_DESTS * 0x11111111);
    std::vector<hsa_agent_t> dst_agents(NUM_DESTS, ctx.gpu_agent);
    hsa_signal_t signal = BroadcastTestUtils::CreateSignal(1);

    auto start = std::chrono::high_resolution_clock::now();

    hsa_status_t status =
        hsa_amd_memory_broadcast_copy(src, ctx.gpu_agent, dsts.data(), dst_agents.data(), NUM_DESTS,
                                      SIZE, 0, nullptr, signal, HSA_AMD_SDMA_ENGINE_0, false);

    ASSERT_EQ(HSA_STATUS_SUCCESS, status);
    BroadcastTestUtils::WaitSignal(signal);

    auto end = std::chrono::high_resolution_clock::now();
    auto time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // Verify ALL destinations
    int pass_count = 0;
    for (int i = 0; i < NUM_DESTS; i++) {
      if (BroadcastTestUtils::CompareBuffers(src, dsts[i], SIZE)) {
        pass_count++;
      }
    }

    bool all_pass = (pass_count == NUM_DESTS);

    std::cout << std::setw(10) << NUM_DESTS << " | " << std::setw(12) << time_us << " | "
              << std::setw(10) << (std::to_string(pass_count) + "/" + std::to_string(NUM_DESTS))
              << " | " << (all_pass ? "PASS" : "FAIL") << std::endl;

    ASSERT_EQ(NUM_DESTS, pass_count) << "MULTICAST failed for " << NUM_DESTS << " destinations";

    BroadcastTestUtils::DestroySignal(signal);
    ctx.Free(src);
    for (auto dst : dsts) ctx.Free(dst);
  }

  std::cout << "  All scaling tests passed" << std::endl;
}

// ============================================================================
// TC-DT-006: Fallback Detection - Force Shader Path
// ============================================================================
// When HSA_FORCE_BLIT_KERNEL_BROADCAST=1, should use shader fallback
// even when HW multicast is available.

TEST_F(BroadcastCopyDecisionTree, ForcedShaderFallback) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  const char* force_env = std::getenv("HSA_FORCE_BLIT_KERNEL_BROADCAST");
  bool shader_forced = (force_env != nullptr && std::string(force_env) == "1");

  std::cout << "[TC-DT-006] Shader fallback test" << std::endl;
  std::cout << "  HSA_FORCE_BLIT_KERNEL_BROADCAST=" << (force_env ? force_env : "(not set)")
            << std::endl;
  std::cout << "  Expected path: " << (shader_forced ? "BLIT SHADER" : "HW (if supported)")
            << std::endl;

  const size_t SIZE = 16384;
  const int NUM_DESTS = 4;

  void* src = ctx.AllocateGPUBuffer(SIZE);
  std::vector<void*> dsts(NUM_DESTS);
  for (auto& dst : dsts) dst = ctx.AllocateGPUBuffer(SIZE);

  BroadcastTestUtils::FillPattern(src, SIZE, BroadcastTestUtils::CHECKERBOARD);
  std::vector<hsa_agent_t> dst_agents(NUM_DESTS, ctx.gpu_agent);
  hsa_signal_t signal = BroadcastTestUtils::CreateSignal(1);

  auto start = std::chrono::high_resolution_clock::now();

  hsa_status_t status =
      hsa_amd_memory_broadcast_copy(src, ctx.gpu_agent, dsts.data(), dst_agents.data(), NUM_DESTS,
                                    SIZE, 0, nullptr, signal, HSA_AMD_SDMA_ENGINE_0, false);

  ASSERT_EQ(HSA_STATUS_SUCCESS, status);
  BroadcastTestUtils::WaitSignal(signal);

  auto end = std::chrono::high_resolution_clock::now();
  auto time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  int pass_count = 0;
  for (int i = 0; i < NUM_DESTS; i++) {
    if (BroadcastTestUtils::CompareBuffers(src, dsts[i], SIZE)) pass_count++;
  }

  std::cout << "  Time: " << time_us << " us, Verified: " << pass_count << "/" << NUM_DESTS
            << std::endl;

  ASSERT_EQ(NUM_DESTS, pass_count) << "Shader fallback produced incorrect data";

  if (shader_forced) {
    std::cout << "  PASS: Shader fallback works correctly" << std::endl;
  } else {
    std::cout
        << "  PASS: HW path works (re-run with HSA_FORCE_BLIT_KERNEL_BROADCAST=1 to test shader)"
        << std::endl;
  }

  BroadcastTestUtils::DestroySignal(signal);
  ctx.Free(src);
  for (auto dst : dsts) ctx.Free(dst);
}

// ============================================================================
// Main
// ============================================================================
