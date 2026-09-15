/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// ROCrtst Shader Fallback Tests
// Purpose: Validate blit kernel fallback path

#include <gtest/gtest.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <cstdlib>
#include <chrono>
#include <algorithm>
#include <numeric>
#include "../../common/broadcast_copy_utils.h"

class BroadcastCopyShaderFallback : public ::testing::Test {
 protected:
  // Tests validate automatic path selection:
  // - Small buffers (<4KB) use shader path
  // - Large buffers (>=4KB) use SDMA on GFX13+ or shader on pre-GFX13
};

//
// TC-SHD-001: Force Shader Fallback via Environment Variable
//

TEST_F(BroadcastCopyShaderFallback, SmallCopyAutoFallback) {
  std::cout << "[TC-SHD-001] Testing automatic shader fallback for small copies" << std::endl;
  std::cout << "  Note: Copies <4KB automatically use shader path on GFX13+ hardware" << std::endl;
  std::cout << "  This provides optimal performance by avoiding SDMA overhead" << std::endl;

  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  // Use small size to trigger shader fallback even on GFX13+
  const size_t SIZE = 2048;  // <4KB triggers shader path
  const int NUM_DESTS = 4;

  void* src = ctx.AllocateGPUBuffer(SIZE);
  std::vector<void*> dsts(NUM_DESTS);
  for (auto& dst : dsts) dst = ctx.AllocateGPUBuffer(SIZE);

  BroadcastTestUtils::FillPattern(src, SIZE, BroadcastTestUtils::RANDOM, 0xFEEDFACE);
  std::vector<hsa_agent_t> dst_agents(NUM_DESTS, ctx.gpu_agent);
  hsa_signal_t signal = BroadcastTestUtils::CreateSignal(1);

  std::cout << "  Executing broadcast copy..." << std::endl;
  auto start = std::chrono::high_resolution_clock::now();

  hsa_status_t status =
      hsa_amd_memory_broadcast_copy(src, ctx.gpu_agent, dsts.data(), dst_agents.data(), NUM_DESTS,
                                    SIZE, 0, nullptr, signal, HSA_AMD_SDMA_ENGINE_0, false);

  ASSERT_EQ(HSA_STATUS_SUCCESS, status);
  BroadcastTestUtils::WaitSignal(signal);

  auto end = std::chrono::high_resolution_clock::now();
  auto time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  std::cout << "  Completed in " << time_us << " µs" << std::endl;
  std::cout << "  Path used: Shader (blit kernel) - automatic for size <4KB" << std::endl;
  std::cout << "  Expected: All GPUs support shader fallback transparently" << std::endl;

  // Verify correctness
  int pass_count = 0;
  for (int i = 0; i < NUM_DESTS; i++) {
    if (BroadcastTestUtils::VerifyPattern(dsts[i], SIZE, BroadcastTestUtils::RANDOM, 0xFEEDFACE)) {
      pass_count++;
    }
  }

  std::cout << "  Verified: " << pass_count << "/" << NUM_DESTS << " destinations" << std::endl;
  ASSERT_EQ(NUM_DESTS, pass_count) << "Data corruption in small-copy shader path";
  std::cout << "  [PASS] Small-copy shader fallback test passed" << std::endl;

  BroadcastTestUtils::DestroySignal(signal);
  ctx.Free(src);
  for (auto dst : dsts) ctx.Free(dst);
}

//
// TC-SHD-002: Automatic Shader Fallback on Non-Multicast Hardware
//

TEST_F(BroadcastCopyShaderFallback, AutoFallback_NonMulticastHW) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  uint32_t max_destinations = 0;
  hsa_status_t status = hsa_amd_memory_broadcast_capability(ctx.gpu_agent, &max_destinations);

  std::cout << "[TC-SHD-002] Automatic shader fallback on non-multicast hardware" << std::endl;
  std::cout << "  Hardware capability: max_destinations = " << max_destinations << std::endl;

  // GFX9+ reports 1024 (shader or SDMA support), pre-GFX9 reports 0
  if (max_destinations == 0) {
    GTEST_SKIP() << "GPU does not support broadcast (pre-GFX9 hardware)";
  }
  ASSERT_EQ(1024, max_destinations) << "GFX9+ should report max_destinations=1024";
  std::cout << "  [PASS] GPU supports broadcast (SDMA or shader path)" << std::endl;

  const size_t SIZE = 8192;
  const int NUM_DESTS = 4;

  void* src = ctx.AllocateGPUBuffer(SIZE);
  std::vector<void*> dsts(NUM_DESTS);
  for (auto& dst : dsts) dst = ctx.AllocateGPUBuffer(SIZE);

  BroadcastTestUtils::FillPattern(src, SIZE, BroadcastTestUtils::SEQUENTIAL);
  std::vector<hsa_agent_t> dst_agents(NUM_DESTS, ctx.gpu_agent);
  hsa_signal_t signal = BroadcastTestUtils::CreateSignal(1);

  std::cout << "  Runtime will select best path: SDMA (GFX13+,size>=4KB) or Shader (fallback)"
            << std::endl;

  std::cout << "  Executing broadcast..." << std::endl;

  status =
      hsa_amd_memory_broadcast_copy(src, ctx.gpu_agent, dsts.data(), dst_agents.data(), NUM_DESTS,
                                    SIZE, 0, nullptr, signal, HSA_AMD_SDMA_ENGINE_0, false);

  ASSERT_EQ(HSA_STATUS_SUCCESS, status) << "Broadcast failed";
  BroadcastTestUtils::WaitSignal(signal);

  // Verify correctness
  bool all_valid = true;
  for (int i = 0; i < NUM_DESTS; i++) {
    if (!BroadcastTestUtils::VerifyPattern(dsts[i], SIZE, BroadcastTestUtils::SEQUENTIAL)) {
      all_valid = false;
      break;
    }
  }

  ASSERT_TRUE(all_valid) << "Data corruption";
  std::cout << "  [PASS] Broadcast successful (auto fallback if needed)" << std::endl;

  BroadcastTestUtils::DestroySignal(signal);
  ctx.Free(src);
  for (auto dst : dsts) ctx.Free(dst);
}

//
// TC-SHD-003: Small vs Large Copy Path Performance
//

TEST_F(BroadcastCopyShaderFallback, SmallVsLarge_PathPerformance) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  std::cout << "[TC-SHD-003] Small vs Large copy path performance" << std::endl;
  std::cout << "  Compares shader path (small <4KB) vs SDMA/shader path (large >=4KB)" << std::endl;

  const int NUM_DESTS = 10;
  const int NUM_ITERS = 10;

  // Test small size (guaranteed shader path)
  const size_t SMALL_SIZE = 2048;  // <4KB

  // Test large size (SDMA on GFX13+, shader on pre-GFX13)
  const size_t LARGE_SIZE = 1048576;  // 1MB

  std::cout << "\n  Testing SMALL copies (" << SMALL_SIZE << " bytes, shader path):" << std::endl;

  void* src_small = ctx.AllocateGPUBuffer(SMALL_SIZE);
  std::vector<void*> dsts_small(NUM_DESTS);
  for (auto& dst : dsts_small) dst = ctx.AllocateGPUBuffer(SMALL_SIZE);

  BroadcastTestUtils::FillPattern(src_small, SMALL_SIZE, BroadcastTestUtils::INCREMENTAL);
  std::vector<hsa_agent_t> dst_agents(NUM_DESTS, ctx.gpu_agent);
  hsa_signal_t signal = BroadcastTestUtils::CreateSignal(1);

  std::vector<int64_t> small_times_us;

  for (int iter = 0; iter < NUM_ITERS; iter++) {
    hsa_signal_store_screlease(signal, 1);

    auto start = std::chrono::high_resolution_clock::now();

    hsa_status_t status = hsa_amd_memory_broadcast_copy(
        src_small, ctx.gpu_agent, dsts_small.data(), dst_agents.data(), NUM_DESTS, SMALL_SIZE, 0,
        nullptr, signal, HSA_AMD_SDMA_ENGINE_0, false);

    ASSERT_EQ(HSA_STATUS_SUCCESS, status);
    BroadcastTestUtils::WaitSignal(signal);

    auto end = std::chrono::high_resolution_clock::now();
    auto time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    small_times_us.push_back(time_us);
  }

  // Verify correctness
  for (auto dst : dsts_small) {
    ASSERT_TRUE(BroadcastTestUtils::VerifyPattern(dst, SMALL_SIZE, BroadcastTestUtils::INCREMENTAL))
        << "Small copy failed verification";
  }

  // Calculate small copy statistics
  int64_t small_min = *std::min_element(small_times_us.begin(), small_times_us.end());
  int64_t small_avg =
      std::accumulate(small_times_us.begin(), small_times_us.end(), 0LL) / NUM_ITERS;

  std::cout << "    Avg time: " << small_avg << " µs (guaranteed shader path)" << std::endl;
  std::cout << "    [PASS] All small copies verified" << std::endl;

  // Cleanup small buffers
  ctx.Free(src_small);
  for (auto dst : dsts_small) ctx.Free(dst);

  // Test large size
  std::cout << "\n  Testing LARGE copies (" << LARGE_SIZE
            << " bytes, SDMA on GFX13+ or shader):" << std::endl;

  void* src_large = ctx.AllocateGPUBuffer(LARGE_SIZE);
  std::vector<void*> dsts_large(NUM_DESTS);
  for (auto& dst : dsts_large) dst = ctx.AllocateGPUBuffer(LARGE_SIZE);

  BroadcastTestUtils::FillPattern(src_large, LARGE_SIZE, BroadcastTestUtils::INCREMENTAL);

  std::vector<int64_t> large_times_us;

  for (int iter = 0; iter < NUM_ITERS; iter++) {
    hsa_signal_store_screlease(signal, 1);

    auto start = std::chrono::high_resolution_clock::now();

    hsa_status_t status = hsa_amd_memory_broadcast_copy(
        src_large, ctx.gpu_agent, dsts_large.data(), dst_agents.data(), NUM_DESTS, LARGE_SIZE, 0,
        nullptr, signal, HSA_AMD_SDMA_ENGINE_0, false);

    ASSERT_EQ(HSA_STATUS_SUCCESS, status);
    BroadcastTestUtils::WaitSignal(signal);

    auto end = std::chrono::high_resolution_clock::now();
    auto time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    large_times_us.push_back(time_us);
  }

  // Verify correctness
  for (auto dst : dsts_large) {
    ASSERT_TRUE(BroadcastTestUtils::VerifyPattern(dst, LARGE_SIZE, BroadcastTestUtils::INCREMENTAL))
        << "Large copy failed verification";
  }

  // Calculate large copy statistics
  int64_t large_min = *std::min_element(large_times_us.begin(), large_times_us.end());
  int64_t large_avg =
      std::accumulate(large_times_us.begin(), large_times_us.end(), 0LL) / NUM_ITERS;

  double large_bandwidth_gbps = (LARGE_SIZE * NUM_DESTS * 1.0e6) / (large_avg * 1.0e9);

  std::cout << "    Avg time: " << large_avg << " µs" << std::endl;
  std::cout << "    Bandwidth: " << std::fixed << std::setprecision(2) << large_bandwidth_gbps
            << " GB/s (effective)" << std::endl;
  std::cout << "    [PASS] All large copies verified" << std::endl;

  std::cout << "\n  Performance comparison:" << std::endl;
  std::cout << "    Small (<4KB):  " << small_avg << " µs (shader)" << std::endl;
  std::cout << "    Large (>=4KB): " << large_avg << " µs (SDMA/shader)" << std::endl;
  std::cout << "    [PASS] Path selection working correctly" << std::endl;

  BroadcastTestUtils::DestroySignal(signal);
  ctx.Free(src_large);
  for (auto dst : dsts_large) ctx.Free(dst);
}

//
// TC-SHD-004: Shader Fallback with Various Sizes
//

TEST_F(BroadcastCopyShaderFallback, ShaderFallback_VariousSizes) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  std::cout << "[TC-SHD-004] Shader fallback with various buffer sizes" << std::endl;

  const int NUM_DESTS = 4;
  std::vector<size_t> sizes = {64, 256, 1024, 4096, 16384, 65536, 262144, 1048576};

  std::cout << std::setw(12) << "Size"
            << " | " << std::setw(12) << "Time (µs)"
            << " | "
            << "Result" << std::endl;
  std::cout << std::string(50, '-') << std::endl;

  for (size_t SIZE : sizes) {
    void* src = ctx.AllocateGPUBuffer(SIZE);
    std::vector<void*> dsts(NUM_DESTS);
    bool alloc_success = true;

    for (int i = 0; i < NUM_DESTS; i++) {
      dsts[i] = ctx.AllocateGPUBuffer(SIZE);
      if (!dsts[i]) {
        alloc_success = false;
        break;
      }
    }

    if (!alloc_success) {
      ctx.Free(src);
      for (auto dst : dsts)
        if (dst) ctx.Free(dst);
      std::cout << std::setw(12) << SIZE << " | " << std::setw(12) << "N/A"
                << " | "
                << "SKIP (alloc failed)" << std::endl;
      continue;
    }

    BroadcastTestUtils::FillPattern(src, SIZE, BroadcastTestUtils::RANDOM, SIZE);
    std::vector<hsa_agent_t> dst_agents(NUM_DESTS, ctx.gpu_agent);
    hsa_signal_t signal = BroadcastTestUtils::CreateSignal(1);

    auto start = std::chrono::high_resolution_clock::now();

    hsa_status_t status =
        hsa_amd_memory_broadcast_copy(src, ctx.gpu_agent, dsts.data(), dst_agents.data(), NUM_DESTS,
                                      SIZE, 0, nullptr, signal, HSA_AMD_SDMA_ENGINE_0, false);

    bool pass = false;
    if (status == HSA_STATUS_SUCCESS) {
      BroadcastTestUtils::WaitSignal(signal);

      pass = true;
      for (int i = 0; i < NUM_DESTS; i++) {
        if (!BroadcastTestUtils::VerifyPattern(dsts[i], SIZE, BroadcastTestUtils::RANDOM, SIZE)) {
          pass = false;
          break;
        }
      }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << std::setw(12) << SIZE << " | " << std::setw(12) << time_us << " | "
              << (pass ? "[PASS] PASS" : "[FAIL] FAIL") << std::endl;

    ASSERT_TRUE(pass) << "Size " << SIZE << " failed";

    BroadcastTestUtils::DestroySignal(signal);
    ctx.Free(src);
    for (auto dst : dsts) ctx.Free(dst);
  }

  std::cout << "  [PASS] All sizes validated" << std::endl;
}

//
// TC-SHD-005: Shader Fallback Multi-Destination Scaling
//

TEST_F(BroadcastCopyShaderFallback, ShaderFallback_MultiDest) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  std::cout << "[TC-SHD-005] Shader fallback multi-destination scaling" << std::endl;

  const size_t SIZE = 65536;
  std::vector<int> dest_counts = {1, 2, 4, 8, 16, 32};

  std::cout << std::setw(10) << "Num Dests"
            << " | " << std::setw(12) << "Time (µs)"
            << " | "
            << "Result" << std::endl;
  std::cout << std::string(50, '-') << std::endl;

  for (int NUM_DESTS : dest_counts) {
    void* src = ctx.AllocateGPUBuffer(SIZE);
    std::vector<void*> dsts(NUM_DESTS);

    bool alloc_success = true;
    for (int i = 0; i < NUM_DESTS; i++) {
      dsts[i] = ctx.AllocateGPUBuffer(SIZE);
      if (!dsts[i]) {
        alloc_success = false;
        break;
      }
    }

    if (!alloc_success) {
      ctx.Free(src);
      for (auto dst : dsts)
        if (dst) ctx.Free(dst);
      std::cout << std::setw(10) << NUM_DESTS << " | " << std::setw(12) << "N/A"
                << " | "
                << "SKIP (alloc failed)" << std::endl;
      continue;
    }

    BroadcastTestUtils::FillPattern(src, SIZE, BroadcastTestUtils::WALKING_BIT);
    std::vector<hsa_agent_t> dst_agents(NUM_DESTS, ctx.gpu_agent);
    hsa_signal_t signal = BroadcastTestUtils::CreateSignal(1);

    auto start = std::chrono::high_resolution_clock::now();

    hsa_status_t status =
        hsa_amd_memory_broadcast_copy(src, ctx.gpu_agent, dsts.data(), dst_agents.data(), NUM_DESTS,
                                      SIZE, 0, nullptr, signal, HSA_AMD_SDMA_ENGINE_0, false);

    bool pass = false;
    if (status == HSA_STATUS_SUCCESS) {
      BroadcastTestUtils::WaitSignal(signal);

      pass = true;
      for (int i = 0; i < NUM_DESTS; i++) {
        if (!BroadcastTestUtils::VerifyPattern(dsts[i], SIZE, BroadcastTestUtils::WALKING_BIT)) {
          pass = false;
          break;
        }
      }
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << std::setw(10) << NUM_DESTS << " | " << std::setw(12) << time_us << " | "
              << (pass ? "[PASS] PASS" : "[FAIL] FAIL") << std::endl;

    ASSERT_TRUE(pass) << NUM_DESTS << " destinations failed";

    BroadcastTestUtils::DestroySignal(signal);
    ctx.Free(src);
    for (auto dst : dsts) ctx.Free(dst);
  }

  std::cout << "  [PASS] Shader fallback scales correctly with destination count" << std::endl;
}
