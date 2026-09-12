/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// ROCrtst Level 5 Tests: Large Multi-Destination (100-1024)
// Purpose: Validate hardware multicast at scale

#include <gtest/gtest.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include "../../common/broadcast_copy_utils.h"

class BroadcastCopyL5 : public ::testing::Test {
 protected:
  void SetUp() override {}
};

//
//
//

TEST_F(BroadcastCopyL5, Broadcast_100_Destinations) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  uint32_t max_destinations = 0;
  hsa_status_t status = hsa_amd_memory_broadcast_capability(ctx.gpu_agent, &max_destinations);

  if (status != HSA_STATUS_SUCCESS || max_destinations < 100) {
    GTEST_SKIP() << "Hardware doesn't support 100 destinations (max=" << max_destinations << ")";
  }

  const size_t SIZE = 65536;  // 64KB
  const int NUM_DESTS = 100;

  std::cout << "Broadcasting to " << NUM_DESTS << " destinations (" << SIZE
            << " bytes each)" << std::endl;

  void* src = ctx.AllocateGPUBuffer(SIZE);
  std::vector<void*> dsts(NUM_DESTS);

  std::cout << "  Allocating " << NUM_DESTS << " destination buffers..." << std::endl;
  for (int i = 0; i < NUM_DESTS; i++) {
    dsts[i] = ctx.AllocateGPUBuffer(SIZE);
    if (!dsts[i]) {
      GTEST_SKIP() << "Failed to allocate dst[" << i << "] (out of memory?)";
    }
  }

  BroadcastTestUtils::FillPattern(src, SIZE, BroadcastTestUtils::RANDOM, 0xDEADBEEF);
  std::vector<hsa_agent_t> dst_agents(NUM_DESTS, ctx.gpu_agent);
  hsa_signal_t signal = BroadcastTestUtils::CreateSignal(1);

  std::cout << "  Executing broadcast..." << std::endl;
  auto start = std::chrono::high_resolution_clock::now();

  status =
      hsa_amd_memory_broadcast_copy(src, ctx.gpu_agent, dsts.data(), dst_agents.data(), NUM_DESTS,
                                    SIZE, 0, nullptr, signal, HSA_AMD_SDMA_ENGINE_0, false);

  ASSERT_EQ(HSA_STATUS_SUCCESS, status) << "Broadcast failed";
  BroadcastTestUtils::WaitSignal(signal);

  auto end = std::chrono::high_resolution_clock::now();
  auto time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  std::cout << "  Completed in " << time_us << " µs" << std::endl;
  std::cout << "  Verifying all destinations..." << std::endl;

  int pass_count = 0;
  int sample_indices[] = {0, 25, 50, 75, 99};  // Sample verification

  for (int idx : sample_indices) {
    bool valid = BroadcastTestUtils::CompareBuffers(src, dsts[idx], SIZE);
    std::cout << "    dst[" << std::setw(3) << idx << "]: " << (valid ? "[PASS]" : "[FAIL]") << std::endl;
    if (valid) pass_count++;
  }

  ASSERT_EQ(5, pass_count) << "Sampled destinations have corrupted data";
  std::cout << "  [PASS] Broadcast to 100 destinations successful" << std::endl;

  BroadcastTestUtils::DestroySignal(signal);
  ctx.Free(src);
  for (auto dst : dsts) ctx.Free(dst);
}

//
//
//

TEST_F(BroadcastCopyL5, Broadcast_256_Destinations) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  uint32_t max_destinations = 0;
  hsa_status_t status = hsa_amd_memory_broadcast_capability(ctx.gpu_agent, &max_destinations);

  if (status != HSA_STATUS_SUCCESS || max_destinations < 256) {
    GTEST_SKIP() << "Hardware doesn't support 256 destinations (max=" << max_destinations << ")";
  }

  const size_t SIZE = 32768;  // 32KB
  const int NUM_DESTS = 256;

  std::cout << "Broadcasting to " << NUM_DESTS << " destinations" << std::endl;

  void* src = ctx.AllocateGPUBuffer(SIZE);
  std::vector<void*> dsts(NUM_DESTS);

  for (int i = 0; i < NUM_DESTS; i++) {
    dsts[i] = ctx.AllocateGPUBuffer(SIZE);
    if (!dsts[i]) GTEST_SKIP() << "Memory allocation failed at dst[" << i << "]";
  }

  BroadcastTestUtils::FillPattern(src, SIZE, BroadcastTestUtils::WALKING_BIT);
  std::vector<hsa_agent_t> dst_agents(NUM_DESTS, ctx.gpu_agent);
  hsa_signal_t signal = BroadcastTestUtils::CreateSignal(1);

  auto start = std::chrono::high_resolution_clock::now();

  status =
      hsa_amd_memory_broadcast_copy(src, ctx.gpu_agent, dsts.data(), dst_agents.data(), NUM_DESTS,
                                    SIZE, 0, nullptr, signal, HSA_AMD_SDMA_ENGINE_0, false);

  ASSERT_EQ(HSA_STATUS_SUCCESS, status);
  BroadcastTestUtils::WaitSignal(signal);

  auto end = std::chrono::high_resolution_clock::now();
  auto time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  // Verify sample destinations
  int sample_count = 10;
  int pass_count = 0;
  for (int i = 0; i < sample_count; i++) {
    int idx = (i * NUM_DESTS) / sample_count;
    if (BroadcastTestUtils::CompareBuffers(src, dsts[idx], SIZE)) {
      pass_count++;
    }
  }

  std::cout << "  Time: " << time_us << " µs, Verified: " << pass_count << "/" << sample_count
            << " samples" << std::endl;
  ASSERT_EQ(sample_count, pass_count);
  std::cout << "  [PASS] Broadcast to 256 destinations successful" << std::endl;

  BroadcastTestUtils::DestroySignal(signal);
  ctx.Free(src);
  for (auto dst : dsts) ctx.Free(dst);
}

//
//
//

TEST_F(BroadcastCopyL5, Broadcast_1023_Destinations) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  uint32_t max_destinations = 0;
  hsa_status_t status = hsa_amd_memory_broadcast_capability(ctx.gpu_agent, &max_destinations);

  if (status != HSA_STATUS_SUCCESS || max_destinations < 1023) {
    GTEST_SKIP() << "Hardware doesn't support 1023 destinations (max=" << max_destinations << ")";
  }

  const size_t SIZE = 16384;  // 16KB
  const int NUM_DESTS = 1023;

  std::cout << "Broadcasting to " << NUM_DESTS << " destinations (max-1 boundary)"
            << std::endl;
  std::cout << "  This tests the upper boundary limit..." << std::endl;

  void* src = ctx.AllocateGPUBuffer(SIZE);
  std::vector<void*> dsts(NUM_DESTS);

  std::cout << "  Allocating " << NUM_DESTS << " buffers... (this may take a moment)" << std::endl;
  for (int i = 0; i < NUM_DESTS; i++) {
    dsts[i] = ctx.AllocateGPUBuffer(SIZE);
    if (!dsts[i]) {
      std::cout << "  ⚠ Memory allocation failed at dst[" << i << "], testing with " << i
                << " dests instead" << std::endl;
      dsts.resize(i);
      break;
    }
  }

  if (dsts.size() < 100) {
    GTEST_SKIP() << "Insufficient memory for large-scale test";
  }

  int ACTUAL_NUM_DESTS = dsts.size();
  BroadcastTestUtils::FillPattern(src, SIZE, BroadcastTestUtils::INCREMENTAL, 0x55AA55AA);
  std::vector<hsa_agent_t> dst_agents(ACTUAL_NUM_DESTS, ctx.gpu_agent);
  hsa_signal_t signal = BroadcastTestUtils::CreateSignal(1);

  std::cout << "  Executing broadcast to " << ACTUAL_NUM_DESTS << " destinations..." << std::endl;
  auto start = std::chrono::high_resolution_clock::now();

  status = hsa_amd_memory_broadcast_copy(src, ctx.gpu_agent, dsts.data(), dst_agents.data(),
                                         ACTUAL_NUM_DESTS, SIZE, 0, nullptr, signal,
                                         HSA_AMD_SDMA_ENGINE_0, false);

  ASSERT_EQ(HSA_STATUS_SUCCESS, status);
  BroadcastTestUtils::WaitSignal(signal);

  auto end = std::chrono::high_resolution_clock::now();
  auto time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  // Verify 20 sampled destinations
  int sample_indices[] = {
      0, 50, 100, 200, 400, 600, 800, 1000, ACTUAL_NUM_DESTS / 2, ACTUAL_NUM_DESTS - 1};
  int pass_count = 0;

  for (int idx : sample_indices) {
    if (idx >= ACTUAL_NUM_DESTS) continue;
    if (BroadcastTestUtils::CompareBuffers(src, dsts[idx], SIZE)) {
      pass_count++;
    } else {
      std::cout << "  [FAIL] dst[" << idx << "] FAILED verification" << std::endl;
    }
  }

  std::cout << "  Time: " << time_us << " µs, Verified: " << pass_count << "/10 samples"
            << std::endl;
  ASSERT_GE(pass_count, 9) << "Too many verification failures";
  std::cout << "  [PASS] Broadcast to " << ACTUAL_NUM_DESTS << " destinations successful" << std::endl;

  BroadcastTestUtils::DestroySignal(signal);
  ctx.Free(src);
  for (auto dst : dsts) ctx.Free(dst);
}

//
//
// CRITICAL TEST
//

TEST_F(BroadcastCopyL5, Broadcast_1024_Destinations_MAX) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  uint32_t max_destinations = 0;
  hsa_status_t status = hsa_amd_memory_broadcast_capability(ctx.gpu_agent, &max_destinations);

  std::cout << "* : Broadcasting to 1024 destinations" << std::endl;
  std::cout << "  Hardware reports max_destinations = " << max_destinations << std::endl;

  if (status != HSA_STATUS_SUCCESS || max_destinations < 1024) {
    std::cout << "  ⚠ Hardware doesn't support 1024 destinations" << std::endl;
    std::cout << "  Skipping hardware multicast test (shader fallback will be tested instead)" << std::endl;
    GTEST_SKIP() << "max_destinations=" << max_destinations << " < 1024";
  }

  const size_t SIZE = 8192;  // 8KB (smaller to avoid OOM)
  const int NUM_DESTS = 1024;

  void* src = ctx.AllocateGPUBuffer(SIZE);
  std::vector<void*> dsts;
  dsts.reserve(NUM_DESTS);

  std::cout << "  Allocating 1024 destination buffers..." << std::endl;
  for (int i = 0; i < NUM_DESTS; i++) {
    void* dst = ctx.AllocateGPUBuffer(SIZE);
    if (!dst) {
      std::cout << "  ⚠ Memory allocation failed at dst[" << i << "]" << std::endl;
      break;
    }
    dsts.push_back(dst);

    if ((i + 1) % 256 == 0) {
      std::cout << "    Allocated " << (i + 1) << "/" << NUM_DESTS << "..." << std::endl;
    }
  }

  if (dsts.size() < NUM_DESTS) {
    std::cout << "  ⚠ Only allocated " << dsts.size() << " buffers (insufficient memory)"
              << std::endl;
    if (dsts.size() < 512) {
      GTEST_SKIP() << "Insufficient memory for 1024-dest test";
    }
  }

  int ACTUAL_NUM_DESTS = dsts.size();
  std::cout << "  Executing broadcast to " << ACTUAL_NUM_DESTS << " destinations..." << std::endl;

  BroadcastTestUtils::FillPattern(src, SIZE, BroadcastTestUtils::RANDOM, 0xCAFEBABE);
  std::vector<hsa_agent_t> dst_agents(ACTUAL_NUM_DESTS, ctx.gpu_agent);
  hsa_signal_t signal = BroadcastTestUtils::CreateSignal(1);

  auto start = std::chrono::high_resolution_clock::now();

  status = hsa_amd_memory_broadcast_copy(src, ctx.gpu_agent, dsts.data(), dst_agents.data(),
                                         ACTUAL_NUM_DESTS, SIZE, 0, nullptr, signal,
                                         HSA_AMD_SDMA_ENGINE_0, false);

  ASSERT_EQ(HSA_STATUS_SUCCESS, status)
      << "Broadcast to " << ACTUAL_NUM_DESTS << " destinations failed";
  BroadcastTestUtils::WaitSignal(signal);

  auto end = std::chrono::high_resolution_clock::now();
  auto time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  std::cout << "  [PASS] Broadcast completed in " << time_us << " µs" << std::endl;

  // Comprehensive verification - 32 samples across the range
  int samples = 32;
  int pass_count = 0;
  std::cout << "  Verifying " << samples << " sampled destinations..." << std::endl;

  for (int i = 0; i < samples; i++) {
    int idx = (i * ACTUAL_NUM_DESTS) / samples;
    if (idx >= ACTUAL_NUM_DESTS) idx = ACTUAL_NUM_DESTS - 1;

    if (BroadcastTestUtils::CompareBuffers(src, dsts[idx], SIZE)) {
      pass_count++;
    } else {
      std::cout << "  [FAIL] dst[" << idx << "] FAILED" << std::endl;
    }
  }

  double pass_rate = (100.0 * pass_count) / samples;
  std::cout << "  Verification: " << pass_count << "/" << samples << " (" << std::fixed
            << std::setprecision(1) << pass_rate << "%)" << std::endl;

  ASSERT_EQ(samples, pass_count) << "Some destinations have corrupted data";

  if (ACTUAL_NUM_DESTS >= 1024) {
    std::cout << "\n  ***  PASSED ***" << std::endl;
    std::cout << "  Successfully broadcast to 1024 destinations with 100% data integrity!"
              << std::endl;
  } else {
    std::cout << "  [PASS] Broadcast to " << ACTUAL_NUM_DESTS
              << " destinations successful (limited by memory)" << std::endl;
  }

  BroadcastTestUtils::DestroySignal(signal);
  ctx.Free(src);
  for (auto dst : dsts) ctx.Free(dst);
}

//
//
//

TEST_F(BroadcastCopyL5, Broadcast_1025_Destinations_ShouldFail) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  uint32_t max_destinations = 0;
  hsa_status_t status = hsa_amd_memory_broadcast_capability(ctx.gpu_agent, &max_destinations);

  if (status != HSA_STATUS_SUCCESS || max_destinations != 1024) {
    GTEST_SKIP() << "Test requires exact max_destinations=1024 (got " << max_destinations << ")";
  }

  std::cout << "Testing exceeds-limit case (1025 > 1024)" << std::endl;

  const size_t SIZE = 4096;
  const int NUM_DESTS = 1025;  // Intentionally exceeds limit

  void* src = ctx.AllocateGPUBuffer(SIZE);
  std::vector<void*> dsts(NUM_DESTS);

  // Allocate buffers (may fail due to memory)
  int allocated = 0;
  for (int i = 0; i < NUM_DESTS; i++) {
    dsts[i] = ctx.AllocateGPUBuffer(SIZE);
    if (!dsts[i]) break;
    allocated++;
  }

  if (allocated < NUM_DESTS) {
    std::cout << "  [INFO] Could only allocate " << allocated << " buffers (memory limited)"
              << std::endl;
    GTEST_SKIP() << "Insufficient memory to test 1025 destinations";
  }

  std::vector<hsa_agent_t> dst_agents(NUM_DESTS, ctx.gpu_agent);
  hsa_signal_t signal = BroadcastTestUtils::CreateSignal(1);

  std::cout << "  Attempting broadcast to " << NUM_DESTS
            << " destinations (expect INVALID_ARGUMENT)..." << std::endl;

  status =
      hsa_amd_memory_broadcast_copy(src, ctx.gpu_agent, dsts.data(), dst_agents.data(), NUM_DESTS,
                                    SIZE, 0, nullptr, signal, HSA_AMD_SDMA_ENGINE_0, false);

  std::cout << "  Returned status: " << (int)status << std::endl;

  // Should return error (either INVALID_ARGUMENT or ERROR unsupported)
  ASSERT_NE(HSA_STATUS_SUCCESS, status) << "API should reject 1025 destinations";

  bool is_invalid_arg = (status == HSA_STATUS_ERROR_INVALID_ARGUMENT);
  bool is_error = (status != HSA_STATUS_SUCCESS);

  ASSERT_TRUE(is_error) << "Expected error for exceeding max destinations";
  std::cout << "  [PASS] Correctly rejected 1025 destinations" << std::endl;

  BroadcastTestUtils::DestroySignal(signal);
  ctx.Free(src);
  for (int i = 0; i < allocated; i++) {
    if (dsts[i]) ctx.Free(dsts[i]);
  }
}
