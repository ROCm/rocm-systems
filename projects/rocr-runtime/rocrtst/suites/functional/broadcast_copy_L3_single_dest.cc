/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// ROCrtst Level 3 Tests: Single Destination E2E
// Purpose: First end-to-end functional test with single destination (N=1)

#include <gtest/gtest.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <cstring>
#include "../../common/broadcast_copy_utils.h"

class BroadcastCopyL3 : public ::testing::Test {
 protected:
  void SetUp() override {}
};

//
//
//

TEST_F(BroadcastCopyL3, SingleDest_4KB_Minimal) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  const size_t SIZE = 4096;

  // Allocate and initialize source
  void* src = ctx.AllocateGPUBuffer(SIZE);
  uint32_t* src_ptr = reinterpret_cast<uint32_t*>(src);
  for (size_t i = 0; i < SIZE / 4; i++) {
    src_ptr[i] = 0xCAFEBABE + i;  // Sequential pattern
  }

  // Allocate destination
  void* dst = ctx.AllocateGPUBuffer(SIZE);
  memset(dst, 0x00, SIZE);  // Clear destination

  void* dst_list[1] = {dst};
  hsa_agent_t dst_agents[1] = {ctx.gpu_agent};

  // Create completion signal
  hsa_signal_t signal = BroadcastTestUtils::CreateSignal(1);

  std::cout << "Starting single-dest broadcast copy:" << std::endl;
  std::cout << "  src=" << src << ", dst=" << dst << ", size=" << SIZE << std::endl;

  // Execute broadcast copy
  auto start = std::chrono::high_resolution_clock::now();

  hsa_status_t status =
      hsa_amd_memory_broadcast_copy(src, ctx.gpu_agent, dst_list, dst_agents,
                                    1,  // Single destination
                                    SIZE, 0, nullptr, signal, HSA_AMD_SDMA_ENGINE_0, false);

  std::cout << "  submit_status=" << status << std::endl;
  ASSERT_EQ(HSA_STATUS_SUCCESS, status) << "Broadcast copy submission failed";

  // Wait for completion
  std::cout << "  Waiting for completion signal..." << std::endl;
  BroadcastTestUtils::WaitSignal(signal);

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  std::cout << "  Completed in " << duration.count() << " µs" << std::endl;

  // Verify data integrity
  uint32_t* dst_ptr = reinterpret_cast<uint32_t*>(dst);
  int errors = 0;
  for (size_t i = 0; i < SIZE / 4 && errors < 10; i++) {
    if (dst_ptr[i] != src_ptr[i]) {
      std::cout << "  [FAIL] Mismatch at offset " << i * 4 << ": expected=0x" << std::hex << src_ptr[i]
                << ", actual=0x" << dst_ptr[i] << std::dec << std::endl;
      errors++;
    }
  }

  if (errors == 0) {
    std::cout << "  [PASS] Data integrity verified (4096 bytes)" << std::endl;
  }

  ASSERT_EQ(0, errors) << "Data corruption detected";

  BroadcastTestUtils::DestroySignal(signal);
  ctx.Free(src);
  ctx.Free(dst);
}

//
//
//

TEST_F(BroadcastCopyL3, SingleDest_VariousSizes) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  std::vector<size_t> sizes = {64, 256, 1024, 4096, 16384, 65536, 262144, 1048576};

  std::cout << "Testing various copy sizes:" << std::endl;

  for (size_t size : sizes) {
    void* src = ctx.AllocateGPUBuffer(size);
    if (!src) {
      std::cout << "  Size " << std::setw(10) << size << " bytes: Allocation failed, skipping"
                << std::endl;
      continue;
    }

    void* dst = ctx.AllocateGPUBuffer(size);

    // Fill with pattern
    BroadcastTestUtils::FillPattern(src, size, BroadcastTestUtils::WALKING_BIT);

    void* dst_list[1] = {dst};
    hsa_agent_t dst_agents[1] = {ctx.gpu_agent};

    hsa_signal_t signal = BroadcastTestUtils::CreateSignal(1);

    hsa_status_t status =
        hsa_amd_memory_broadcast_copy(src, ctx.gpu_agent, dst_list, dst_agents, 1, size, 0, nullptr,
                                      signal, HSA_AMD_SDMA_ENGINE_0, false);

    ASSERT_EQ(HSA_STATUS_SUCCESS, status);
    BroadcastTestUtils::WaitSignal(signal);

    bool valid = BroadcastTestUtils::VerifyPattern(dst, size, BroadcastTestUtils::WALKING_BIT);

    std::cout << "  Size " << std::setw(10) << size << " bytes: " << (valid ? "[PASS] PASS" : "[FAIL] FAIL")
              << std::endl;

    ASSERT_TRUE(valid) << "Data corruption at size=" << size;

    BroadcastTestUtils::DestroySignal(signal);
    ctx.Free(src);
    ctx.Free(dst);
  }
}

//
//
//

TEST_F(BroadcastCopyL3, SingleDest_WithDependencySignal) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  const size_t SIZE = 4096;
  void* src = ctx.AllocateGPUBuffer(SIZE);
  void* dst = ctx.AllocateGPUBuffer(SIZE);
  void* dst_list[1] = {dst};
  hsa_agent_t dst_agents[1] = {ctx.gpu_agent};

  BroadcastTestUtils::FillPattern(src, SIZE, BroadcastTestUtils::SEQUENTIAL);

  // Create dependency signal (initially unsatisfied)
  hsa_signal_t dep_signal = BroadcastTestUtils::CreateSignal(1);
  hsa_signal_t completion_signal = BroadcastTestUtils::CreateSignal(1);

  std::cout << "Testing dependency signal handling:" << std::endl;
  std::cout << "  Submitting copy with dep_signal=1 (unsatisfied)..." << std::endl;

  hsa_signal_t dep_signals[1] = {dep_signal};

  hsa_status_t status = hsa_amd_memory_broadcast_copy(
      src, ctx.gpu_agent, dst_list, dst_agents, 1, SIZE, 1, dep_signals,  // Wait on dep_signal
      completion_signal, HSA_AMD_SDMA_ENGINE_0, false);

  ASSERT_EQ(HSA_STATUS_SUCCESS, status);

  // Copy should NOT complete yet
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  hsa_signal_value_t value = hsa_signal_load_scacquire(completion_signal);
  std::cout << "  After 100ms, completion_signal=" << value << " (expected 1, not completed)"
            << std::endl;
  ASSERT_EQ(1, value) << "Copy completed before dependency satisfied!";

  // Now satisfy dependency
  std::cout << "  Satisfying dependency signal..." << std::endl;
  hsa_signal_store_screlease(dep_signal, 0);  // Signal satisfied

  // Now copy should complete
  BroadcastTestUtils::WaitSignal(completion_signal);
  std::cout << "  [PASS] Copy completed after dependency satisfied" << std::endl;

  bool valid = BroadcastTestUtils::VerifyPattern(dst, SIZE, BroadcastTestUtils::SEQUENTIAL);
  ASSERT_TRUE(valid);

  BroadcastTestUtils::DestroySignal(dep_signal);
  BroadcastTestUtils::DestroySignal(completion_signal);
  ctx.Free(src);
  ctx.Free(dst);
}

//
//
//

TEST_F(BroadcastCopyL3, SingleDest_BasicCopy) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  const size_t SIZE = 4096;
  void* src = ctx.AllocateGPUBuffer(SIZE);
  void* dst = ctx.AllocateGPUBuffer(SIZE);
  void* dst_list[1] = {dst};
  hsa_agent_t dst_agents[1] = {ctx.gpu_agent};

  BroadcastTestUtils::FillPattern(src, SIZE, BroadcastTestUtils::INCREMENTAL);

  std::cout << "Basic single-dest copy with completion signal:" << std::endl;

  // Always use a real completion signal
  hsa_signal_t signal = BroadcastTestUtils::CreateSignal(1);

  hsa_status_t status =
      hsa_amd_memory_broadcast_copy(src, ctx.gpu_agent, dst_list, dst_agents, 1, SIZE, 0, nullptr,
                                    signal, HSA_AMD_SDMA_ENGINE_0, false);

  std::cout << "  submit_status=" << status << std::endl;
  ASSERT_EQ(HSA_STATUS_SUCCESS, status);

  // Wait for completion using real signal
  BroadcastTestUtils::WaitSignal(signal);

  bool valid = BroadcastTestUtils::VerifyPattern(dst, SIZE, BroadcastTestUtils::INCREMENTAL);
  if (valid) {
    std::cout << "  [PASS] Copy completed successfully" << std::endl;
  } else {
    std::cout << "  [FAIL] Data corruption detected" << std::endl;
  }

  ASSERT_TRUE(valid);

  BroadcastTestUtils::DestroySignal(signal);
  ctx.Free(src);
  ctx.Free(dst);
}

//
//
//

TEST_F(BroadcastCopyL3, SingleDest_DataPatterns) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  const size_t SIZE = 65536;  // 64KB

  BroadcastTestUtils::Pattern patterns[] = {
      BroadcastTestUtils::SEQUENTIAL,   BroadcastTestUtils::RANDOM,
      BroadcastTestUtils::WALKING_BIT,  BroadcastTestUtils::INCREMENTAL,
      BroadcastTestUtils::CHECKERBOARD, BroadcastTestUtils::ZERO,
      BroadcastTestUtils::ONES};

  const char* pattern_names[] = {"SEQUENTIAL",   "RANDOM", "WALKING_BIT", "INCREMENTAL",
                                 "CHECKERBOARD", "ZERO",   "ONES"};

  std::cout << "Data pattern validation tests:" << std::endl;

  for (size_t i = 0; i < sizeof(patterns) / sizeof(patterns[0]); i++) {
    void* src = ctx.AllocateGPUBuffer(SIZE);
    void* dst = ctx.AllocateGPUBuffer(SIZE);

    BroadcastTestUtils::FillPattern(src, SIZE, patterns[i]);
    memset(dst, 0xFF, SIZE);  // Fill with garbage

    void* dst_list[1] = {dst};
    hsa_agent_t dst_agents[1] = {ctx.gpu_agent};
    hsa_signal_t signal = BroadcastTestUtils::CreateSignal(1);

    hsa_status_t status =
        hsa_amd_memory_broadcast_copy(src, ctx.gpu_agent, dst_list, dst_agents, 1, SIZE, 0, nullptr,
                                      signal, HSA_AMD_SDMA_ENGINE_0, false);

    ASSERT_EQ(HSA_STATUS_SUCCESS, status);
    BroadcastTestUtils::WaitSignal(signal);

    bool valid = BroadcastTestUtils::VerifyPattern(dst, SIZE, patterns[i]);

    std::cout << "  Pattern " << std::setw(15) << std::left << pattern_names[i] << ": "
              << (valid ? "[PASS] PASS" : "[FAIL] FAIL") << std::endl;

    ASSERT_TRUE(valid) << "Data corruption with pattern " << pattern_names[i];

    BroadcastTestUtils::DestroySignal(signal);
    ctx.Free(src);
    ctx.Free(dst);
  }
}
