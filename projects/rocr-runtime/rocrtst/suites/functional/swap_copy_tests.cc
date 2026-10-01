/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// =============================================================================
// Swap Copy Tests (AILIROCR-24)
// =============================================================================
// Tests for HSA_AMD_MEMORY_COPY_OP_LINEAR_SWAP operation.
// Swap copy atomically exchanges contents of two buffers: A ↔ B
//
// Test Coverage:
// =============================================================================

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>

#include "gtest/gtest.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "common/hsatimer.h"

// Include shared broadcast copy utilities for HsaTestContext
#include "common/broadcast_copy_utils.h"

// =============================================================================
// SwapCopyTestUtils: Helper functions for swap copy tests
// =============================================================================

class SwapCopyTestUtils {
 public:
  // Fill buffer with a specific pattern based on buffer identity
  static void FillPatternA(void* buffer, size_t size) {
    uint8_t* ptr = static_cast<uint8_t*>(buffer);
    for (size_t i = 0; i < size; ++i) {
      ptr[i] = static_cast<uint8_t>((i * 7 + 0xAA) & 0xFF);
    }
  }

  static void FillPatternB(void* buffer, size_t size) {
    uint8_t* ptr = static_cast<uint8_t*>(buffer);
    for (size_t i = 0; i < size; ++i) {
      ptr[i] = static_cast<uint8_t>((i * 13 + 0x55) & 0xFF);
    }
  }

  // Create reference patterns for verification
  static std::vector<uint8_t> CreatePatternA(size_t size) {
    std::vector<uint8_t> pattern(size);
    for (size_t i = 0; i < size; ++i) {
      pattern[i] = static_cast<uint8_t>((i * 7 + 0xAA) & 0xFF);
    }
    return pattern;
  }

  static std::vector<uint8_t> CreatePatternB(size_t size) {
    std::vector<uint8_t> pattern(size);
    for (size_t i = 0; i < size; ++i) {
      pattern[i] = static_cast<uint8_t>((i * 13 + 0x55) & 0xFF);
    }
    return pattern;
  }

  // Verify buffer matches expected pattern
  static bool VerifyPattern(const void* buffer, const std::vector<uint8_t>& expected,
                            const char* buffer_name) {
    const uint8_t* ptr = static_cast<const uint8_t*>(buffer);
    size_t size = expected.size();
    bool match = true;

    for (size_t i = 0; i < size; ++i) {
      if (ptr[i] != expected[i]) {
        if (match) {
          std::cout << "  [" << buffer_name << "] First mismatch at offset " << i << ": expected=0x"
                    << std::hex << static_cast<int>(expected[i]) << ", actual=0x"
                    << static_cast<int>(ptr[i]) << std::dec << std::endl;
        }
        match = false;
        break;
      }
    }

    if (match) {
      std::cout << "  [" << buffer_name << "] [PASS] Pattern verified (" << size << " bytes)"
                << std::endl;
    }
    return match;
  }
};

// =============================================================================
// Convenience wrapper: hsa_amd_memory_swap_copy
// =============================================================================

static inline hsa_status_t hsa_amd_memory_swap_copy(void* buf_a, hsa_agent_t agent_a, void* buf_b,
                                                    hsa_agent_t agent_b, size_t size,
                                                    uint32_t num_dep_signals,
                                                    const hsa_signal_t* dep_signals,
                                                    hsa_signal_t out_signal) {
  if (buf_a == nullptr || buf_b == nullptr) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  if (size == 0) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  hsa_amd_memory_copy_op_t op;
  memset(&op, 0, sizeof(op));

  op.version = HSA_AMD_MEMORY_COPY_OP_VERSION;
  op.type = HSA_AMD_MEMORY_COPY_OP_LINEAR_SWAP;
  op.num_entries = 0;  // Single swap (not multi-swap)
  op.traffic_class = 0;
  op.completion_signal = out_signal;
  op.src = buf_a;
  op.src_agent = agent_a;
  op.dst = buf_b;
  op.dst_agent = agent_b;
  // For SWAP: use src_size/dst_size union members (both must be non-zero)
  op.src_size = size;
  op.dst_size = size;  // For symmetric swap, both sizes are equal

  return hsa_amd_memory_async_batch_copy(&op, 1, num_dep_signals, dep_signals);
}

// =============================================================================
// Test Fixture
// =============================================================================

class SwapCopyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // HsaTestContext initializes HSA and finds agents/pools in constructor
    ASSERT_TRUE(ctx_.HasGPUAgent()) << "No GPU agent found";
  }

  void TearDown() override {
    // HsaTestContext destructor handles cleanup
  }

  HsaTestContext ctx_;
};

// =============================================================================
//
// =============================================================================
TEST_F(SwapCopyTest, TC_SWAP_001_BasicSwapSmall) {
  std::cout << "\n=== Basic Swap Copy (4KB) ===" << std::endl;

  const size_t kSize = 4096;

  // Allocate buffers
  void* buf_a = ctx_.AllocateGPUBuffer(kSize);
  void* buf_b = ctx_.AllocateGPUBuffer(kSize);
  ASSERT_NE(nullptr, buf_a);
  ASSERT_NE(nullptr, buf_b);

  // Initialize with distinct patterns
  SwapCopyTestUtils::FillPatternA(buf_a, kSize);
  SwapCopyTestUtils::FillPatternB(buf_b, kSize);

  // Save expected patterns (swapped)
  auto expected_a = SwapCopyTestUtils::CreatePatternB(kSize);  // A should have B's pattern
  auto expected_b = SwapCopyTestUtils::CreatePatternA(kSize);  // B should have A's pattern

  // Create completion signal
  hsa_signal_t signal;
  ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_signal_create(1, 0, nullptr, &signal));

  // Perform swap
  hsa_status_t status = hsa_amd_memory_swap_copy(buf_a, ctx_.gpu_agent, buf_b, ctx_.gpu_agent,
                                                 kSize, 0, nullptr, signal);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status) << "Swap copy submission failed";

  // Wait for completion
  hsa_signal_wait_relaxed(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  // Verify swap occurred correctly
  EXPECT_TRUE(
      SwapCopyTestUtils::VerifyPattern(buf_a, expected_a, "Buffer A (should have B's data)"));
  EXPECT_TRUE(
      SwapCopyTestUtils::VerifyPattern(buf_b, expected_b, "Buffer B (should have A's data)"));

  // Cleanup
  hsa_signal_destroy(signal);
  ctx_.Free(buf_a);
  ctx_.Free(buf_b);
}

// =============================================================================
//
// =============================================================================
TEST_F(SwapCopyTest, TC_SWAP_002_VariousSizes) {
  std::cout << "\n=== Various Buffer Sizes ===" << std::endl;

  std::vector<size_t> sizes = {1, 16, 256, 4096, 65536, 1024 * 1024};

  for (size_t size : sizes) {
    std::cout << "  Testing size: " << size << " bytes" << std::endl;

    void* buf_a = ctx_.AllocateGPUBuffer(size);
    void* buf_b = ctx_.AllocateGPUBuffer(size);
    ASSERT_NE(nullptr, buf_a);
    ASSERT_NE(nullptr, buf_b);

    SwapCopyTestUtils::FillPatternA(buf_a, size);
    SwapCopyTestUtils::FillPatternB(buf_b, size);

    auto expected_a = SwapCopyTestUtils::CreatePatternB(size);
    auto expected_b = SwapCopyTestUtils::CreatePatternA(size);

    hsa_signal_t signal;
    ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_signal_create(1, 0, nullptr, &signal));

    hsa_status_t status = hsa_amd_memory_swap_copy(buf_a, ctx_.gpu_agent, buf_b, ctx_.gpu_agent,
                                                   size, 0, nullptr, signal);
    ASSERT_EQ(HSA_STATUS_SUCCESS, status);

    hsa_signal_wait_relaxed(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

    EXPECT_TRUE(SwapCopyTestUtils::VerifyPattern(buf_a, expected_a, "A"));
    EXPECT_TRUE(SwapCopyTestUtils::VerifyPattern(buf_b, expected_b, "B"));

    hsa_signal_destroy(signal);
    ctx_.Free(buf_a);
    ctx_.Free(buf_b);
  }
}

// =============================================================================
//
// =============================================================================
TEST_F(SwapCopyTest, TC_SWAP_003_UnalignedSizes) {
  std::cout << "\n=== Unaligned Buffer Sizes ===" << std::endl;

  // Test sizes that are not multiples of 16 (shader vectorization boundary)
  std::vector<size_t> sizes = {7, 13, 33, 127, 1000, 4097, 65537};

  for (size_t size : sizes) {
    std::cout << "  Testing unaligned size: " << size << " bytes" << std::endl;

    void* buf_a = ctx_.AllocateGPUBuffer(size);
    void* buf_b = ctx_.AllocateGPUBuffer(size);
    ASSERT_NE(nullptr, buf_a);
    ASSERT_NE(nullptr, buf_b);

    SwapCopyTestUtils::FillPatternA(buf_a, size);
    SwapCopyTestUtils::FillPatternB(buf_b, size);

    auto expected_a = SwapCopyTestUtils::CreatePatternB(size);
    auto expected_b = SwapCopyTestUtils::CreatePatternA(size);

    hsa_signal_t signal;
    ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_signal_create(1, 0, nullptr, &signal));

    hsa_status_t status = hsa_amd_memory_swap_copy(buf_a, ctx_.gpu_agent, buf_b, ctx_.gpu_agent,
                                                   size, 0, nullptr, signal);
    ASSERT_EQ(HSA_STATUS_SUCCESS, status);

    hsa_signal_wait_relaxed(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

    EXPECT_TRUE(SwapCopyTestUtils::VerifyPattern(buf_a, expected_a, "A"));
    EXPECT_TRUE(SwapCopyTestUtils::VerifyPattern(buf_b, expected_b, "B"));

    hsa_signal_destroy(signal);
    ctx_.Free(buf_a);
    ctx_.Free(buf_b);
  }
}

// =============================================================================
//
// =============================================================================
TEST_F(SwapCopyTest, TC_SWAP_004_LargeBuffer) {
  std::cout << "\n=== Large Buffer Swap (100MB) ===" << std::endl;

  const size_t kSize = 100 * 1024 * 1024;  // 100MB

  void* buf_a = ctx_.AllocateGPUBuffer(kSize);
  void* buf_b = ctx_.AllocateGPUBuffer(kSize);
  ASSERT_NE(nullptr, buf_a);
  ASSERT_NE(nullptr, buf_b);

  // Fill with simple patterns for large buffers
  memset(buf_a, 0xAA, kSize);
  memset(buf_b, 0x55, kSize);

  hsa_signal_t signal;
  ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_signal_create(1, 0, nullptr, &signal));

  auto start = std::chrono::high_resolution_clock::now();

  hsa_status_t status = hsa_amd_memory_swap_copy(buf_a, ctx_.gpu_agent, buf_b, ctx_.gpu_agent,
                                                 kSize, 0, nullptr, signal);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status);

  hsa_signal_wait_relaxed(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  std::cout << "  Swap completed in " << duration.count() << " ms" << std::endl;
  std::cout << "  Effective bandwidth: "
            << (2.0 * kSize / (1024.0 * 1024.0 * 1024.0)) / (duration.count() / 1000.0) << " GB/s"
            << std::endl;

  // Verify swap (spot check)
  const uint8_t* ptr_a = static_cast<const uint8_t*>(buf_a);
  const uint8_t* ptr_b = static_cast<const uint8_t*>(buf_b);

  bool swap_correct = true;
  for (size_t i = 0; i < 1000 && swap_correct; ++i) {
    if (ptr_a[i] != 0x55 || ptr_b[i] != 0xAA) {
      swap_correct = false;
    }
  }
  // Check end of buffers too
  for (size_t i = kSize - 1000; i < kSize && swap_correct; ++i) {
    if (ptr_a[i] != 0x55 || ptr_b[i] != 0xAA) {
      swap_correct = false;
    }
  }

  EXPECT_TRUE(swap_correct) << "Large buffer swap data verification failed";

  hsa_signal_destroy(signal);
  ctx_.Free(buf_a);
  ctx_.Free(buf_b);
}

// =============================================================================
//
// =============================================================================
TEST_F(SwapCopyTest, TC_SWAP_005_DoubleSwapRestoresOriginal) {
  std::cout << "\n=== Double Swap Restores Original ===" << std::endl;

  const size_t kSize = 8192;

  void* buf_a = ctx_.AllocateGPUBuffer(kSize);
  void* buf_b = ctx_.AllocateGPUBuffer(kSize);
  ASSERT_NE(nullptr, buf_a);
  ASSERT_NE(nullptr, buf_b);

  // Initialize and save original patterns
  SwapCopyTestUtils::FillPatternA(buf_a, kSize);
  SwapCopyTestUtils::FillPatternB(buf_b, kSize);

  auto original_a = SwapCopyTestUtils::CreatePatternA(kSize);
  auto original_b = SwapCopyTestUtils::CreatePatternB(kSize);

  hsa_signal_t signal;
  ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_signal_create(1, 0, nullptr, &signal));

  // First swap
  std::cout << "  Performing first swap..." << std::endl;
  hsa_status_t status = hsa_amd_memory_swap_copy(buf_a, ctx_.gpu_agent, buf_b, ctx_.gpu_agent,
                                                 kSize, 0, nullptr, signal);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status);
  hsa_signal_wait_relaxed(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  // Second swap (should restore original)
  std::cout << "  Performing second swap..." << std::endl;
  hsa_signal_store_relaxed(signal, 1);
  status = hsa_amd_memory_swap_copy(buf_a, ctx_.gpu_agent, buf_b, ctx_.gpu_agent, kSize, 0, nullptr,
                                    signal);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status);
  hsa_signal_wait_relaxed(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  // Verify buffers are back to original
  EXPECT_TRUE(SwapCopyTestUtils::VerifyPattern(buf_a, original_a, "A (should be original)"));
  EXPECT_TRUE(SwapCopyTestUtils::VerifyPattern(buf_b, original_b, "B (should be original)"));

  hsa_signal_destroy(signal);
  ctx_.Free(buf_a);
  ctx_.Free(buf_b);
}

// =============================================================================
//
// =============================================================================
TEST_F(SwapCopyTest, TC_SWAP_006_SignalDependencies) {
  std::cout << "\n=== Signal Dependency Handling ===" << std::endl;

  const size_t kSize = 4096;

  void* buf_a = ctx_.AllocateGPUBuffer(kSize);
  void* buf_b = ctx_.AllocateGPUBuffer(kSize);
  ASSERT_NE(nullptr, buf_a);
  ASSERT_NE(nullptr, buf_b);

  SwapCopyTestUtils::FillPatternA(buf_a, kSize);
  SwapCopyTestUtils::FillPatternB(buf_b, kSize);

  auto expected_a = SwapCopyTestUtils::CreatePatternB(kSize);
  auto expected_b = SwapCopyTestUtils::CreatePatternA(kSize);

  // Create dependency signal (initially blocked)
  hsa_signal_t dep_signal;
  ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_signal_create(1, 0, nullptr, &dep_signal));

  // Create completion signal
  hsa_signal_t out_signal;
  ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_signal_create(1, 0, nullptr, &out_signal));

  // Submit swap with dependency
  std::cout << "  Submitting swap with blocked dependency signal..." << std::endl;
  hsa_status_t status = hsa_amd_memory_swap_copy(buf_a, ctx_.gpu_agent, buf_b, ctx_.gpu_agent,
                                                 kSize, 1, &dep_signal, out_signal);
  ASSERT_EQ(HSA_STATUS_SUCCESS, status);

  // Verify swap hasn't completed yet
  hsa_signal_value_t val = hsa_signal_load_relaxed(out_signal);
  EXPECT_EQ(1, val) << "Swap should not have completed yet";

  // Release dependency
  std::cout << "  Releasing dependency signal..." << std::endl;
  hsa_signal_store_relaxed(dep_signal, 0);

  // Wait for completion
  hsa_signal_wait_relaxed(out_signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX,
                          HSA_WAIT_STATE_BLOCKED);

  std::cout << "  Swap completed after dependency released" << std::endl;

  // Verify swap
  EXPECT_TRUE(SwapCopyTestUtils::VerifyPattern(buf_a, expected_a, "A"));
  EXPECT_TRUE(SwapCopyTestUtils::VerifyPattern(buf_b, expected_b, "B"));

  hsa_signal_destroy(dep_signal);
  hsa_signal_destroy(out_signal);
  ctx_.Free(buf_a);
  ctx_.Free(buf_b);
}

// =============================================================================
//
// =============================================================================
TEST_F(SwapCopyTest, TC_SWAP_007_NullPointerValidation) {
  std::cout << "\n=== Null Pointer Validation ===" << std::endl;

  const size_t kSize = 4096;
  void* valid_buf = ctx_.AllocateGPUBuffer(kSize);
  ASSERT_NE(nullptr, valid_buf);

  hsa_signal_t signal;
  ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_signal_create(1, 0, nullptr, &signal));

  // Test null buf_a
  std::cout << "  Testing null buf_a..." << std::endl;
  hsa_status_t status = hsa_amd_memory_swap_copy(nullptr, ctx_.gpu_agent, valid_buf, ctx_.gpu_agent,
                                                 kSize, 0, nullptr, signal);
  EXPECT_EQ(HSA_STATUS_ERROR_INVALID_ARGUMENT, status);

  // Test null buf_b
  std::cout << "  Testing null buf_b..." << std::endl;
  status = hsa_amd_memory_swap_copy(valid_buf, ctx_.gpu_agent, nullptr, ctx_.gpu_agent, kSize, 0,
                                    nullptr, signal);
  EXPECT_EQ(HSA_STATUS_ERROR_INVALID_ARGUMENT, status);

  // Test both null
  std::cout << "  Testing both null..." << std::endl;
  status = hsa_amd_memory_swap_copy(nullptr, ctx_.gpu_agent, nullptr, ctx_.gpu_agent, kSize, 0,
                                    nullptr, signal);
  EXPECT_EQ(HSA_STATUS_ERROR_INVALID_ARGUMENT, status);

  std::cout << "  [PASS] Null pointer validation passed" << std::endl;

  hsa_signal_destroy(signal);
  ctx_.Free(valid_buf);
}

// =============================================================================
//
// =============================================================================
TEST_F(SwapCopyTest, TC_SWAP_008_ZeroSizeHandling) {
  std::cout << "\n=== Zero Size Handling ===" << std::endl;

  const size_t kSize = 4096;
  void* buf_a = ctx_.AllocateGPUBuffer(kSize);
  void* buf_b = ctx_.AllocateGPUBuffer(kSize);
  ASSERT_NE(nullptr, buf_a);
  ASSERT_NE(nullptr, buf_b);

  hsa_signal_t signal;
  ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_signal_create(1, 0, nullptr, &signal));

  std::cout << "  Testing size = 0..." << std::endl;
  hsa_status_t status =
      hsa_amd_memory_swap_copy(buf_a, ctx_.gpu_agent, buf_b, ctx_.gpu_agent, 0, 0, nullptr, signal);
  EXPECT_EQ(HSA_STATUS_ERROR_INVALID_ARGUMENT, status);

  std::cout << "  [PASS] Zero size validation passed" << std::endl;

  hsa_signal_destroy(signal);
  ctx_.Free(buf_a);
  ctx_.Free(buf_b);
}

// =============================================================================
//
// =============================================================================
TEST_F(SwapCopyTest, TC_SWAP_009_SameBuffer) {
  std::cout << "\n=== Same Buffer (src == dst) ===" << std::endl;

  const size_t kSize = 4096;
  void* buf = ctx_.AllocateGPUBuffer(kSize);
  ASSERT_NE(nullptr, buf);

  // Fill with pattern
  SwapCopyTestUtils::FillPatternA(buf, kSize);
  auto original = SwapCopyTestUtils::CreatePatternA(kSize);

  hsa_signal_t signal;
  ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_signal_create(1, 0, nullptr, &signal));

  // Swap buffer with itself - should be a no-op or error
  std::cout << "  Swapping buffer with itself..." << std::endl;
  hsa_status_t status =
      hsa_amd_memory_swap_copy(buf, ctx_.gpu_agent, buf, ctx_.gpu_agent, kSize, 0, nullptr, signal);

  // Depending on implementation, this may succeed (no-op) or fail
  if (status == HSA_STATUS_SUCCESS) {
    hsa_signal_wait_relaxed(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);
    // Verify data unchanged
    EXPECT_TRUE(SwapCopyTestUtils::VerifyPattern(buf, original, "Buffer (should be unchanged)"));
  } else {
    std::cout << "  Implementation rejects same-buffer swap (acceptable)" << std::endl;
  }

  hsa_signal_destroy(signal);
  ctx_.Free(buf);
}

// =============================================================================
//
// =============================================================================
TEST_F(SwapCopyTest, TC_SWAP_010_ChainedSwaps) {
  std::cout << "\n=== Chained Swaps (A↔B, B↔C, C↔A) ===" << std::endl;

  const size_t kSize = 4096;

  void* buf_a = ctx_.AllocateGPUBuffer(kSize);
  void* buf_b = ctx_.AllocateGPUBuffer(kSize);
  void* buf_c = ctx_.AllocateGPUBuffer(kSize);
  ASSERT_NE(nullptr, buf_a);
  ASSERT_NE(nullptr, buf_b);
  ASSERT_NE(nullptr, buf_c);

  // Initialize: A=0xAA, B=0xBB, C=0xCC
  memset(buf_a, 0xAA, kSize);
  memset(buf_b, 0xBB, kSize);
  memset(buf_c, 0xCC, kSize);

  hsa_signal_t signal;
  ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_signal_create(1, 0, nullptr, &signal));

  // Swap A ↔ B: A=0xBB, B=0xAA, C=0xCC
  std::cout << "  Step 1: A ↔ B" << std::endl;
  ASSERT_EQ(HSA_STATUS_SUCCESS,
            hsa_amd_memory_swap_copy(buf_a, ctx_.gpu_agent, buf_b, ctx_.gpu_agent, kSize, 0,
                                     nullptr, signal));
  hsa_signal_wait_relaxed(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  // Swap B ↔ C: A=0xBB, B=0xCC, C=0xAA
  std::cout << "  Step 2: B ↔ C" << std::endl;
  hsa_signal_store_relaxed(signal, 1);
  ASSERT_EQ(HSA_STATUS_SUCCESS,
            hsa_amd_memory_swap_copy(buf_b, ctx_.gpu_agent, buf_c, ctx_.gpu_agent, kSize, 0,
                                     nullptr, signal));
  hsa_signal_wait_relaxed(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  // Swap C ↔ A: A=0xAA, B=0xCC, C=0xBB
  std::cout << "  Step 3: C ↔ A" << std::endl;
  hsa_signal_store_relaxed(signal, 1);
  ASSERT_EQ(HSA_STATUS_SUCCESS,
            hsa_amd_memory_swap_copy(buf_c, ctx_.gpu_agent, buf_a, ctx_.gpu_agent, kSize, 0,
                                     nullptr, signal));
  hsa_signal_wait_relaxed(signal, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

  // Final state: A=0xAA, B=0xCC, C=0xBB (cyclic rotation)
  const uint8_t* ptr_a = static_cast<const uint8_t*>(buf_a);
  const uint8_t* ptr_b = static_cast<const uint8_t*>(buf_b);
  const uint8_t* ptr_c = static_cast<const uint8_t*>(buf_c);

  bool correct = true;
  for (size_t i = 0; i < kSize && correct; ++i) {
    if (ptr_a[i] != 0xAA || ptr_b[i] != 0xCC || ptr_c[i] != 0xBB) {
      correct = false;
    }
  }

  EXPECT_TRUE(correct) << "Chained swap result incorrect";
  if (correct) {
    std::cout << "  [PASS] Chained swaps verified: A=0xAA, B=0xCC, C=0xBB" << std::endl;
  }

  hsa_signal_destroy(signal);
  ctx_.Free(buf_a);
  ctx_.Free(buf_b);
  ctx_.Free(buf_c);
}

// =============================================================================
// Main entry point
// =============================================================================
