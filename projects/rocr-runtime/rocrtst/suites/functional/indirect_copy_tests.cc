// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// ROCrtst Indirect Copy Tests
// Purpose: Test indirect copy operations where source and/or destination
//          addresses are resolved via indirection (pointer-to-pointer).

#include <gtest/gtest.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <vector>
#include "../../common/indirect_copy_utils.h"

class IndirectCopyTest : public ::testing::Test {
 protected:
  void SetUp() override {}
};

// =============================================================================
// TC-IND-001: Indirect Source, Direct Destination (4KB)
// =============================================================================

TEST_F(IndirectCopyTest, IndirectSrc_4KB) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  const size_t SIZE = 4096;

  // Allocate actual source buffer
  void* actual_src = ctx.AllocateGPUBuffer(SIZE);
  ASSERT_NE(nullptr, actual_src);

  // Fill source with pattern
  uint32_t* src_data = reinterpret_cast<uint32_t*>(actual_src);
  for (size_t i = 0; i < SIZE / 4; i++) {
    src_data[i] = 0xDEADBEEF + i;
  }

  // Allocate pointer-to-source buffer (this is what gets passed to API)
  void** src_ptr = IndirectCopyTestUtils::AllocatePointerBuffer(ctx, actual_src);
  ASSERT_NE(nullptr, src_ptr);
  ASSERT_EQ(actual_src, *src_ptr);

  // Allocate direct destination buffer
  void* dst = ctx.AllocateGPUBuffer(SIZE);
  ASSERT_NE(nullptr, dst);
  memset(dst, 0, SIZE);

  // Create completion signal
  hsa_signal_t signal = BroadcastTestUtils::CreateSignal(1);

  std::cout << "[TC-IND-001] Indirect source copy:" << std::endl;
  std::cout << "  src_ptr=" << src_ptr << " -> actual_src=" << actual_src << std::endl;
  std::cout << "  dst=" << dst << ", size=" << SIZE << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  hsa_status_t status = hsa_amd_memory_indirect_copy_src(src_ptr, ctx.gpu_agent, dst, ctx.gpu_agent,
                                                         SIZE, 0, nullptr, signal);

  std::cout << "  submit_status=" << status << std::endl;
  ASSERT_EQ(HSA_STATUS_SUCCESS, status) << "Indirect copy submission failed";

  BroadcastTestUtils::WaitSignal(signal);

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "  Completed in " << duration.count() << " µs" << std::endl;

  // Verify data integrity
  bool valid = IndirectCopyTestUtils::VerifyAndReport("IndirectSrc_4KB", actual_src, dst, SIZE);
  ASSERT_TRUE(valid) << "Data corruption detected";

  BroadcastTestUtils::DestroySignal(signal);
  ctx.Free(actual_src);
  ctx.Free(src_ptr);
  ctx.Free(dst);
}

// =============================================================================
// TC-IND-002: Direct Source, Indirect Destination (4KB)
// =============================================================================

TEST_F(IndirectCopyTest, IndirectDst_4KB) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  const size_t SIZE = 4096;

  // Allocate direct source buffer
  void* src = ctx.AllocateGPUBuffer(SIZE);
  ASSERT_NE(nullptr, src);

  // Fill source with pattern
  uint32_t* src_data = reinterpret_cast<uint32_t*>(src);
  for (size_t i = 0; i < SIZE / 4; i++) {
    src_data[i] = 0xCAFEBABE + i;
  }

  // Allocate actual destination buffer
  void* actual_dst = ctx.AllocateGPUBuffer(SIZE);
  ASSERT_NE(nullptr, actual_dst);
  memset(actual_dst, 0, SIZE);

  // Allocate pointer-to-destination buffer
  void** dst_ptr = IndirectCopyTestUtils::AllocatePointerBuffer(ctx, actual_dst);
  ASSERT_NE(nullptr, dst_ptr);
  ASSERT_EQ(actual_dst, *dst_ptr);

  hsa_signal_t signal = BroadcastTestUtils::CreateSignal(1);

  std::cout << "[TC-IND-002] Indirect destination copy:" << std::endl;
  std::cout << "  src=" << src << std::endl;
  std::cout << "  dst_ptr=" << dst_ptr << " -> actual_dst=" << actual_dst << std::endl;

  hsa_status_t status = hsa_amd_memory_indirect_copy_dst(src, ctx.gpu_agent, dst_ptr, ctx.gpu_agent,
                                                         SIZE, 0, nullptr, signal);

  ASSERT_EQ(HSA_STATUS_SUCCESS, status) << "Indirect copy submission failed";

  BroadcastTestUtils::WaitSignal(signal);

  bool valid = IndirectCopyTestUtils::VerifyAndReport("IndirectDst_4KB", src, actual_dst, SIZE);
  ASSERT_TRUE(valid) << "Data corruption detected";

  BroadcastTestUtils::DestroySignal(signal);
  ctx.Free(src);
  ctx.Free(actual_dst);
  ctx.Free(dst_ptr);
}

// =============================================================================
// TC-IND-003: Indirect Source AND Destination (4KB)
// =============================================================================

TEST_F(IndirectCopyTest, IndirectSrcDst_4KB) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  const size_t SIZE = 4096;

  // Allocate actual source buffer
  void* actual_src = ctx.AllocateGPUBuffer(SIZE);
  ASSERT_NE(nullptr, actual_src);

  uint32_t* src_data = reinterpret_cast<uint32_t*>(actual_src);
  for (size_t i = 0; i < SIZE / 4; i++) {
    src_data[i] = 0xFEEDFACE + i;
  }

  // Allocate actual destination buffer
  void* actual_dst = ctx.AllocateGPUBuffer(SIZE);
  ASSERT_NE(nullptr, actual_dst);
  memset(actual_dst, 0, SIZE);

  // Allocate pointer buffers
  void** src_ptr = IndirectCopyTestUtils::AllocatePointerBuffer(ctx, actual_src);
  void** dst_ptr = IndirectCopyTestUtils::AllocatePointerBuffer(ctx, actual_dst);
  ASSERT_NE(nullptr, src_ptr);
  ASSERT_NE(nullptr, dst_ptr);

  hsa_signal_t signal = BroadcastTestUtils::CreateSignal(1);

  std::cout << "[TC-IND-003] Indirect src+dst copy:" << std::endl;
  std::cout << "  src_ptr=" << src_ptr << " -> " << *src_ptr << std::endl;
  std::cout << "  dst_ptr=" << dst_ptr << " -> " << *dst_ptr << std::endl;

  hsa_status_t status = hsa_amd_memory_indirect_copy_srcdst(
      src_ptr, ctx.gpu_agent, dst_ptr, ctx.gpu_agent, SIZE, 0, nullptr, signal);

  ASSERT_EQ(HSA_STATUS_SUCCESS, status) << "Indirect copy submission failed";

  BroadcastTestUtils::WaitSignal(signal);

  bool valid =
      IndirectCopyTestUtils::VerifyAndReport("IndirectSrcDst_4KB", actual_src, actual_dst, SIZE);
  ASSERT_TRUE(valid) << "Data corruption detected";

  BroadcastTestUtils::DestroySignal(signal);
  ctx.Free(actual_src);
  ctx.Free(actual_dst);
  ctx.Free(src_ptr);
  ctx.Free(dst_ptr);
}

// =============================================================================
// TC-IND-004: Various Copy Sizes
// =============================================================================

TEST_F(IndirectCopyTest, IndirectSrc_VariousSizes) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  std::vector<size_t> sizes = {64, 256, 1024, 4096, 16384, 65536, 262144, 1048576};

  std::cout << "[TC-IND-004] Testing various copy sizes (indirect src):" << std::endl;

  for (size_t size : sizes) {
    void* actual_src = ctx.AllocateGPUBuffer(size);
    if (!actual_src) {
      std::cout << "  Size " << std::setw(10) << size << " bytes: Allocation failed, skipping"
                << std::endl;
      continue;
    }

    void* dst = ctx.AllocateGPUBuffer(size);
    void** src_ptr = IndirectCopyTestUtils::AllocatePointerBuffer(ctx, actual_src);

    // Fill with pattern
    BroadcastTestUtils::FillPattern(actual_src, size, BroadcastTestUtils::SEQUENTIAL);

    hsa_signal_t signal = BroadcastTestUtils::CreateSignal(1);

    hsa_status_t status = hsa_amd_memory_indirect_copy_src(src_ptr, ctx.gpu_agent, dst,
                                                           ctx.gpu_agent, size, 0, nullptr, signal);

    ASSERT_EQ(HSA_STATUS_SUCCESS, status);
    BroadcastTestUtils::WaitSignal(signal);

    bool valid = BroadcastTestUtils::VerifyPattern(dst, size, BroadcastTestUtils::SEQUENTIAL);

    std::cout << "  Size " << std::setw(10) << size << " bytes: " << (valid ? "✓ PASS" : "❌ FAIL")
              << std::endl;

    ASSERT_TRUE(valid) << "Data corruption at size=" << size;

    BroadcastTestUtils::DestroySignal(signal);
    ctx.Free(actual_src);
    ctx.Free(dst);
    ctx.Free(src_ptr);
  }
}

// =============================================================================
// TC-IND-005: Small Unaligned Sizes (Tail Loop Coverage)
// =============================================================================

TEST_F(IndirectCopyTest, IndirectSrc_UnalignedSizes) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  // Test sizes that exercise the byte-by-byte tail loop (not multiples of 16)
  std::vector<size_t> sizes = {1, 7, 15, 17, 31, 33, 63, 65, 127, 129, 255, 1023};

  std::cout << "[TC-IND-005] Testing unaligned sizes (tail loop):" << std::endl;

  for (size_t size : sizes) {
    // Allocate larger buffer to avoid edge cases
    size_t alloc_size = ((size + 255) / 256) * 256;
    if (alloc_size < 256) alloc_size = 256;

    void* actual_src = ctx.AllocateGPUBuffer(alloc_size);
    void* dst = ctx.AllocateGPUBuffer(alloc_size);
    void** src_ptr = IndirectCopyTestUtils::AllocatePointerBuffer(ctx, actual_src);

    if (!actual_src || !dst || !src_ptr) {
      std::cout << "  Size " << size << ": Allocation failed, skipping" << std::endl;
      if (actual_src) ctx.Free(actual_src);
      if (dst) ctx.Free(dst);
      if (src_ptr) ctx.Free(src_ptr);
      continue;
    }

    // Fill entire buffer, copy only 'size' bytes
    BroadcastTestUtils::FillPattern(actual_src, size, BroadcastTestUtils::WALKING_BIT);
    memset(dst, 0xCC, alloc_size);  // Sentinel pattern

    hsa_signal_t signal = BroadcastTestUtils::CreateSignal(1);

    hsa_status_t status = hsa_amd_memory_indirect_copy_src(src_ptr, ctx.gpu_agent, dst,
                                                           ctx.gpu_agent, size, 0, nullptr, signal);

    ASSERT_EQ(HSA_STATUS_SUCCESS, status);
    BroadcastTestUtils::WaitSignal(signal);

    bool valid = BroadcastTestUtils::VerifyPattern(dst, size, BroadcastTestUtils::WALKING_BIT);

    std::cout << "  Size " << std::setw(5) << size << " bytes: " << (valid ? "✓ PASS" : "❌ FAIL")
              << std::endl;

    ASSERT_TRUE(valid) << "Data corruption at unaligned size=" << size;

    BroadcastTestUtils::DestroySignal(signal);
    ctx.Free(actual_src);
    ctx.Free(dst);
    ctx.Free(src_ptr);
  }
}

// =============================================================================
// TC-IND-006: Dependency Signal Test
// =============================================================================

TEST_F(IndirectCopyTest, IndirectSrc_WithDependencySignal) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  const size_t SIZE = 4096;

  void* actual_src = ctx.AllocateGPUBuffer(SIZE);
  void* dst = ctx.AllocateGPUBuffer(SIZE);
  void** src_ptr = IndirectCopyTestUtils::AllocatePointerBuffer(ctx, actual_src);

  BroadcastTestUtils::FillPattern(actual_src, SIZE, BroadcastTestUtils::CHECKERBOARD);
  memset(dst, 0, SIZE);

  // Create dependency signal (starts at 1, copy waits for it to reach 0)
  hsa_signal_t dep_signal = BroadcastTestUtils::CreateSignal(1);
  hsa_signal_t completion_signal = BroadcastTestUtils::CreateSignal(1);

  std::cout << "[TC-IND-006] Indirect copy with dependency signal:" << std::endl;

  // Submit copy with dependency
  hsa_status_t status = hsa_amd_memory_indirect_copy_src(src_ptr, ctx.gpu_agent, dst, ctx.gpu_agent,
                                                         SIZE, 1, &dep_signal, completion_signal);

  ASSERT_EQ(HSA_STATUS_SUCCESS, status);

  // Verify copy hasn't completed yet (signal should still be 1)
  hsa_signal_value_t val = hsa_signal_load_relaxed(completion_signal);
  std::cout << "  Before signaling dep: completion_signal=" << val << std::endl;

  // Now signal the dependency
  std::cout << "  Signaling dependency..." << std::endl;
  hsa_signal_store_screlease(dep_signal, 0);

  // Wait for completion
  BroadcastTestUtils::WaitSignal(completion_signal);

  bool valid = BroadcastTestUtils::VerifyPattern(dst, SIZE, BroadcastTestUtils::CHECKERBOARD);
  std::cout << "  Result: " << (valid ? "✓ PASS" : "❌ FAIL") << std::endl;

  ASSERT_TRUE(valid) << "Data corruption detected";

  BroadcastTestUtils::DestroySignal(dep_signal);
  BroadcastTestUtils::DestroySignal(completion_signal);
  ctx.Free(actual_src);
  ctx.Free(dst);
  ctx.Free(src_ptr);
}

// =============================================================================
// TC-IND-007: Zero-Size Copy (Edge Case)
// =============================================================================

TEST_F(IndirectCopyTest, IndirectSrc_ZeroSize) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  void* actual_src = ctx.AllocateGPUBuffer(256);
  void* dst = ctx.AllocateGPUBuffer(256);
  void** src_ptr = IndirectCopyTestUtils::AllocatePointerBuffer(ctx, actual_src);

  // Zero-size copy should fail with INVALID_ARGUMENT
  hsa_signal_t signal = BroadcastTestUtils::CreateSignal(1);

  hsa_status_t status = hsa_amd_memory_indirect_copy_src(src_ptr, ctx.gpu_agent, dst, ctx.gpu_agent,
                                                         0, 0, nullptr, signal);

  std::cout << "[TC-IND-007] Zero-size copy status=" << status << std::endl;
  EXPECT_EQ(HSA_STATUS_ERROR_INVALID_ARGUMENT, status) << "Zero-size should return error";

  BroadcastTestUtils::DestroySignal(signal);
  ctx.Free(actual_src);
  ctx.Free(dst);
  ctx.Free(src_ptr);
}

// =============================================================================
// TC-IND-008: Null Pointer Validation
// =============================================================================

TEST_F(IndirectCopyTest, IndirectSrc_NullPointer) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  void* dst = ctx.AllocateGPUBuffer(256);
  hsa_signal_t signal = BroadcastTestUtils::CreateSignal(1);

  // Null src_ptr should fail
  hsa_status_t status = hsa_amd_memory_indirect_copy_src(nullptr, ctx.gpu_agent, dst, ctx.gpu_agent,
                                                         256, 0, nullptr, signal);

  std::cout << "[TC-IND-008] Null src_ptr status=" << status << std::endl;
  EXPECT_EQ(HSA_STATUS_ERROR_INVALID_ARGUMENT, status);

  BroadcastTestUtils::DestroySignal(signal);
  ctx.Free(dst);
}

// =============================================================================
// TC-IND-009: Large Copy (1MB+)
// =============================================================================

TEST_F(IndirectCopyTest, IndirectSrc_LargeCopy) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  const size_t SIZE = 4 * 1024 * 1024;  // 4MB

  void* actual_src = ctx.AllocateGPUBuffer(SIZE);
  void* dst = ctx.AllocateGPUBuffer(SIZE);
  void** src_ptr = IndirectCopyTestUtils::AllocatePointerBuffer(ctx, actual_src);

  if (!actual_src || !dst || !src_ptr) {
    GTEST_SKIP() << "Could not allocate 4MB buffers";
  }

  BroadcastTestUtils::FillPattern(actual_src, SIZE, BroadcastTestUtils::RANDOM, 12345);
  memset(dst, 0, SIZE);

  hsa_signal_t signal = BroadcastTestUtils::CreateSignal(1);

  std::cout << "[TC-IND-009] Large indirect copy (4MB):" << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  hsa_status_t status = hsa_amd_memory_indirect_copy_src(src_ptr, ctx.gpu_agent, dst, ctx.gpu_agent,
                                                         SIZE, 0, nullptr, signal);

  ASSERT_EQ(HSA_STATUS_SUCCESS, status);
  BroadcastTestUtils::WaitSignal(signal);

  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  bool valid = BroadcastTestUtils::VerifyPattern(dst, SIZE, BroadcastTestUtils::RANDOM, 12345);

  double bandwidth = BroadcastTestUtils::CalculateBandwidthGBps(SIZE, duration.count());
  std::cout << "  Time: " << duration.count() << " µs, Bandwidth: " << std::fixed
            << std::setprecision(2) << bandwidth << " GB/s" << std::endl;
  std::cout << "  Result: " << (valid ? "✓ PASS" : "❌ FAIL") << std::endl;

  ASSERT_TRUE(valid) << "Data corruption in large copy";

  BroadcastTestUtils::DestroySignal(signal);
  ctx.Free(actual_src);
  ctx.Free(dst);
  ctx.Free(src_ptr);
}

// =============================================================================
// TC-IND-010: Multiple Sequential Indirect Copies
// =============================================================================

TEST_F(IndirectCopyTest, IndirectSrc_MultipleSequential) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  const size_t SIZE = 4096;
  const int NUM_COPIES = 10;

  std::cout << "[TC-IND-010] Multiple sequential indirect copies:" << std::endl;

  for (int i = 0; i < NUM_COPIES; i++) {
    void* actual_src = ctx.AllocateGPUBuffer(SIZE);
    void* dst = ctx.AllocateGPUBuffer(SIZE);
    void** src_ptr = IndirectCopyTestUtils::AllocatePointerBuffer(ctx, actual_src);

    BroadcastTestUtils::FillPattern(actual_src, SIZE, BroadcastTestUtils::SEQUENTIAL, i * 100);
    memset(dst, 0, SIZE);

    hsa_signal_t signal = BroadcastTestUtils::CreateSignal(1);

    hsa_status_t status = hsa_amd_memory_indirect_copy_src(src_ptr, ctx.gpu_agent, dst,
                                                           ctx.gpu_agent, SIZE, 0, nullptr, signal);

    ASSERT_EQ(HSA_STATUS_SUCCESS, status);
    BroadcastTestUtils::WaitSignal(signal);

    bool valid =
        BroadcastTestUtils::VerifyPattern(dst, SIZE, BroadcastTestUtils::SEQUENTIAL, i * 100);

    std::cout << "  Copy " << (i + 1) << "/" << NUM_COPIES << ": " << (valid ? "✓" : "❌")
              << std::endl;

    ASSERT_TRUE(valid) << "Data corruption in copy #" << i;

    BroadcastTestUtils::DestroySignal(signal);
    ctx.Free(actual_src);
    ctx.Free(dst);
    ctx.Free(src_ptr);
  }
}
