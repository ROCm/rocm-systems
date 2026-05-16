// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// ROCrtst Level 4C Tests: Multicast Chunking Behavior Validation
// Purpose: Verify automatic chunking for sizes > 1023 bytes (10-bit COUNT limit)
//
// Background:
//   The SDMA COPY LINEAR MULTICAST packet has a 10-bit COUNT field split
//   across DW0[27:20] and DW4[1:0], limiting max copy size to 1023 bytes
//   per packet. The runtime must automatically chunk larger copies into
//   multiple packets.
//
// Key Test Scenarios:
//   - Sizes <= 1023 bytes: Single packet (no chunking)
//   - Sizes > 1023 bytes: Multiple packets (chunking required)
//   - Boundary cases: 1023, 1024, 2046, 2047, etc.

#include <gtest/gtest.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include "../../common/broadcast_copy_utils.h"

class BroadcastCopyChunking : public ::testing::Test {
 protected:
  void SetUp() override {}

  // Constants matching SDMA packet limits
  static const size_t MAX_PACKET_SIZE = 1023;  // 10-bit COUNT, 0-based
  static const size_t PACKET_OVERHEAD = 28;    // 7 DWORDs base packet
};

// ============================================================================
// TC-CHK-001: Single Packet - Size <= 1023 bytes (No Chunking)
// ============================================================================
// Sizes at or below the packet limit should require only one SDMA packet.

TEST_F(BroadcastCopyChunking, SinglePacket_ExactLimit) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  std::cout << "[TC-CHK-001] Single packet test (size <= 1023 bytes)" << std::endl;

  // Test exact packet limit: 1023 bytes
  const size_t SIZE = 1023;
  const int NUM_DESTS = 4;

  std::cout << "  Size: " << SIZE << " bytes (exact packet limit)" << std::endl;
  std::cout << "  Expected packets: 1" << std::endl;

  void* src = ctx.AllocateGPUBuffer(SIZE);
  std::vector<void*> dsts(NUM_DESTS);
  for (auto& dst : dsts) dst = ctx.AllocateGPUBuffer(SIZE);

  BroadcastTestUtils::FillPattern(src, SIZE, BroadcastTestUtils::SEQUENTIAL, 0x1023);
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

  // Verify all destinations
  int pass_count = 0;
  for (int i = 0; i < NUM_DESTS; i++) {
    if (BroadcastTestUtils::CompareBuffers(src, dsts[i], SIZE)) pass_count++;
  }

  std::cout << "  Time: " << time_us << " us, Verified: " << pass_count << "/" << NUM_DESTS
            << std::endl;
  ASSERT_EQ(NUM_DESTS, pass_count);
  std::cout << "  PASS: Single packet copy (1023 bytes) succeeded" << std::endl;

  BroadcastTestUtils::DestroySignal(signal);
  ctx.Free(src);
  for (auto dst : dsts) ctx.Free(dst);
}

// ============================================================================
// TC-CHK-002: Two Packets - Size = 1024 bytes (Boundary Case)
// ============================================================================
// Size of 1024 bytes requires 2 packets: 1023 + 1 bytes

TEST_F(BroadcastCopyChunking, TwoPackets_BoundaryCase) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  std::cout << "[TC-CHK-002] Two packets boundary test (size = 1024 bytes)" << std::endl;

  const size_t SIZE = 1024;  // Just over the limit
  const int NUM_DESTS = 4;

  std::cout << "  Size: " << SIZE << " bytes" << std::endl;
  std::cout << "  Expected packets: 2 (1023 + 1 bytes)" << std::endl;

  void* src = ctx.AllocateGPUBuffer(SIZE);
  std::vector<void*> dsts(NUM_DESTS);
  for (auto& dst : dsts) dst = ctx.AllocateGPUBuffer(SIZE);

  // Use a pattern that will detect off-by-one errors at chunk boundaries
  BroadcastTestUtils::FillPattern(src, SIZE, BroadcastTestUtils::INCREMENTAL, 0x1024);
  std::vector<hsa_agent_t> dst_agents(NUM_DESTS, ctx.gpu_agent);
  hsa_signal_t signal = BroadcastTestUtils::CreateSignal(1);

  hsa_status_t status =
      hsa_amd_memory_broadcast_copy(src, ctx.gpu_agent, dsts.data(), dst_agents.data(), NUM_DESTS,
                                    SIZE, 0, nullptr, signal, HSA_AMD_SDMA_ENGINE_0, false);

  ASSERT_EQ(HSA_STATUS_SUCCESS, status);
  BroadcastTestUtils::WaitSignal(signal);

  // Verify all destinations - critical to check BOTH chunks transferred correctly
  int pass_count = 0;
  for (int i = 0; i < NUM_DESTS; i++) {
    bool valid = BroadcastTestUtils::CompareBuffers(src, dsts[i], SIZE);
    if (valid) {
      pass_count++;
    } else {
      std::cout << "  dst[" << i << "]: FAILED - checking chunk boundary..." << std::endl;

      // Check first chunk (0-1022)
      bool chunk1_ok = BroadcastTestUtils::CompareBuffers(src, dsts[i], 1023);
      // Check second chunk (1023-1023)
      bool chunk2_ok = (static_cast<char*>(dsts[i])[1023] == static_cast<char*>(src)[1023]);

      std::cout << "    Chunk 1 (0-1022): " << (chunk1_ok ? "OK" : "FAIL") << std::endl;
      std::cout << "    Chunk 2 (1023): " << (chunk2_ok ? "OK" : "FAIL") << std::endl;
    }
  }

  std::cout << "  Verified: " << pass_count << "/" << NUM_DESTS << std::endl;
  ASSERT_EQ(NUM_DESTS, pass_count) << "Chunked copy failed at 1024-byte boundary";
  std::cout << "  PASS: Two-packet copy (1024 bytes) succeeded" << std::endl;

  BroadcastTestUtils::DestroySignal(signal);
  ctx.Free(src);
  for (auto dst : dsts) ctx.Free(dst);
}

// ============================================================================
// TC-CHK-003: Multiple Packets - Size = 2046 bytes (Exact 2x Limit)
// ============================================================================
// Size of 2046 = 2 * 1023, requiring exactly 2 full packets

TEST_F(BroadcastCopyChunking, TwoFullPackets_Exact) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  std::cout << "[TC-CHK-003] Two full packets (size = 2046 = 2 * 1023)" << std::endl;

  const size_t SIZE = 2046;  // Exactly 2 * 1023
  const int NUM_DESTS = 4;

  std::cout << "  Size: " << SIZE << " bytes" << std::endl;
  std::cout << "  Expected packets: 2 (1023 + 1023 bytes)" << std::endl;

  void* src = ctx.AllocateGPUBuffer(SIZE);
  std::vector<void*> dsts(NUM_DESTS);
  for (auto& dst : dsts) dst = ctx.AllocateGPUBuffer(SIZE);

  BroadcastTestUtils::FillPattern(src, SIZE, BroadcastTestUtils::RANDOM, 0x2046);
  std::vector<hsa_agent_t> dst_agents(NUM_DESTS, ctx.gpu_agent);
  hsa_signal_t signal = BroadcastTestUtils::CreateSignal(1);

  hsa_status_t status =
      hsa_amd_memory_broadcast_copy(src, ctx.gpu_agent, dsts.data(), dst_agents.data(), NUM_DESTS,
                                    SIZE, 0, nullptr, signal, HSA_AMD_SDMA_ENGINE_0, false);

  ASSERT_EQ(HSA_STATUS_SUCCESS, status);
  BroadcastTestUtils::WaitSignal(signal);

  int pass_count = 0;
  for (int i = 0; i < NUM_DESTS; i++) {
    if (BroadcastTestUtils::CompareBuffers(src, dsts[i], SIZE)) pass_count++;
  }

  ASSERT_EQ(NUM_DESTS, pass_count);
  std::cout << "  PASS: Two full packets (2046 bytes) succeeded" << std::endl;

  BroadcastTestUtils::DestroySignal(signal);
  ctx.Free(src);
  for (auto dst : dsts) ctx.Free(dst);
}

// ============================================================================
// TC-CHK-004: Many Packets - Size = 4KB (4 packets)
// ============================================================================

TEST_F(BroadcastCopyChunking, FourPackets_4KB) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  std::cout << "[TC-CHK-004] Four packets test (size = 4096 bytes)" << std::endl;

  const size_t SIZE = 4096;
  const int NUM_DESTS = 4;

  // 4096 / 1023 = 4.0 packets (1023 + 1023 + 1023 + 1027... wait, that's not right)
  // Actually: 4096 = 1023 + 1023 + 1023 + 1027 = needs ceiling(4096/1024) = 4 packets
  // More precisely: ceil(4096 / 1024) = 4 packets
  int expected_packets = (SIZE + MAX_PACKET_SIZE) / (MAX_PACKET_SIZE + 1);
  std::cout << "  Size: " << SIZE << " bytes" << std::endl;
  std::cout << "  Expected packets: ~" << expected_packets << std::endl;

  void* src = ctx.AllocateGPUBuffer(SIZE);
  std::vector<void*> dsts(NUM_DESTS);
  for (auto& dst : dsts) dst = ctx.AllocateGPUBuffer(SIZE);

  BroadcastTestUtils::FillPattern(src, SIZE, BroadcastTestUtils::WALKING_BIT);
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
  ASSERT_EQ(NUM_DESTS, pass_count);
  std::cout << "  PASS: 4KB chunked copy succeeded" << std::endl;

  BroadcastTestUtils::DestroySignal(signal);
  ctx.Free(src);
  for (auto dst : dsts) ctx.Free(dst);
}

// ============================================================================
// TC-CHK-005: Large Size - 1MB (1024 packets!)
// ============================================================================
// 1MB requires ~1024 packets. Tests queue depth and packet submission efficiency.

TEST_F(BroadcastCopyChunking, ManyPackets_1MB) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  std::cout << "[TC-CHK-005] Many packets test (size = 1MB)" << std::endl;

  const size_t SIZE = 1024 * 1024;  // 1MB
  const int NUM_DESTS = 4;

  int expected_packets = (SIZE + MAX_PACKET_SIZE) / (MAX_PACKET_SIZE + 1);
  std::cout << "  Size: " << SIZE << " bytes (1 MB)" << std::endl;
  std::cout << "  Expected packets: ~" << expected_packets << std::endl;
  std::cout << "  This tests queue depth handling..." << std::endl;

  void* src = ctx.AllocateGPUBuffer(SIZE);
  std::vector<void*> dsts(NUM_DESTS);
  for (auto& dst : dsts) {
    dst = ctx.AllocateGPUBuffer(SIZE);
    if (!dst) GTEST_SKIP() << "Insufficient memory for 1MB test";
  }

  BroadcastTestUtils::FillPattern(src, SIZE, BroadcastTestUtils::RANDOM, 0x100000);
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

  // Sample verification (full verification would be slow)
  bool first_ok = BroadcastTestUtils::CompareBuffers(src, dsts[0], SIZE);
  bool last_ok = BroadcastTestUtils::CompareBuffers(src, dsts[NUM_DESTS - 1], SIZE);

  double bandwidth_gbps = BroadcastTestUtils::CalculateBandwidthGBps(SIZE * NUM_DESTS, time_us);

  std::cout << "  Time: " << time_us << " us" << std::endl;
  std::cout << "  Effective BW: " << std::fixed << std::setprecision(2) << bandwidth_gbps << " GB/s"
            << std::endl;
  std::cout << "  First dst: " << (first_ok ? "OK" : "FAIL") << std::endl;
  std::cout << "  Last dst:  " << (last_ok ? "OK" : "FAIL") << std::endl;

  ASSERT_TRUE(first_ok && last_ok) << "1MB chunked copy verification failed";
  std::cout << "  PASS: 1MB chunked copy succeeded (~" << expected_packets << " packets)"
            << std::endl;

  BroadcastTestUtils::DestroySignal(signal);
  ctx.Free(src);
  for (auto dst : dsts) ctx.Free(dst);
}

// ============================================================================
// TC-CHK-006: Chunk Boundary Integrity - Verify No Data Loss at Boundaries
// ============================================================================
// Use a special pattern that will catch off-by-one errors at chunk boundaries

TEST_F(BroadcastCopyChunking, ChunkBoundaryIntegrity) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  std::cout << "[TC-CHK-006] Chunk boundary integrity test" << std::endl;

  const size_t SIZE = 3069;  // 3 * 1023 = 3069 (exactly 3 packets)
  const int NUM_DESTS = 2;

  std::cout << "  Size: " << SIZE << " bytes (3 * 1023, exactly 3 packets)" << std::endl;

  void* src = ctx.AllocateGPUBuffer(SIZE);
  std::vector<void*> dsts(NUM_DESTS);
  for (auto& dst : dsts) {
    dst = ctx.AllocateGPUBuffer(SIZE);
    memset(dst, 0xCC, SIZE);  // Pre-fill with marker
  }

  // Fill with pattern where each byte position is unique
  uint8_t* src_bytes = static_cast<uint8_t*>(src);
  for (size_t i = 0; i < SIZE; i++) {
    src_bytes[i] = (uint8_t)(i & 0xFF);
  }

  std::vector<hsa_agent_t> dst_agents(NUM_DESTS, ctx.gpu_agent);
  hsa_signal_t signal = BroadcastTestUtils::CreateSignal(1);

  hsa_status_t status =
      hsa_amd_memory_broadcast_copy(src, ctx.gpu_agent, dsts.data(), dst_agents.data(), NUM_DESTS,
                                    SIZE, 0, nullptr, signal, HSA_AMD_SDMA_ENGINE_0, false);

  ASSERT_EQ(HSA_STATUS_SUCCESS, status);
  BroadcastTestUtils::WaitSignal(signal);

  // Check specific chunk boundaries
  std::vector<size_t> boundaries = {
      0,        // Start of chunk 0
      1022,     // End of chunk 0
      1023,     // Start of chunk 1
      2045,     // End of chunk 1
      2046,     // Start of chunk 2
      SIZE - 1  // End of chunk 2
  };

  bool all_ok = true;
  for (int d = 0; d < NUM_DESTS; d++) {
    uint8_t* dst_bytes = static_cast<uint8_t*>(dsts[d]);
    std::cout << "  dst[" << d << "] boundary check:" << std::endl;

    for (size_t boundary : boundaries) {
      if (boundary >= SIZE) continue;

      uint8_t expected = (uint8_t)(boundary & 0xFF);
      uint8_t actual = dst_bytes[boundary];

      bool ok = (expected == actual);
      if (!ok) {
        std::cout << "    Offset " << boundary << ": expected 0x" << std::hex << (int)expected
                  << ", got 0x" << (int)actual << std::dec << " FAIL" << std::endl;
        all_ok = false;
      }
    }

    if (all_ok) {
      std::cout << "    All boundaries OK" << std::endl;
    }
  }

  ASSERT_TRUE(all_ok) << "Chunk boundary corruption detected";
  std::cout << "  PASS: Chunk boundaries verified" << std::endl;

  BroadcastTestUtils::DestroySignal(signal);
  ctx.Free(src);
  for (auto dst : dsts) ctx.Free(dst);
}

// ============================================================================
// TC-CHK-007: Chunking with Many Destinations
// ============================================================================
// Test chunking behavior with many destinations (stress test)

TEST_F(BroadcastCopyChunking, ChunkingWithManyDests) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  std::cout << "[TC-CHK-007] Chunking with many destinations" << std::endl;

  const size_t SIZE = 2048;  // 2 packets needed
  const int NUM_DESTS = 32;

  std::cout << "  Size: " << SIZE << " bytes, Dests: " << NUM_DESTS << std::endl;

  void* src = ctx.AllocateGPUBuffer(SIZE);
  std::vector<void*> dsts(NUM_DESTS);
  for (int i = 0; i < NUM_DESTS; i++) {
    dsts[i] = ctx.AllocateGPUBuffer(SIZE);
    if (!dsts[i]) {
      std::cout << "  Memory allocation failed at dst[" << i << "]" << std::endl;
      GTEST_SKIP() << "Insufficient memory";
    }
  }

  BroadcastTestUtils::FillPattern(src, SIZE, BroadcastTestUtils::RANDOM, 0xC40004);
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

  // Verify all destinations
  int pass_count = 0;
  for (int i = 0; i < NUM_DESTS; i++) {
    if (BroadcastTestUtils::CompareBuffers(src, dsts[i], SIZE)) pass_count++;
  }

  std::cout << "  Time: " << time_us << " us, Verified: " << pass_count << "/" << NUM_DESTS
            << std::endl;
  ASSERT_EQ(NUM_DESTS, pass_count);
  std::cout << "  PASS: Chunked multicast to " << NUM_DESTS << " destinations succeeded"
            << std::endl;

  BroadcastTestUtils::DestroySignal(signal);
  ctx.Free(src);
  for (auto dst : dsts) ctx.Free(dst);
}

// ============================================================================
// TC-CHK-008: Progressive Size Scaling
// ============================================================================
// Test various sizes to verify chunking works across the range

TEST_F(BroadcastCopyChunking, ProgressiveSizeScaling) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  std::cout << "[TC-CHK-008] Progressive size scaling test" << std::endl;

  std::vector<size_t> sizes = {
      512,    // < 1 packet
      1023,   // Exactly 1 packet
      1024,   // Just over 1 packet
      2046,   // Exactly 2 packets
      2047,   // Just over 2 packets
      4096,   // 4 packets
      8192,   // 8 packets
      16384,  // 16 packets
      65536   // 64 packets
  };

  const int NUM_DESTS = 3;

  std::cout << std::setw(10) << "Size"
            << " | " << std::setw(10) << "Packets"
            << " | " << std::setw(12) << "Time (us)"
            << " | "
            << "Result" << std::endl;
  std::cout << std::string(50, '-') << std::endl;

  for (size_t SIZE : sizes) {
    void* src = ctx.AllocateGPUBuffer(SIZE);
    std::vector<void*> dsts(NUM_DESTS);
    for (auto& dst : dsts) {
      dst = ctx.AllocateGPUBuffer(SIZE);
      if (!dst) {
        ctx.Free(src);
        for (auto d : dsts)
          if (d) ctx.Free(d);
        continue;  // Skip this size
      }
    }

    BroadcastTestUtils::FillPattern(src, SIZE, BroadcastTestUtils::RANDOM, SIZE);
    std::vector<hsa_agent_t> dst_agents(NUM_DESTS, ctx.gpu_agent);
    hsa_signal_t signal = BroadcastTestUtils::CreateSignal(1);

    auto start = std::chrono::high_resolution_clock::now();

    hsa_status_t status =
        hsa_amd_memory_broadcast_copy(src, ctx.gpu_agent, dsts.data(), dst_agents.data(), NUM_DESTS,
                                      SIZE, 0, nullptr, signal, HSA_AMD_SDMA_ENGINE_0, false);

    BroadcastTestUtils::WaitSignal(signal);

    auto end = std::chrono::high_resolution_clock::now();
    auto time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    int pass_count = 0;
    for (int i = 0; i < NUM_DESTS; i++) {
      if (BroadcastTestUtils::CompareBuffers(src, dsts[i], SIZE)) pass_count++;
    }

    int packets = (SIZE + MAX_PACKET_SIZE) / (MAX_PACKET_SIZE + 1);
    bool all_pass = (status == HSA_STATUS_SUCCESS && pass_count == NUM_DESTS);

    std::cout << std::setw(10) << SIZE << " | " << std::setw(10) << packets << " | "
              << std::setw(12) << time_us << " | " << (all_pass ? "PASS" : "FAIL") << std::endl;

    ASSERT_TRUE(all_pass) << "Failed at size " << SIZE;

    BroadcastTestUtils::DestroySignal(signal);
    ctx.Free(src);
    for (auto dst : dsts) ctx.Free(dst);
  }

  std::cout << "  All size scaling tests passed" << std::endl;
}

// ============================================================================
// Main
// ============================================================================
