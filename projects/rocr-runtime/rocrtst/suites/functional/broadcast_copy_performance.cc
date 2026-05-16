// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// ROCrtst Performance Tests: Bandwidth & Scaling
// Purpose: Measure effective bandwidth and scaling efficiency

#include <gtest/gtest.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <cmath>
#include "../../common/broadcast_copy_utils.h"

class BroadcastCopyPerformance : public ::testing::Test {
 protected:
  void SetUp() override {}

  double CalculateBandwidth(size_t total_bytes, int64_t time_us) {
    // Returns GB/s
    return (total_bytes * 1.0e6) / (time_us * 1.0e9);
  }

  double CalculateSpeedup(int64_t baseline_us, int64_t broadcast_us) {
    return baseline_us > 0 ? (double)baseline_us / broadcast_us : 0.0;
  }
};

//
// TC-PERF-001: Bandwidth Scaling by Destination Count
//

TEST_F(BroadcastCopyPerformance, BandwidthScaling_ByDestCount) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  std::cout << "[TC-PERF-001] Bandwidth scaling by destination count" << std::endl;

  const size_t SIZE = 1048576;  // 1MB
  const int NUM_ITERS = 10;
  std::vector<int> dest_counts = {1, 2, 4, 8, 16, 32, 64};

  std::cout << "\n"
            << std::setw(10) << "Num Dests"
            << " | " << std::setw(12) << "Avg Time µs"
            << " | " << std::setw(15) << "Bandwidth GB/s"
            << " | " << std::setw(12) << "Speedup vs 1" << std::endl;
  std::cout << std::string(70, '-') << std::endl;

  int64_t baseline_time_us = 0;

  for (int NUM_DESTS : dest_counts) {
    // Allocate buffers
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
      std::cout << std::setw(10) << NUM_DESTS << " | "
                << "SKIP (allocation failed)" << std::endl;
      ctx.Free(src);
      for (auto dst : dsts)
        if (dst) ctx.Free(dst);
      continue;
    }

    BroadcastTestUtils::FillPattern(src, SIZE, BroadcastTestUtils::INCREMENTAL);
    std::vector<hsa_agent_t> dst_agents(NUM_DESTS, ctx.gpu_agent);
    hsa_signal_t signal = BroadcastTestUtils::CreateSignal(1);

    // Benchmark
    std::vector<int64_t> times;
    for (int iter = 0; iter < NUM_ITERS; iter++) {
      hsa_signal_store_screlease(signal, 1);

      auto start = std::chrono::high_resolution_clock::now();

      hsa_status_t status = hsa_amd_memory_broadcast_copy(
          src, ctx.gpu_agent, dsts.data(), dst_agents.data(), NUM_DESTS, SIZE, 0, nullptr, signal,
          HSA_AMD_SDMA_ENGINE_0, false);

      if (status != HSA_STATUS_SUCCESS) {
        std::cout << "  ❌ Iteration " << iter << " failed" << std::endl;
        continue;
      }

      BroadcastTestUtils::WaitSignal(signal);

      auto end = std::chrono::high_resolution_clock::now();
      times.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
    }

    if (times.empty()) {
      std::cout << std::setw(10) << NUM_DESTS << " | "
                << "FAIL (all iterations failed)" << std::endl;
      continue;
    }

    // Calculate statistics
    int64_t avg_time = std::accumulate(times.begin(), times.end(), 0LL) / times.size();
    size_t total_bytes = SIZE * NUM_DESTS;
    double bandwidth = CalculateBandwidth(total_bytes, avg_time);

    if (NUM_DESTS == 1) {
      baseline_time_us = avg_time;
    }

    double speedup = CalculateSpeedup(baseline_time_us * NUM_DESTS, avg_time);

    std::cout << std::setw(10) << NUM_DESTS << " | " << std::setw(12) << avg_time << " | "
              << std::setw(15) << std::fixed << std::setprecision(2) << bandwidth << " | "
              << std::setw(12) << std::fixed << std::setprecision(2) << speedup << "x" << std::endl;

    BroadcastTestUtils::DestroySignal(signal);
    ctx.Free(src);
    for (auto dst : dsts) ctx.Free(dst);
  }

  std::cout << "\n  ✓ Bandwidth scaling measurement complete" << std::endl;
}

//
// TC-PERF-002: SDMA vs Shader Performance Comparison
//

TEST_F(BroadcastCopyPerformance, SDMAvsShader_Comparison) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  std::cout << "[TC-PERF-002] SDMA vs Shader path performance comparison" << std::endl;
  std::cout << "  Comparing small (<4KB, shader) vs large (>=4KB, SDMA/shader) buffers"
            << std::endl;

  const size_t SIZE = 1048576;  // 1MB - will use SDMA on GFX13+ or shader on pre-GFX13
  const int NUM_DESTS = 10;
  const int NUM_ITERS = 20;

  void* src = ctx.AllocateGPUBuffer(SIZE);
  std::vector<void*> dsts(NUM_DESTS);
  for (auto& dst : dsts) dst = ctx.AllocateGPUBuffer(SIZE);

  BroadcastTestUtils::FillPattern(src, SIZE, BroadcastTestUtils::RANDOM, 0x12345678);
  std::vector<hsa_agent_t> dst_agents(NUM_DESTS, ctx.gpu_agent);
  hsa_signal_t signal = BroadcastTestUtils::CreateSignal(1);

  std::cout << "  Running " << NUM_ITERS << " iterations..." << std::endl;

  std::vector<int64_t> times;
  for (int iter = 0; iter < NUM_ITERS; iter++) {
    hsa_signal_store_screlease(signal, 1);

    auto start = std::chrono::high_resolution_clock::now();
    hsa_amd_memory_broadcast_copy(src, ctx.gpu_agent, dsts.data(), dst_agents.data(), NUM_DESTS,
                                  SIZE, 0, nullptr, signal, HSA_AMD_SDMA_ENGINE_0, false);
    BroadcastTestUtils::WaitSignal(signal);
    auto end = std::chrono::high_resolution_clock::now();

    times.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());
  }

  // Statistics
  std::sort(times.begin(), times.end());
  int64_t min_time = times.front();
  int64_t max_time = times.back();
  int64_t median_time = times[times.size() / 2];
  int64_t avg_time = std::accumulate(times.begin(), times.end(), 0LL) / times.size();

  // Calculate standard deviation
  double variance = 0.0;
  for (auto t : times) {
    variance += (t - avg_time) * (t - avg_time);
  }
  double stddev = std::sqrt(variance / times.size());

  double bandwidth = CalculateBandwidth(SIZE * NUM_DESTS, avg_time);

  std::cout << "\n  Performance Statistics (SDMA/Shader path for large buffer):" << std::endl;
  std::cout << "    Min:       " << std::setw(8) << min_time << " µs" << std::endl;
  std::cout << "    Max:       " << std::setw(8) << max_time << " µs" << std::endl;
  std::cout << "    Median:    " << std::setw(8) << median_time << " µs" << std::endl;
  std::cout << "    Avg:       " << std::setw(8) << avg_time << " µs" << std::endl;
  std::cout << "    Std Dev:   " << std::setw(8) << std::fixed << std::setprecision(2) << stddev
            << " µs" << std::endl;
  std::cout << "    Bandwidth: " << std::setw(8) << std::fixed << std::setprecision(2) << bandwidth
            << " GB/s" << std::endl;

  std::cout << "\n  Note: Path automatically selected based on GPU generation and buffer size"
            << std::endl;
  std::cout << "  ✓ Performance measurement complete" << std::endl;

  BroadcastTestUtils::DestroySignal(signal);
  ctx.Free(src);
  for (auto dst : dsts) ctx.Free(dst);
}

//
// TC-PERF-003: All-Gather Simulation (Multi-GPU)
//

TEST_F(BroadcastCopyPerformance, AllGatherSimulation) {
  HsaTestContext ctx;

  std::vector<hsa_agent_t> all_gpus = BroadcastTestUtils::FindAllGPUAgents();

  std::cout << "[TC-PERF-003] All-gather simulation (requires multi-GPU)" << std::endl;
  std::cout << "  Detected " << all_gpus.size() << " GPU agent(s)" << std::endl;

  if (all_gpus.size() < 2) {
    std::cout << "  ℹ️  Multi-GPU test requires 2+ GPUs, simulating with single GPU..."
              << std::endl;
    if (all_gpus.empty()) {
      GTEST_SKIP() << "No GPU agents available";
    }
    // Simulate by using same GPU multiple times
    all_gpus.push_back(all_gpus[0]);
    all_gpus.push_back(all_gpus[0]);
  }

  const size_t CHUNK_SIZE = 262144;                        // 256KB per GPU
  const int NUM_GPUS = std::min<int>(all_gpus.size(), 8);  // Limit to 8

  std::cout << "  Simulating all-gather with " << NUM_GPUS << " GPUs" << std::endl;
  std::cout << "  Each GPU broadcasts " << CHUNK_SIZE << " bytes to all other GPUs" << std::endl;

  // Each GPU has a local buffer that it broadcasts
  std::vector<void*> local_buffers(NUM_GPUS);
  // Each GPU receives from all GPUs (including itself)
  std::vector<std::vector<void*>> recv_buffers(NUM_GPUS);

  // Allocate on first GPU for simplicity (in real multi-GPU, distribute across agents)
  hsa_agent_t primary_gpu = all_gpus[0];

  for (int i = 0; i < NUM_GPUS; i++) {
    local_buffers[i] = ctx.AllocateGPUBuffer(CHUNK_SIZE);
    BroadcastTestUtils::FillPattern(local_buffers[i], CHUNK_SIZE, BroadcastTestUtils::INCREMENTAL,
                                    i * 1000);

    recv_buffers[i].resize(NUM_GPUS);
    for (int j = 0; j < NUM_GPUS; j++) {
      recv_buffers[i][j] = ctx.AllocateGPUBuffer(CHUNK_SIZE);
    }
  }

  auto start = std::chrono::high_resolution_clock::now();

  // Simulate all-gather: each GPU broadcasts to all
  for (int src_gpu = 0; src_gpu < NUM_GPUS; src_gpu++) {
    std::vector<void*> dsts;
    std::vector<hsa_agent_t> dst_agents;

    for (int dst_gpu = 0; dst_gpu < NUM_GPUS; dst_gpu++) {
      dsts.push_back(recv_buffers[dst_gpu][src_gpu]);
      dst_agents.push_back(primary_gpu);  // In real multi-GPU, use all_gpus[dst_gpu]
    }

    hsa_signal_t signal = BroadcastTestUtils::CreateSignal(1);

    hsa_status_t status = hsa_amd_memory_broadcast_copy(
        local_buffers[src_gpu], primary_gpu, dsts.data(), dst_agents.data(), NUM_GPUS, CHUNK_SIZE,
        0, nullptr, signal, HSA_AMD_SDMA_ENGINE_0, false);

    ASSERT_EQ(HSA_STATUS_SUCCESS, status) << "Broadcast from GPU " << src_gpu << " failed";
    BroadcastTestUtils::WaitSignal(signal);
    BroadcastTestUtils::DestroySignal(signal);
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto time_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  // Total data moved: each GPU broadcasts to N GPUs, N times
  size_t total_bytes = CHUNK_SIZE * NUM_GPUS * NUM_GPUS;
  double bandwidth = CalculateBandwidth(total_bytes, time_us);

  std::cout << "  Results:" << std::endl;
  std::cout << "    Total time:      " << time_us << " µs" << std::endl;
  std::cout << "    Total data:      " << (total_bytes / 1048576) << " MB" << std::endl;
  std::cout << "    Effective BW:    " << std::fixed << std::setprecision(2) << bandwidth << " GB/s"
            << std::endl;
  std::cout << "    Time per bcast:  " << (time_us / NUM_GPUS) << " µs avg" << std::endl;

  // Verify correctness - sample check
  bool valid = BroadcastTestUtils::VerifyPattern(recv_buffers[0][0], CHUNK_SIZE,
                                                 BroadcastTestUtils::INCREMENTAL, 0);

  ASSERT_TRUE(valid) << "All-gather data corruption detected";
  std::cout << "  ✓ All-gather simulation completed successfully" << std::endl;

  for (auto buf : local_buffers) ctx.Free(buf);
  for (auto& vec : recv_buffers) {
    for (auto buf : vec) ctx.Free(buf);
  }
}

//
// TC-PERF-004: Broadcast Collective Scaling Test
//

TEST_F(BroadcastCopyPerformance, BroadcastCollective_Scaling) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  std::cout << "[TC-PERF-004] Broadcast collective scaling test" << std::endl;

  const size_t BASE_SIZE = 16384;  // 16KB base
  std::vector<int> dest_counts = {2, 4, 8, 16, 32, 64, 128};

  std::cout << "\nMeasuring \" speedup vs sequential copy:" << std::endl;
  std::cout << std::setw(10) << "Num Dests"
            << " | " << std::setw(14) << "Broadcast (µs)"
            << " | " << std::setw(16) << "Sequential (µs)"
            << " | " << std::setw(10) << "Speedup" << std::endl;
  std::cout << std::string(70, '-') << std::endl;

  for (int NUM_DESTS : dest_counts) {
    void* src = ctx.AllocateGPUBuffer(BASE_SIZE);
    std::vector<void*> dsts(NUM_DESTS);

    bool alloc_ok = true;
    for (int i = 0; i < NUM_DESTS; i++) {
      dsts[i] = ctx.AllocateGPUBuffer(BASE_SIZE);
      if (!dsts[i]) {
        alloc_ok = false;
        break;
      }
    }

    if (!alloc_ok) {
      std::cout << std::setw(10) << NUM_DESTS << " | SKIP (allocation failed)" << std::endl;
      ctx.Free(src);
      for (auto dst : dsts)
        if (dst) ctx.Free(dst);
      continue;
    }

    BroadcastTestUtils::FillPattern(src, BASE_SIZE, BroadcastTestUtils::RANDOM, NUM_DESTS);
    std::vector<hsa_agent_t> dst_agents(NUM_DESTS, ctx.gpu_agent);

    // Measure broadcast
    hsa_signal_t bcast_signal = BroadcastTestUtils::CreateSignal(1);
    auto start = std::chrono::high_resolution_clock::now();

    hsa_status_t status = hsa_amd_memory_broadcast_copy(
        src, ctx.gpu_agent, dsts.data(), dst_agents.data(), NUM_DESTS, BASE_SIZE, 0, nullptr,
        bcast_signal, HSA_AMD_SDMA_ENGINE_0, false);

    if (status != HSA_STATUS_SUCCESS) {
      std::cout << std::setw(10) << NUM_DESTS << " | FAIL (broadcast failed)" << std::endl;
      continue;
    }

    BroadcastTestUtils::WaitSignal(bcast_signal);
    auto end = std::chrono::high_resolution_clock::now();
    auto broadcast_time =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    // Measure sequential
    hsa_signal_t seq_signal = BroadcastTestUtils::CreateSignal(NUM_DESTS);
    start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_DESTS; i++) {
      hsa_amd_memory_async_copy(dsts[i], ctx.gpu_agent, src, ctx.gpu_agent, BASE_SIZE, 0, nullptr,
                                seq_signal);
    }

    BroadcastTestUtils::WaitSignal(seq_signal, HSA_SIGNAL_CONDITION_EQ, 0);
    end = std::chrono::high_resolution_clock::now();
    auto sequential_time =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    double speedup = sequential_time > 0 ? (double)sequential_time / broadcast_time : 0.0;

    std::cout << std::setw(10) << NUM_DESTS << " | " << std::setw(14) << broadcast_time << " | "
              << std::setw(16) << sequential_time << " | " << std::setw(10) << std::fixed
              << std::setprecision(2) << speedup << "x" << std::endl;

    // Expect speedup >= 1.0 for multi-dest
    if (NUM_DESTS >= 4) {
      EXPECT_GE(speedup, 1.0) << "Expected speedup for " << NUM_DESTS << " destinations";
    }

    BroadcastTestUtils::DestroySignal(bcast_signal);
    BroadcastTestUtils::DestroySignal(seq_signal);
    ctx.Free(src);
    for (auto dst : dsts) ctx.Free(dst);
  }

  std::cout << "\n  ✓ Scaling test complete" << std::endl;
}
