// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// ROCrtst Level 6 Tests: Stress & Concurrent Testing
// Purpose: Long-running stability and multi-threaded concurrency

#include <gtest/gtest.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <iostream>
#include <iomanip>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>
#include "../../common/broadcast_copy_utils.h"

class BroadcastCopyL6 : public ::testing::Test {
 protected:
  void SetUp() override {}
};

//
// TC-L6-001: 1000 Iterations Stability Test
//

TEST_F(BroadcastCopyL6, StabilityTest_1000_Iterations) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  const size_t SIZE = 16384;
  const int NUM_DESTS = 4;
  const int ITERATIONS = 1000;

  std::cout << "[TC-L6-001] Stability test: " << ITERATIONS << " iterations" << std::endl;
  std::cout << "  Size: " << SIZE << " bytes, Destinations: " << NUM_DESTS << std::endl;

  void* src = ctx.AllocateGPUBuffer(SIZE);
  std::vector<void*> dsts(NUM_DESTS);
  for (auto& dst : dsts) dst = ctx.AllocateGPUBuffer(SIZE);

  std::vector<hsa_agent_t> dst_agents(NUM_DESTS, ctx.gpu_agent);
  hsa_signal_t signal = BroadcastTestUtils::CreateSignal(1);

  int pass_count = 0;
  int fail_count = 0;
  auto start_time = std::chrono::high_resolution_clock::now();

  for (int iter = 0; iter < ITERATIONS; iter++) {
    // Vary pattern with iteration
    BroadcastTestUtils::Pattern pattern = static_cast<BroadcastTestUtils::Pattern>(iter % 7);
    BroadcastTestUtils::FillPattern(src, SIZE, pattern, iter);

    hsa_signal_store_screlease(signal, 1);

    hsa_status_t status =
        hsa_amd_memory_broadcast_copy(src, ctx.gpu_agent, dsts.data(), dst_agents.data(), NUM_DESTS,
                                      SIZE, 0, nullptr, signal, HSA_AMD_SDMA_ENGINE_0, false);

    if (status != HSA_STATUS_SUCCESS) {
      std::cout << "  ❌ Iteration " << iter << " failed with status " << (int)status << std::endl;
      fail_count++;
      continue;
    }

    BroadcastTestUtils::WaitSignal(signal);

    // Verify all destinations
    bool all_valid = true;
    for (int i = 0; i < NUM_DESTS; i++) {
      if (!BroadcastTestUtils::VerifyPattern(dsts[i], SIZE, pattern, iter)) {
        std::cout << "  ❌ Iteration " << iter << ", dst[" << i << "] verification failed"
                  << std::endl;
        all_valid = false;
        break;
      }
    }

    if (all_valid) {
      pass_count++;
    } else {
      fail_count++;
    }

    if ((iter + 1) % 100 == 0) {
      std::cout << "  Progress: " << (iter + 1) << "/" << ITERATIONS << " (" << pass_count
                << " pass, " << fail_count << " fail)" << std::endl;
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto total_time_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

  std::cout << "  Total time: " << total_time_ms << " ms (" << (total_time_ms / ITERATIONS)
            << " ms/iter avg)" << std::endl;
  std::cout << "  Results: " << pass_count << " pass, " << fail_count << " fail" << std::endl;

  double success_rate = (100.0 * pass_count) / ITERATIONS;
  std::cout << "  Success rate: " << std::fixed << std::setprecision(2) << success_rate << "%"
            << std::endl;

  ASSERT_EQ(ITERATIONS, pass_count) << "Stability test had failures";
  std::cout << "  ✓ 1000-iteration stability test passed" << std::endl;

  BroadcastTestUtils::DestroySignal(signal);
  ctx.Free(src);
  for (auto dst : dsts) ctx.Free(dst);
}

//
// TC-L6-002: Concurrent Broadcasts (Multi-Threaded)
//

void BroadcastWorkerThread(int thread_id, hsa_agent_t gpu_agent, std::atomic<int>& pass_count,
                           std::atomic<int>& fail_count) {
  const size_t SIZE = 8192;
  const int NUM_DESTS = 4;
  const int ITERS_PER_THREAD = 50;

  for (int iter = 0; iter < ITERS_PER_THREAD; iter++) {
    void* src = nullptr;
    std::vector<void*> dsts(NUM_DESTS);
    hsa_signal_t signal;

    hsa_status_t status =
        hsa_memory_allocate(BroadcastTestUtils::FindDefaultRegion(gpu_agent), SIZE, &src);

    if (status != HSA_STATUS_SUCCESS) {
      fail_count++;
      continue;
    }

    bool alloc_success = true;
    for (int i = 0; i < NUM_DESTS; i++) {
      status =
          hsa_memory_allocate(BroadcastTestUtils::FindDefaultRegion(gpu_agent), SIZE, &dsts[i]);
      if (status != HSA_STATUS_SUCCESS) {
        alloc_success = false;
        break;
      }
    }

    if (!alloc_success) {
      hsa_memory_free(src);
      for (auto dst : dsts)
        if (dst) hsa_memory_free(dst);
      fail_count++;
      continue;
    }

    BroadcastTestUtils::FillPattern(src, SIZE, BroadcastTestUtils::INCREMENTAL,
                                    thread_id * 1000 + iter);
    std::vector<hsa_agent_t> dst_agents(NUM_DESTS, gpu_agent);

    status = hsa_signal_create(1, 0, nullptr, &signal);
    if (status != HSA_STATUS_SUCCESS) {
      fail_count++;
      continue;
    }

    status =
        hsa_amd_memory_broadcast_copy(src, gpu_agent, dsts.data(), dst_agents.data(), NUM_DESTS,
                                      SIZE, 0, nullptr, signal, HSA_AMD_SDMA_ENGINE_0, false);

    if (status != HSA_STATUS_SUCCESS) {
      fail_count++;
    } else {
      BroadcastTestUtils::WaitSignal(signal);

      bool all_valid = true;
      for (int i = 0; i < NUM_DESTS; i++) {
        if (!BroadcastTestUtils::VerifyPattern(dsts[i], SIZE, BroadcastTestUtils::INCREMENTAL,
                                               thread_id * 1000 + iter)) {
          all_valid = false;
          break;
        }
      }

      if (all_valid) {
        pass_count++;
      } else {
        fail_count++;
      }
    }

    hsa_signal_destroy(signal);
    hsa_memory_free(src);
    for (auto dst : dsts) hsa_memory_free(dst);
  }
}

TEST_F(BroadcastCopyL6, ConcurrentBroadcasts_4_Threads) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  const int NUM_THREADS = 4;
  const int ITERS_PER_THREAD = 50;

  std::cout << "[TC-L6-002] Concurrent broadcasts with " << NUM_THREADS << " threads" << std::endl;
  std::cout << "  Each thread: " << ITERS_PER_THREAD << " iterations" << std::endl;

  std::atomic<int> pass_count{0};
  std::atomic<int> fail_count{0};
  std::vector<std::thread> threads;

  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < NUM_THREADS; i++) {
    threads.emplace_back(BroadcastWorkerThread, i, ctx.gpu_agent, std::ref(pass_count),
                         std::ref(fail_count));
  }

  for (auto& t : threads) {
    t.join();
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

  int expected = NUM_THREADS * ITERS_PER_THREAD;
  std::cout << "  Completed in " << time_ms << " ms" << std::endl;
  std::cout << "  Results: " << pass_count << " pass, " << fail_count << " fail (expected "
            << expected << " total)" << std::endl;

  ASSERT_EQ(expected, pass_count.load()) << "Some concurrent broadcasts failed";
  std::cout << "  ✓ Concurrent broadcasts successful" << std::endl;
}

//
// TC-L6-003: Memory Pressure Test (Allocate Until Failure)
//

TEST_F(BroadcastCopyL6, MemoryPressureTest) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  std::cout << "[TC-L6-003] Memory pressure test (allocate until low memory)" << std::endl;

  const size_t BUFFER_SIZE = 1048576;  // 1MB per buffer
  const int NUM_DESTS = 4;
  std::vector<void*> pressure_buffers;

  // Allocate until we approach memory limits
  std::cout << "  Allocating pressure buffers..." << std::endl;
  for (int i = 0; i < 100; i++) {  // Max 100MB pressure
    void* buf = nullptr;
    hsa_status_t status = hsa_memory_allocate(BroadcastTestUtils::FindDefaultRegion(ctx.gpu_agent),
                                              BUFFER_SIZE, &buf);

    if (status != HSA_STATUS_SUCCESS || !buf) {
      std::cout << "  Reached memory limit after " << i << " buffers ("
                << (i * BUFFER_SIZE / 1048576) << " MB)" << std::endl;
      break;
    }

    pressure_buffers.push_back(buf);

    if ((i + 1) % 10 == 0) {
      std::cout << "    Allocated " << (i + 1) << " buffers (" << ((i + 1) * BUFFER_SIZE / 1048576)
                << " MB)..." << std::endl;
    }
  }

  // Now try broadcast copy under memory pressure
  const size_t SIZE = 16384;
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
    std::cout << "  ⚠ Could not allocate test buffers under pressure (expected)" << std::endl;
    // Clean up pressure
    for (auto buf : pressure_buffers) hsa_memory_free(buf);
    GTEST_SKIP() << "Insufficient memory for test under pressure";
  }

  BroadcastTestUtils::FillPattern(src, SIZE, BroadcastTestUtils::RANDOM, 0xABCDEF);
  std::vector<hsa_agent_t> dst_agents(NUM_DESTS, ctx.gpu_agent);
  hsa_signal_t signal = BroadcastTestUtils::CreateSignal(1);

  std::cout << "  Executing broadcast copy under memory pressure..." << std::endl;

  hsa_status_t status =
      hsa_amd_memory_broadcast_copy(src, ctx.gpu_agent, dsts.data(), dst_agents.data(), NUM_DESTS,
                                    SIZE, 0, nullptr, signal, HSA_AMD_SDMA_ENGINE_0, false);

  ASSERT_EQ(HSA_STATUS_SUCCESS, status) << "Broadcast failed under memory pressure";
  BroadcastTestUtils::WaitSignal(signal);

  bool all_valid = true;
  for (int i = 0; i < NUM_DESTS; i++) {
    if (!BroadcastTestUtils::VerifyPattern(dsts[i], SIZE, BroadcastTestUtils::RANDOM, 0xABCDEF)) {
      all_valid = false;
      break;
    }
  }

  ASSERT_TRUE(all_valid) << "Data corruption under memory pressure";
  std::cout << "  ✓ Broadcast successful under memory pressure" << std::endl;

  // Cleanup
  BroadcastTestUtils::DestroySignal(signal);
  ctx.Free(src);
  for (auto dst : dsts) ctx.Free(dst);
  for (auto buf : pressure_buffers) hsa_memory_free(buf);
}

//
// TC-L6-004: Rapid Sequential Broadcasts (Stress SDMA Queue)
//

TEST_F(BroadcastCopyL6, RapidSequentialBroadcasts) {
  HsaTestContext ctx;
  if (!ctx.HasGPUAgent()) GTEST_SKIP() << "No GPU agent";

  const size_t SIZE = 4096;
  const int NUM_DESTS = 4;
  const int NUM_RAPID = 100;

  std::cout << "[TC-L6-004] Rapid sequential broadcasts (" << NUM_RAPID << " back-to-back)"
            << std::endl;

  void* src = ctx.AllocateGPUBuffer(SIZE);
  std::vector<void*> dsts(NUM_DESTS);
  for (auto& dst : dsts) dst = ctx.AllocateGPUBuffer(SIZE);

  std::vector<hsa_signal_t> signals(NUM_RAPID);
  for (auto& sig : signals) sig = BroadcastTestUtils::CreateSignal(1);

  std::vector<hsa_agent_t> dst_agents(NUM_DESTS, ctx.gpu_agent);
  BroadcastTestUtils::FillPattern(src, SIZE, BroadcastTestUtils::WALKING_BIT);

  std::cout << "  Submitting " << NUM_RAPID << " broadcasts..." << std::endl;
  auto start = std::chrono::high_resolution_clock::now();

  // Rapid-fire submission (do NOT wait)
  for (int i = 0; i < NUM_RAPID; i++) {
    hsa_status_t status =
        hsa_amd_memory_broadcast_copy(src, ctx.gpu_agent, dsts.data(), dst_agents.data(), NUM_DESTS,
                                      SIZE, 0, nullptr, signals[i], HSA_AMD_SDMA_ENGINE_0, false);

    ASSERT_EQ(HSA_STATUS_SUCCESS, status) << "Rapid broadcast #" << i << " failed";
  }

  // Now wait for all
  std::cout << "  Waiting for all " << NUM_RAPID << " to complete..." << std::endl;
  for (int i = 0; i < NUM_RAPID; i++) {
    BroadcastTestUtils::WaitSignal(signals[i]);
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

  std::cout << "  Completed in " << time_ms << " ms (" << (time_ms / NUM_RAPID)
            << " ms/broadcast avg)" << std::endl;

  // Verify final state
  bool all_valid = true;
  for (int i = 0; i < NUM_DESTS; i++) {
    if (!BroadcastTestUtils::VerifyPattern(dsts[i], SIZE, BroadcastTestUtils::WALKING_BIT)) {
      all_valid = false;
      break;
    }
  }

  ASSERT_TRUE(all_valid) << "Data corruption in rapid sequential test";
  std::cout << "  ✓ All " << NUM_RAPID << " broadcasts completed successfully" << std::endl;

  for (auto sig : signals) BroadcastTestUtils::DestroySignal(sig);
  ctx.Free(src);
  for (auto dst : dsts) ctx.Free(dst);
}
