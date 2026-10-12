// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/device_cache_coherence.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l1_scalar_cache.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <barrier>
#include <cstring>
#include <latch>
#include <memory>
#include <span>
#include <string>
#include <thread>
#include <vector>

namespace {
using namespace rocjitsu::amdgpu;

TEST(L2ShardEpochTest, EightSharedCachesPublishDirtyLinesAndRefreshConcurrentReaders) {
  constexpr size_t kCaches = 8;
  constexpr uint32_t kAtomicIterations = 128;
  constexpr uint32_t kReadIterations = 512;
  constexpr uint64_t kCounter = 0x540000;
  GpuMemory memory("shard_epoch_memory");
  auto coherence = std::make_shared<DeviceCacheCoherence>();
  std::array<std::unique_ptr<L2Cache>, kCaches> l2s;
  std::array<std::unique_ptr<L1ScalarCache>, kCaches> readers;
  memory.write32(kCounter, 0);
  for (size_t i = 0; i < kCaches; ++i) {
    l2s[i] = std::make_unique<L2Cache>("shard_epoch_l2_" + std::to_string(i), coherence);
    l2s[i]->set_backing_memory(&memory);
    readers[i] = std::make_unique<L1ScalarCache>(l2s[i].get());
    uint32_t initially_cached = 1;
    ASSERT_EQ(readers[i]->load(kCounter, 1, &initially_cached), VmAccessOutcome::Complete);
    ASSERT_EQ(initially_cached, 0u);
    std::array<uint8_t, L2Cache::LINE_SIZE> dirty;
    dirty.fill(static_cast<uint8_t>(i + 1));
    const uint64_t address = kCounter + (i + 1) * L2Cache::LINE_SIZE;
    ASSERT_EQ(l2s[i]->writeback_line(address, dirty.data()), VmAccessOutcome::Complete);
    // Keep the dirty line unpublished so the later atomic must flush it.
    ASSERT_EQ(memory.read32(address), 0u);
  }

  std::barrier start(kCaches + 1);
  std::latch first_atomic_complete(1), all_atomics_complete(1);
  std::atomic<unsigned> failures = 0;
  std::vector<std::jthread> threads;
  for (size_t i = 0; i < kCaches; ++i) {
    threads.emplace_back([&, i] {
      start.arrive_and_wait();
      first_atomic_complete.wait();
      uint32_t previous = 0;
      for (uint32_t n = 0; n < kReadIterations; ++n) {
        uint32_t observed = 0;
        if (readers[i]->load(kCounter, 1, &observed) != VmAccessOutcome::Complete ||
            observed < previous || observed == 0 || observed > kAtomicIterations)
          failures.fetch_add(1, std::memory_order_relaxed);
        previous = observed;
        if ((n & 15) == 0)
          std::this_thread::yield();
      }
      all_atomics_complete.wait();
      uint32_t final_value = 0;
      if (readers[i]->load(kCounter, 1, &final_value) != VmAccessOutcome::Complete ||
          final_value != kAtomicIterations)
        failures.fetch_add(1, std::memory_order_relaxed);
    });
  }
  start.arrive_and_wait();
  for (uint32_t n = 0; n < kAtomicIterations; ++n) {
    const auto outcome = l2s[n % kCaches]->atomic_rmw(
        kCounter, sizeof(uint32_t), [](uint8_t *line, uint32_t offset) {
          uint32_t value;
          std::memcpy(&value, line + offset, sizeof(value));
          ++value;
          std::memcpy(line + offset, &value, sizeof(value));
        });
    if (outcome != VmAccessOutcome::Complete)
      failures.fetch_add(1, std::memory_order_relaxed);
    if (n == 0)
      first_atomic_complete.count_down();
    std::this_thread::yield();
  }
  all_atomics_complete.count_down();
  threads.clear();

  EXPECT_EQ(failures.load(), 0u);
  EXPECT_EQ(memory.read32(kCounter), kAtomicIterations);
  // Device atomics must have published dirty state from every registered L2.
  for (size_t i = 0; i < kCaches; ++i) {
    std::array<uint8_t, L2Cache::LINE_SIZE> expected, actual{};
    expected.fill(static_cast<uint8_t>(i + 1));
    memory.read_block(kCounter + (i + 1) * L2Cache::LINE_SIZE, std::span<uint8_t>(actual));
    EXPECT_EQ(actual, expected) << "cache=" << i;
  }
}
TEST(L2ShardEpochTest, ConcurrentReadersRefreshOneCacheAfterEveryAtomicEpoch) {
  constexpr size_t kReaders = 32;
  constexpr uint32_t kEpochs = 32;
  constexpr uint64_t kCounter = 0x640000;
  GpuMemory memory("shared_epoch_memory");
  L2Cache l2("shared_epoch_l2");
  l2.set_backing_memory(&memory);
  memory.write32(kCounter, 0);
  uint32_t initial = 1;
  ASSERT_EQ(l2.read(kCounter, reinterpret_cast<uint8_t *>(&initial), sizeof(initial)),
            VmAccessOutcome::Complete);
  ASSERT_EQ(initial, 0u);
  std::barrier begin(kReaders + 1), done(kReaders + 1);
  std::atomic<unsigned> failures = 0;
  std::vector<std::jthread> readers;
  for (size_t i = 0; i < kReaders; ++i) {
    readers.emplace_back([&] {
      for (uint32_t epoch = 1; epoch <= kEpochs; ++epoch) {
        begin.arrive_and_wait();
        uint32_t observed = 0;
        if (l2.read(kCounter, reinterpret_cast<uint8_t *>(&observed), sizeof(observed)) !=
                VmAccessOutcome::Complete ||
            observed != epoch)
          failures.fetch_add(1, std::memory_order_relaxed);
        done.arrive_and_wait();
      }
    });
  }
  for (uint32_t epoch = 1; epoch <= kEpochs; ++epoch) {
    const auto outcome =
        l2.atomic_rmw(kCounter, sizeof(uint32_t), [](uint8_t *line, uint32_t offset) {
          uint32_t value;
          std::memcpy(&value, line + offset, sizeof(value));
          ++value;
          std::memcpy(line + offset, &value, sizeof(value));
        });
    EXPECT_EQ(outcome, VmAccessOutcome::Complete);
    begin.arrive_and_wait();
    done.arrive_and_wait();
  }
  readers.clear();
  EXPECT_EQ(failures.load(), 0u);
}
} // namespace
