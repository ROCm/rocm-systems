// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <thread>
#include <vector>

namespace {

using rocjitsu::amdgpu::GpuMemory;
using rocjitsu::amdgpu::L2Cache;
using rocjitsu::amdgpu::Mtype;

TEST(L2CacheThreadingTest, ConcurrentDifferentSetWritesArePreserved) {
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);

  constexpr uint32_t kThreads = 8;
  constexpr uint32_t kLinesPerThread = 128;
  constexpr uint64_t kBase = 0x100000;

  std::barrier start(kThreads);
  std::vector<std::thread> workers;
  workers.reserve(kThreads);

  for (uint32_t tid = 0; tid < kThreads; ++tid) {
    workers.emplace_back([&, tid] {
      std::array<uint8_t, L2Cache::LINE_SIZE> line{};
      start.arrive_and_wait();
      for (uint32_t i = 0; i < kLinesPerThread; ++i) {
        const uint64_t addr =
            kBase + (static_cast<uint64_t>(i) * kThreads + tid) * L2Cache::LINE_SIZE;
        for (uint32_t b = 0; b < line.size(); ++b)
          line[b] = static_cast<uint8_t>((tid << 4) ^ i ^ b);
        l2.write(addr, line.data(), line.size());
      }
    });
  }

  for (auto &worker : workers)
    worker.join();

  std::array<uint8_t, L2Cache::LINE_SIZE> expected{};
  std::array<uint8_t, L2Cache::LINE_SIZE> actual{};
  for (uint32_t tid = 0; tid < kThreads; ++tid) {
    for (uint32_t i = 0; i < kLinesPerThread; ++i) {
      const uint64_t addr =
          kBase + (static_cast<uint64_t>(i) * kThreads + tid) * L2Cache::LINE_SIZE;
      for (uint32_t b = 0; b < expected.size(); ++b)
        expected[b] = static_cast<uint8_t>((tid << 4) ^ i ^ b);
      l2.read(addr, actual.data(), actual.size());
      EXPECT_EQ(actual, expected) << "addr=0x" << std::hex << addr;
    }
  }
}

TEST(L2CacheThreadingTest, ConcurrentSameSetAccessesPreserveLines) {
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);

  constexpr uint32_t kThreads = 8;
  constexpr uint32_t kIterations = 256;
  constexpr uint64_t kBase = 0x180000;
  constexpr uint64_t kSetStride = static_cast<uint64_t>(L2Cache::NUM_SETS) * L2Cache::LINE_SIZE;

  std::barrier start(kThreads);
  std::atomic<uint64_t> mismatches{0};
  std::vector<std::thread> workers;
  workers.reserve(kThreads);
  for (uint32_t tid = 0; tid < kThreads; ++tid) {
    workers.emplace_back([&, tid] {
      const uint64_t addr = kBase + tid * kSetStride;
      std::array<uint8_t, L2Cache::LINE_SIZE> line{};
      std::array<uint8_t, L2Cache::LINE_SIZE> actual{};
      start.arrive_and_wait();
      for (uint32_t iteration = 0; iteration < kIterations; ++iteration) {
        line.fill(static_cast<uint8_t>((tid << 4) ^ iteration));
        l2.write(addr, line.data(), line.size());
        l2.read(addr, actual.data(), actual.size());
        if (actual != line)
          mismatches.fetch_add(1, std::memory_order_relaxed);
      }
    });
  }

  for (auto &worker : workers)
    worker.join();

  EXPECT_EQ(mismatches.load(std::memory_order_relaxed), 0u);
}

TEST(L2CacheThreadingTest, ConcurrentAtomicRmwSameLineIsSerialized) {
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);

  constexpr uint32_t kThreads = 8;
  constexpr uint32_t kIterations = 1000;
  constexpr uint64_t kTarget = 0x200000;

  memory.write32(kTarget, 0);

  std::barrier start(kThreads);
  std::vector<std::thread> workers;
  workers.reserve(kThreads);

  for (uint32_t tid = 0; tid < kThreads; ++tid) {
    workers.emplace_back([&] {
      start.arrive_and_wait();
      for (uint32_t i = 0; i < kIterations; ++i) {
        l2.atomic_rmw(kTarget, sizeof(uint32_t), [](uint8_t *line, uint32_t offset) {
          uint32_t value = 0;
          std::memcpy(&value, line + offset, sizeof(value));
          ++value;
          std::memcpy(line + offset, &value, sizeof(value));
        });
      }
    });
  }

  for (auto &worker : workers)
    worker.join();

  EXPECT_EQ(memory.read32(kTarget), kThreads * kIterations);
}

TEST(L2CacheThreadingTest, CrossL2AtomicRmwSameAddressIsSerialized) {
  GpuMemory memory("memory");
  L2Cache l2a("l2a");
  L2Cache l2b("l2b");
  l2a.set_backing_memory(&memory);
  l2b.set_backing_memory(&memory);

  constexpr uint32_t kThreads = 8;
  constexpr uint32_t kIterations = 1000;
  constexpr uint64_t kTarget = 0x280000;

  memory.write32(kTarget, 0);

  std::array<L2Cache *, 2> l2s = {&l2a, &l2b};
  std::barrier start(kThreads);
  std::vector<std::thread> workers;
  workers.reserve(kThreads);

  for (uint32_t tid = 0; tid < kThreads; ++tid) {
    workers.emplace_back([&, tid] {
      auto *l2 = l2s[tid % l2s.size()];
      start.arrive_and_wait();
      for (uint32_t i = 0; i < kIterations; ++i) {
        l2->atomic_rmw(kTarget, sizeof(uint32_t), [](uint8_t *line, uint32_t offset) {
          uint32_t value = 0;
          std::memcpy(&value, line + offset, sizeof(value));
          ++value;
          std::memcpy(line + offset, &value, sizeof(value));
        });
      }
    });
  }

  for (auto &worker : workers)
    worker.join();

  EXPECT_EQ(memory.read32(kTarget), kThreads * kIterations);
}

TEST(L2CacheThreadingTest, CrossL2AtomicRmwAliasedVasIsSerialized) {
  GpuMemory memory("memory");
  L2Cache l2a("l2a");
  L2Cache l2b("l2b");
  l2a.set_backing_memory(&memory);
  l2b.set_backing_memory(&memory);

  constexpr uint32_t kVmidA = 7;
  constexpr uint32_t kVmidB = 8;
  constexpr uint64_t kVaA = 0x100000;
  constexpr uint64_t kVaB = 0x201000;
  constexpr uint64_t kOffset = 64;
  constexpr uint32_t kThreads = 8;
  constexpr uint32_t kIterations = 1000;

  rocjitsu::KfdProcess process_a(kVmidA);
  rocjitsu::KfdProcess process_b(kVmidB);
  std::array<uint8_t, GpuMemory::PAGE_SIZE> backing{};
  process_a.map_pages(kVaA, backing.data(), backing.size());
  process_b.map_pages(kVaB, backing.data(), backing.size());
  memory.register_process(kVmidA, &process_a.page_table_, &process_a.page_table_mutex_,
                          process_a.page_table_generation());
  memory.register_process(kVmidB, &process_b.page_table_, &process_b.page_table_mutex_,
                          process_b.page_table_generation());

  std::barrier start(kThreads);
  std::vector<std::thread> workers;
  workers.reserve(kThreads);
  for (uint32_t tid = 0; tid < kThreads; ++tid) {
    workers.emplace_back([&, tid] {
      L2Cache &l2 = (tid & 1) ? l2b : l2a;
      const uint64_t addr = ((tid & 1) ? kVaB : kVaA) + kOffset;
      const uint32_t vmid = (tid & 1) ? kVmidB : kVmidA;
      start.arrive_and_wait();
      for (uint32_t i = 0; i < kIterations; ++i) {
        l2.atomic_rmw(
            addr, sizeof(uint32_t),
            [](uint8_t *line, uint32_t offset) {
              uint32_t value = 0;
              std::memcpy(&value, line + offset, sizeof(value));
              ++value;
              std::memcpy(line + offset, &value, sizeof(value));
            },
            vmid);
      }
    });
  }

  for (auto &worker : workers)
    worker.join();

  uint32_t actual = 0;
  std::memcpy(&actual, backing.data() + kOffset, sizeof(actual));
  EXPECT_EQ(actual, kThreads * kIterations);
}

TEST(L2CacheTest, AliasedVasRequireCoherenceBoundary) {
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);

  constexpr uint32_t kVmidA = 7;
  constexpr uint32_t kVmidB = 8;
  constexpr uint64_t kVaA = 0x100000;
  constexpr uint64_t kVaB = 0x201000;

  rocjitsu::KfdProcess process_a(kVmidA);
  rocjitsu::KfdProcess process_b(kVmidB);
  std::array<uint8_t, GpuMemory::PAGE_SIZE> backing{};
  process_a.map_pages(kVaA, backing.data(), backing.size());
  process_b.map_pages(kVaB, backing.data(), backing.size());
  memory.register_process(kVmidA, &process_a.page_table_, &process_a.page_table_mutex_,
                          process_a.page_table_generation());
  memory.register_process(kVmidB, &process_b.page_table_, &process_b.page_table_mutex_,
                          process_b.page_table_generation());

  std::array<uint8_t, L2Cache::LINE_SIZE> initial{};
  std::array<uint8_t, L2Cache::LINE_SIZE> replacement{};
  std::array<uint8_t, L2Cache::LINE_SIZE> dirty{};
  std::array<uint8_t, L2Cache::LINE_SIZE> actual{};
  initial.fill(0x11);
  replacement.fill(0x22);
  dirty.fill(0x33);

  memory.write_block(kVaA, std::span<const uint8_t>(initial), kVmidA);
  l2.read(kVaA, actual.data(), actual.size(), Mtype::RW, kVmidA);
  ASSERT_EQ(actual, initial);
  l2.read(kVaB, actual.data(), actual.size(), Mtype::RW, kVmidB);
  ASSERT_EQ(actual, initial);

  l2.write(kVaB, replacement.data(), replacement.size(), Mtype::RW, kVmidB);
  l2.read(kVaA, actual.data(), actual.size(), Mtype::RW, kVmidA);
  EXPECT_EQ(actual, initial);
  l2.read(kVaA, actual.data(), actual.size(), Mtype::CC, kVmidA);
  EXPECT_EQ(actual, replacement);

  l2.writeback_line(kVaA, dirty.data(), Mtype::RW, kVmidA);
  memory.read_block(kVaB, std::span<uint8_t>(actual), kVmidB);
  EXPECT_EQ(actual, replacement);
  l2.flush_line(kVaA, kVmidA);
  memory.read_block(kVaB, std::span<uint8_t>(actual), kVmidB);
  EXPECT_EQ(actual, dirty);

  l2.read(kVaB, actual.data(), actual.size(), Mtype::RW, kVmidB);
  EXPECT_EQ(actual, replacement);
  l2.read(kVaB, actual.data(), actual.size(), Mtype::CC, kVmidB);
  EXPECT_EQ(actual, dirty);
}

TEST(L2CacheThreadingTest, ConcurrentFlushAllPreservesDirtyWritebacks) {
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);

  constexpr uint32_t kWriterThreads = 4;
  constexpr uint32_t kLinesPerThread = 8;
  constexpr uint32_t kIterations = 64;
  constexpr uint64_t kBase = 0x300000;

  std::atomic<uint32_t> active_writers{0};
  std::barrier start(kWriterThreads + 1);
  std::vector<std::thread> workers;
  workers.reserve(kWriterThreads);

  for (uint32_t tid = 0; tid < kWriterThreads; ++tid) {
    workers.emplace_back([&, tid] {
      std::array<uint8_t, L2Cache::LINE_SIZE> line{};
      start.arrive_and_wait();
      active_writers.fetch_add(1, std::memory_order_release);
      for (uint32_t iteration = 0; iteration < kIterations; ++iteration) {
        for (uint32_t i = 0; i < kLinesPerThread; ++i) {
          const uint64_t addr =
              kBase + (static_cast<uint64_t>(i) * kWriterThreads + tid) * L2Cache::LINE_SIZE;
          for (uint32_t b = 0; b < line.size(); ++b)
            line[b] = static_cast<uint8_t>((tid << 5) ^ iteration ^ i ^ b);
          l2.writeback_line(addr, line.data());
        }
        std::this_thread::yield();
      }
    });
  }

  std::thread flusher([&] {
    start.arrive_and_wait();
    while (active_writers.load(std::memory_order_acquire) < kWriterThreads)
      std::this_thread::yield();
    for (uint32_t i = 0; i < 4; ++i) {
      l2.flush_all();
      std::this_thread::yield();
    }
  });

  for (auto &worker : workers)
    worker.join();
  flusher.join();

  l2.flush_all();

  std::array<uint8_t, L2Cache::LINE_SIZE> expected{};
  std::array<uint8_t, L2Cache::LINE_SIZE> actual{};
  for (uint32_t tid = 0; tid < kWriterThreads; ++tid) {
    for (uint32_t i = 0; i < kLinesPerThread; ++i) {
      const uint64_t addr =
          kBase + (static_cast<uint64_t>(i) * kWriterThreads + tid) * L2Cache::LINE_SIZE;
      for (uint32_t b = 0; b < expected.size(); ++b)
        expected[b] = static_cast<uint8_t>((tid << 5) ^ (kIterations - 1) ^ i ^ b);
      memory.read_block(addr, std::span<uint8_t>(actual));
      EXPECT_EQ(actual, expected) << "addr=0x" << std::hex << addr;
    }
  }
}

TEST(L2CacheTest, InvalidateRangeClampsAtAddressSpaceEnd) {
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);

  constexpr uint64_t kLineAddr = std::numeric_limits<uint64_t>::max() - (L2Cache::LINE_SIZE - 1);
  constexpr uint64_t kRangeAddr =
      std::numeric_limits<uint64_t>::max() - (L2Cache::LINE_SIZE / 2 - 1);
  std::array<uint8_t, L2Cache::LINE_SIZE> initial{};
  std::array<uint8_t, L2Cache::LINE_SIZE> replacement{};
  std::array<uint8_t, L2Cache::LINE_SIZE> actual{};
  initial.fill(0x11);
  replacement.fill(0x22);

  memory.write_block(kLineAddr, std::span<const uint8_t>(initial));
  l2.read(kLineAddr, actual.data(), actual.size());
  ASSERT_EQ(actual, initial);

  memory.write_block(kLineAddr, std::span<const uint8_t>(replacement));
  l2.invalidate_range(kRangeAddr, L2Cache::LINE_SIZE, 0);
  l2.read(kLineAddr, actual.data(), actual.size());

  EXPECT_EQ(actual, replacement);
}

TEST(L2CacheTest, InvalidateRangeRefreshesOnlyCoveredSetsAndZeroSizeIsNoOp) {
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);

  constexpr uint64_t kBase = 0x400000;
  constexpr uint32_t kLines = 5;
  std::array<std::array<uint8_t, L2Cache::LINE_SIZE>, kLines> initial{};
  std::array<std::array<uint8_t, L2Cache::LINE_SIZE>, kLines> replacement{};
  std::array<uint8_t, L2Cache::LINE_SIZE> actual{};

  for (uint32_t line = 0; line < kLines; ++line) {
    initial[line].fill(static_cast<uint8_t>(0x10 + line));
    replacement[line].fill(static_cast<uint8_t>(0x80 + line));
    const uint64_t addr = kBase + static_cast<uint64_t>(line) * L2Cache::LINE_SIZE;
    memory.write_block(addr, std::span<const uint8_t>(initial[line]));
    l2.read(addr, actual.data(), actual.size());
    ASSERT_EQ(actual, initial[line]);
    memory.write_block(addr, std::span<const uint8_t>(replacement[line]));
  }

  l2.invalidate_range(kBase + L2Cache::LINE_SIZE + 32, 2 * L2Cache::LINE_SIZE, 0);

  for (uint32_t line = 0; line < kLines; ++line) {
    const uint64_t addr = kBase + static_cast<uint64_t>(line) * L2Cache::LINE_SIZE;
    l2.read(addr, actual.data(), actual.size());
    const auto &expected = (line >= 1 && line <= 3) ? replacement[line] : initial[line];
    EXPECT_EQ(actual, expected) << "line=" << line;
  }

  l2.invalidate_range(kBase, 0, 0);
  l2.read(kBase, actual.data(), actual.size());
  EXPECT_EQ(actual, initial[0]);
}

TEST(L2CacheTest, InvalidateRangeOnlyAffectsRequestedVmid) {
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);

  constexpr uint32_t kVmidA = 7;
  constexpr uint32_t kVmidB = 8;
  constexpr uint64_t kAddr = 0x500000;
  rocjitsu::KfdProcess process_a(kVmidA);
  rocjitsu::KfdProcess process_b(kVmidB);
  std::array<uint8_t, GpuMemory::PAGE_SIZE> backing_a{};
  std::array<uint8_t, GpuMemory::PAGE_SIZE> backing_b{};
  process_a.map_pages(kAddr, backing_a.data(), backing_a.size());
  process_b.map_pages(kAddr, backing_b.data(), backing_b.size());
  memory.register_process(kVmidA, &process_a.page_table_, &process_a.page_table_mutex_,
                          process_a.page_table_generation());
  memory.register_process(kVmidB, &process_b.page_table_, &process_b.page_table_mutex_,
                          process_b.page_table_generation());

  std::array<uint8_t, L2Cache::LINE_SIZE> initial_a{};
  std::array<uint8_t, L2Cache::LINE_SIZE> initial_b{};
  std::array<uint8_t, L2Cache::LINE_SIZE> dirty_a{};
  std::array<uint8_t, L2Cache::LINE_SIZE> replacement_b{};
  std::array<uint8_t, L2Cache::LINE_SIZE> actual{};
  initial_a.fill(0x11);
  initial_b.fill(0x22);
  dirty_a.fill(0x33);
  replacement_b.fill(0x44);

  memory.write_block(kAddr, std::span<const uint8_t>(initial_a), kVmidA);
  memory.write_block(kAddr, std::span<const uint8_t>(initial_b), kVmidB);
  l2.read(kAddr, actual.data(), actual.size(), Mtype::RW, kVmidA);
  ASSERT_EQ(actual, initial_a);
  l2.read(kAddr, actual.data(), actual.size(), Mtype::RW, kVmidB);
  ASSERT_EQ(actual, initial_b);
  l2.writeback_line(kAddr, dirty_a.data(), Mtype::RW, kVmidA);

  memory.write_block(kAddr, std::span<const uint8_t>(replacement_b), kVmidB);
  l2.invalidate_range(kAddr, L2Cache::LINE_SIZE, kVmidB);
  l2.read(kAddr, actual.data(), actual.size(), Mtype::RW, kVmidB);
  EXPECT_EQ(actual, replacement_b);

  l2.flush_line(kAddr, kVmidA);
  memory.read_block(kAddr, std::span<uint8_t>(actual), kVmidA);
  EXPECT_EQ(actual, dirty_a);
}

TEST(L2CacheTest, InvalidateRangePreservesDirtyBytesOutsidePartialWrite) {
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);

  constexpr uint64_t kAddr = 0x600000;
  constexpr uint32_t kOffset = 32;
  std::array<uint8_t, L2Cache::LINE_SIZE> initial{};
  std::array<uint8_t, L2Cache::LINE_SIZE> dirty{};
  std::array<uint8_t, 16> host_write{};
  std::array<uint8_t, L2Cache::LINE_SIZE> expected{};
  std::array<uint8_t, L2Cache::LINE_SIZE> actual{};
  initial.fill(0x11);
  dirty.fill(0x22);
  host_write.fill(0x33);
  expected = dirty;
  std::memcpy(expected.data() + kOffset, host_write.data(), host_write.size());

  memory.write_block(kAddr, std::span<const uint8_t>(initial));
  l2.read(kAddr, actual.data(), actual.size());
  ASSERT_EQ(actual, initial);
  l2.writeback_line(kAddr, dirty.data());

  memory.write_block(kAddr + kOffset, std::span<const uint8_t>(host_write));
  l2.invalidate_range(kAddr + kOffset, host_write.size(), 0);

  memory.read_block(kAddr, std::span<uint8_t>(actual));
  EXPECT_EQ(actual, expected);
  l2.read(kAddr, actual.data(), actual.size());
  EXPECT_EQ(actual, expected);
}

TEST(GpuMemoryTest, BlockAccessHandlesPageBoundaries) {
  GpuMemory memory("memory");

  constexpr uint64_t kAddr = 0x3ff0;
  std::array<uint8_t, 64> input{};
  std::array<uint8_t, 64> output{};
  for (uint32_t i = 0; i < input.size(); ++i)
    input[i] = static_cast<uint8_t>(i * 3);

  memory.write_block(kAddr, std::span<const uint8_t>(input));
  memory.read_block(kAddr, std::span<uint8_t>(output));

  EXPECT_EQ(output, input);
}

} // namespace
