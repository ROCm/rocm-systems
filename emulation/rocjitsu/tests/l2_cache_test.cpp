// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l1_vector_cache.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "simdojo/components/sparse_memory.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

namespace {

using rocjitsu::amdgpu::GpuMemory;
using rocjitsu::amdgpu::L1VectorCache;
using rocjitsu::amdgpu::L2Cache;
using rocjitsu::amdgpu::Mtype;

TEST(SparseMemoryThreadingTest, ConcurrentDifferentPageWritesArePreserved) {
  simdojo::SparseMemory memory("memory");

  constexpr uint32_t kThreads = 8;
  constexpr uint32_t kPagesPerThread = 64;
  constexpr uint64_t kBase = 0x800000;

  std::barrier start(kThreads);
  std::vector<std::thread> workers;
  workers.reserve(kThreads);

  for (uint32_t tid = 0; tid < kThreads; ++tid) {
    workers.emplace_back([&, tid] {
      std::array<uint8_t, simdojo::SparseMemory::PAGE_SIZE> page{};
      start.arrive_and_wait();
      for (uint32_t i = 0; i < kPagesPerThread; ++i) {
        const uint64_t addr =
            kBase + (static_cast<uint64_t>(i) * kThreads + tid) * simdojo::SparseMemory::PAGE_SIZE;
        for (uint32_t b = 0; b < page.size(); ++b)
          page[b] = static_cast<uint8_t>((tid << 4) ^ i ^ b);
        memory.write_block(addr, page.data(), page.size());
      }
    });
  }

  for (auto &worker : workers)
    worker.join();

  EXPECT_EQ(memory.num_pages(), kThreads * kPagesPerThread);

  std::array<uint8_t, simdojo::SparseMemory::PAGE_SIZE> expected{};
  std::array<uint8_t, simdojo::SparseMemory::PAGE_SIZE> actual{};
  for (uint32_t tid = 0; tid < kThreads; ++tid) {
    for (uint32_t i = 0; i < kPagesPerThread; ++i) {
      const uint64_t addr =
          kBase + (static_cast<uint64_t>(i) * kThreads + tid) * simdojo::SparseMemory::PAGE_SIZE;
      for (uint32_t b = 0; b < expected.size(); ++b)
        expected[b] = static_cast<uint8_t>((tid << 4) ^ i ^ b);
      memory.read_block(addr, actual.data(), actual.size());
      EXPECT_EQ(actual, expected) << "addr=0x" << std::hex << addr;
      uint32_t expected_word = 0;
      std::memcpy(&expected_word, expected.data(), sizeof(expected_word));
      EXPECT_EQ(memory.read32(addr), expected_word);
    }
  }
}

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
      memory.read_block(addr, actual.data(), actual.size());
      EXPECT_EQ(actual, expected) << "addr=0x" << std::hex << addr;
    }
  }
}

TEST(L1VectorCacheTest, CoalescesContiguousLaneStores) {
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);
  L1VectorCache l1(&l2);
  l1.set_memory(&memory);

  constexpr uint32_t kLanes = 8;
  constexpr uint32_t kElemSize = sizeof(uint32_t);
  constexpr uint32_t kNumElems = 2;
  constexpr uint32_t kStride = kElemSize * kNumElems;
  constexpr uint64_t kBase = 0x400000;

  std::array<uint64_t, 64> addrs{};
  std::array<uint8_t, 64 * kStride> src{};
  for (uint32_t lane = 0; lane < kLanes; ++lane) {
    addrs[lane] = kBase + lane * kStride;
    for (uint32_t elem = 0; elem < kNumElems; ++elem) {
      uint32_t value = 0xabc00000u | (lane << 8) | elem;
      std::memcpy(src.data() + lane * kStride + elem * kElemSize, &value, sizeof(value));
    }
  }

  l1.store(addrs.data(), (1ULL << kLanes) - 1, kElemSize, kNumElems, src.data(), Mtype::RW, false);

  EXPECT_EQ(l1.store_l2_writes(), kLanes * kNumElems);
  EXPECT_EQ(l2.write_count(), 1u);

  std::array<uint8_t, kLanes * kStride> actual{};
  memory.read_block(kBase, actual.data(), actual.size());
  EXPECT_TRUE(std::equal(actual.begin(), actual.end(), src.begin()));
}

TEST(L1VectorCacheTest, CoalescesContiguousLaneLoads) {
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);
  L1VectorCache l1(&l2);
  l1.set_memory(&memory);

  constexpr uint32_t kLanes = 8;
  constexpr uint32_t kElemSize = sizeof(uint32_t);
  constexpr uint32_t kNumElems = 4;
  constexpr uint32_t kStride = kElemSize * kNumElems;
  constexpr uint64_t kBase = 0x480000;

  std::array<uint64_t, 64> addrs{};
  std::array<uint8_t, kLanes * kStride> input{};
  for (uint32_t lane = 0; lane < kLanes; ++lane) {
    addrs[lane] = kBase + lane * kStride;
    for (uint32_t elem = 0; elem < kNumElems; ++elem) {
      uint32_t value = 0xf00d0000u | (lane << 8) | elem;
      std::memcpy(input.data() + lane * kStride + elem * kElemSize, &value, sizeof(value));
    }
  }
  memory.write_block(kBase, input.data(), input.size());

  std::array<uint8_t, 64 * kStride> output{};
  l1.load(addrs.data(), (1ULL << kLanes) - 1, kElemSize, kNumElems, output.data(), Mtype::RW,
          false);

  EXPECT_EQ(l1.read_count(), 1u);
  EXPECT_TRUE(std::equal(input.begin(), input.end(), output.begin()));
}

TEST(L1VectorCacheTest, MaskGapsBreakContiguousLaneStoreRuns) {
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);
  L1VectorCache l1(&l2);
  l1.set_memory(&memory);

  constexpr uint32_t kElemSize = sizeof(uint32_t);
  constexpr uint32_t kNumElems = 1;
  constexpr uint32_t kStride = kElemSize * kNumElems;
  constexpr uint64_t kBase = 0x500000;
  constexpr uint64_t kMask = (1ULL << 0) | (1ULL << 2) | (1ULL << 3);

  std::array<uint64_t, 64> addrs{};
  addrs[0] = kBase;
  addrs[2] = kBase + kStride;
  addrs[3] = kBase + 2 * kStride;

  std::array<uint8_t, 64 * kStride> src{};
  uint32_t lane0 = 0x10101010;
  uint32_t lane1_inactive = 0xdeadbeef;
  uint32_t lane2 = 0x20202020;
  uint32_t lane3 = 0x30303030;
  std::memcpy(src.data() + 0 * kStride, &lane0, sizeof(lane0));
  std::memcpy(src.data() + 1 * kStride, &lane1_inactive, sizeof(lane1_inactive));
  std::memcpy(src.data() + 2 * kStride, &lane2, sizeof(lane2));
  std::memcpy(src.data() + 3 * kStride, &lane3, sizeof(lane3));

  l1.store(addrs.data(), kMask, kElemSize, kNumElems, src.data(), Mtype::RW, false);

  EXPECT_EQ(l1.store_l2_writes(), 3u);
  EXPECT_EQ(l2.write_count(), 2u);

  std::array<uint32_t, 3> actual{};
  memory.read_block(kBase, reinterpret_cast<uint8_t *>(actual.data()),
                    actual.size() * sizeof(uint32_t));
  EXPECT_EQ(actual[0], lane0);
  EXPECT_EQ(actual[1], lane2);
  EXPECT_EQ(actual[2], lane3);
}

TEST(L1VectorCacheTest, MaskGapsBreakContiguousLaneLoadRuns) {
  GpuMemory memory("memory");
  L2Cache l2("l2");
  l2.set_backing_memory(&memory);
  L1VectorCache l1(&l2);
  l1.set_memory(&memory);

  constexpr uint32_t kElemSize = sizeof(uint32_t);
  constexpr uint32_t kNumElems = 1;
  constexpr uint32_t kStride = kElemSize * kNumElems;
  constexpr uint64_t kBase = 0x580000;
  constexpr uint64_t kMask = (1ULL << 0) | (1ULL << 2) | (1ULL << 3);

  std::array<uint64_t, 64> addrs{};
  addrs[0] = kBase;
  addrs[2] = kBase + kStride;
  addrs[3] = kBase + 2 * kStride;

  std::array<uint32_t, 3> input{0x10101010, 0x20202020, 0x30303030};
  memory.write_block(kBase, reinterpret_cast<const uint8_t *>(input.data()),
                     input.size() * sizeof(uint32_t));

  std::array<uint8_t, 64 * kStride> output{};
  l1.load(addrs.data(), kMask, kElemSize, kNumElems, output.data(), Mtype::RW, false);

  EXPECT_EQ(l1.read_count(), 2u);
  uint32_t lane0 = 0;
  uint32_t lane2 = 0;
  uint32_t lane3 = 0;
  std::memcpy(&lane0, output.data() + 0 * kStride, sizeof(lane0));
  std::memcpy(&lane2, output.data() + 2 * kStride, sizeof(lane2));
  std::memcpy(&lane3, output.data() + 3 * kStride, sizeof(lane3));
  EXPECT_EQ(lane0, input[0]);
  EXPECT_EQ(lane2, input[1]);
  EXPECT_EQ(lane3, input[2]);
}

TEST(GpuMemoryTest, BlockAccessHandlesPageBoundaries) {
  GpuMemory memory("memory");

  constexpr uint64_t kAddr = 0x3ff0;
  std::array<uint8_t, 64> input{};
  std::array<uint8_t, 64> output{};
  for (uint32_t i = 0; i < input.size(); ++i)
    input[i] = static_cast<uint8_t>(i * 3);

  memory.write_block(kAddr, input.data(), input.size());
  memory.read_block(kAddr, output.data(), output.size());

  EXPECT_EQ(output, input);
}

} // namespace
