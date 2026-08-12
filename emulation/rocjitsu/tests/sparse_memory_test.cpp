// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "simdojo/components/sparse_memory.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <barrier>
#include <cstdint>
#include <cstring>
#include <span>
#include <thread>
#include <vector>

namespace {

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
        memory.write_block(addr, page);
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
      memory.read_block(addr, actual);
      EXPECT_EQ(actual, expected) << "addr=0x" << std::hex << addr;
      uint32_t expected_word = 0;
      std::memcpy(&expected_word, expected.data(), sizeof(expected_word));
      EXPECT_EQ(memory.read32(addr), expected_word);
    }
  }
}

TEST(SparseMemoryThreadingTest, ConcurrentSamePageWritesArePreserved) {
  simdojo::SparseMemory memory("memory");

  constexpr uint32_t kThreads = 8;
  constexpr size_t kBytesPerThread = simdojo::SparseMemory::PAGE_SIZE / kThreads;
  constexpr uint64_t kBase = 0x900000;

  std::barrier start(kThreads);
  std::vector<std::thread> workers;
  workers.reserve(kThreads);

  for (uint32_t tid = 0; tid < kThreads; ++tid) {
    workers.emplace_back([&, tid] {
      std::array<uint8_t, kBytesPerThread> bytes{};
      std::fill(bytes.begin(), bytes.end(), static_cast<uint8_t>(0x20 + tid));
      start.arrive_and_wait();
      for (uint32_t iteration = 0; iteration < 64; ++iteration)
        memory.write_block(kBase + tid * kBytesPerThread, bytes);
    });
  }

  for (auto &worker : workers)
    worker.join();

  std::array<uint8_t, simdojo::SparseMemory::PAGE_SIZE> actual{};
  memory.read_block(kBase, actual);
  for (uint32_t tid = 0; tid < kThreads; ++tid) {
    const auto begin = actual.begin() + tid * kBytesPerThread;
    const auto end = begin + kBytesPerThread;
    EXPECT_TRUE(std::all_of(
        begin, end, [tid](uint8_t byte) { return byte == static_cast<uint8_t>(0x20 + tid); }));
  }
}

TEST(SparseMemoryTest, UnalignedBlockRoundTripCrossesPageBoundary) {
  simdojo::SparseMemory memory("memory");

  constexpr uint64_t kAddr = simdojo::SparseMemory::PAGE_SIZE - 37;
  std::array<uint8_t, 128> input{};
  for (size_t i = 0; i < input.size(); ++i)
    input[i] = static_cast<uint8_t>((i * 17) ^ 0xA5);

  memory.write_block(kAddr, input);

  std::array<uint8_t, input.size()> output{};
  memory.read_block(kAddr, output);
  EXPECT_EQ(output, input);
  EXPECT_EQ(memory.num_pages(), 2u);
}

TEST(SparseMemoryThreadingTest, ConcurrentOverlappingBlocksRemainAtomicPerPage) {
  simdojo::SparseMemory memory("memory");

  constexpr uint64_t kBase = 0xA00000;
  constexpr size_t kBlockSize = 2 * simdojo::SparseMemory::PAGE_SIZE;
  std::array<uint8_t, kBlockSize> first{};
  std::array<uint8_t, kBlockSize> second{};
  first.fill(0x3C);
  second.fill(0xC3);

  std::barrier start(2);
  std::thread first_writer([&] {
    start.arrive_and_wait();
    for (uint32_t iteration = 0; iteration < 128; ++iteration)
      memory.write_block(kBase, first);
  });
  std::thread second_writer([&] {
    start.arrive_and_wait();
    for (uint32_t iteration = 0; iteration < 128; ++iteration)
      memory.write_block(kBase, second);
  });

  first_writer.join();
  second_writer.join();

  std::array<uint8_t, kBlockSize> actual{};
  memory.read_block(kBase, actual);
  for (size_t page = 0; page < 2; ++page) {
    const auto bytes = std::span<const uint8_t>(actual).subspan(
        page * simdojo::SparseMemory::PAGE_SIZE, simdojo::SparseMemory::PAGE_SIZE);
    const bool is_first =
        std::all_of(bytes.begin(), bytes.end(), [](uint8_t byte) { return byte == 0x3C; });
    const bool is_second =
        std::all_of(bytes.begin(), bytes.end(), [](uint8_t byte) { return byte == 0xC3; });
    EXPECT_TRUE(is_first || is_second) << "page=" << page;
  }
}

} // namespace
