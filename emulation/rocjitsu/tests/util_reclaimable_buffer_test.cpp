// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file util_reclaimable_buffer_test.cpp
/// @brief Unit tests for stable zero-filled reclaimable storage.

#include "util/reclaimable_buffer.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>

#if defined(__linux__)
#include <cerrno>
#include <cstring>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>
#endif

namespace {

static_assert(!std::is_copy_constructible_v<util::ReclaimableBuffer>);
static_assert(!std::is_copy_assignable_v<util::ReclaimableBuffer>);
static_assert(std::is_nothrow_move_constructible_v<util::ReclaimableBuffer>);
static_assert(std::is_nothrow_move_assignable_v<util::ReclaimableBuffer>);

#if defined(__linux__)
size_t system_page_size() {
  const long result = sysconf(_SC_PAGESIZE);
  if (result <= 0) {
    ADD_FAILURE() << "sysconf(_SC_PAGESIZE) failed";
    return 4096;
  }
  return static_cast<size_t>(result);
}

size_t resident_page_count(const void *data, size_t bytes) {
  const size_t page_size = system_page_size();
  const uintptr_t first = reinterpret_cast<uintptr_t>(data) / page_size * page_size;
  const uintptr_t last_address = reinterpret_cast<uintptr_t>(data) + bytes;
  const uintptr_t last = (last_address + page_size - 1) / page_size * page_size;
  std::vector<unsigned char> residency((last - first) / page_size);
  if (mincore(reinterpret_cast<void *>(first), last - first, residency.data()) != 0) {
    ADD_FAILURE() << "mincore failed: " << std::strerror(errno);
    return std::numeric_limits<size_t>::max();
  }
  return static_cast<size_t>(std::count_if(residency.begin(), residency.end(),
                                           [](unsigned char value) { return value & 1; }));
}
#endif

TEST(ReclaimableBuffer, AllocatesAlignedZeroInitializedStableStorage) {
  constexpr size_t kAlignment = 8192;
  constexpr size_t kBytes = 12345;
  util::ReclaimableBuffer buffer;
  buffer.allocate(kBytes, kAlignment);

  ASSERT_NE(buffer.data(), nullptr);
  EXPECT_EQ(buffer.size(), kBytes);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(buffer.data()) % kAlignment, 0u);
  const size_t reclamation_granularity = util::ReclaimableBuffer::reclamation_granularity();
#if defined(__linux__)
  EXPECT_EQ(reclamation_granularity, system_page_size());
#else
  EXPECT_EQ(reclamation_granularity, 1u);
#endif
  ASSERT_GT(reclamation_granularity, 0u);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(buffer.data()) % reclamation_granularity, 0u);
  const std::byte *original_address = buffer.data();
  for (size_t index = 0; index < buffer.size(); ++index)
    ASSERT_EQ(buffer.data()[index], std::byte{0}) << "byte " << index;

  std::fill(buffer.data(), buffer.data() + buffer.size(), std::byte{0x5a});
  buffer.zero_and_reclaim(/*offset=*/0, buffer.size());
  EXPECT_EQ(buffer.data(), original_address);
  for (size_t index = 0; index < buffer.size(); ++index)
    EXPECT_EQ(buffer.data()[index], std::byte{0}) << "byte " << index;
}

TEST(ReclaimableBuffer, ZeroSizeAllocationLeavesBufferEmpty) {
  util::ReclaimableBuffer buffer;

  buffer.allocate(/*bytes=*/0, alignof(std::max_align_t));

  EXPECT_EQ(buffer.data(), nullptr);
  EXPECT_EQ(buffer.size(), 0u);
}

TEST(ReclaimableBuffer, AllocationOverflowLeavesBufferReusable) {
  util::ReclaimableBuffer buffer;

  EXPECT_THROW(buffer.allocate(std::numeric_limits<size_t>::max(), alignof(std::max_align_t)),
               std::bad_alloc);
  EXPECT_EQ(buffer.data(), nullptr);
  EXPECT_EQ(buffer.size(), 0u);

  buffer.allocate(/*bytes=*/64, alignof(std::max_align_t));
  ASSERT_NE(buffer.data(), nullptr);
  EXPECT_EQ(buffer.size(), 64u);
  EXPECT_TRUE(std::all_of(buffer.data(), buffer.data() + buffer.size(),
                          [](std::byte value) { return value == std::byte{0}; }));
}

TEST(ReclaimableBuffer, PreservesBytesOutsideUnalignedRange) {
#if defined(__linux__)
  const size_t page_size = system_page_size();
#else
  constexpr size_t page_size = 4096;
#endif
  const size_t total_bytes = page_size * 4 + 137;
  const size_t clear_offset = page_size / 2 + 3;
  const size_t clear_bytes = page_size * 2 + 29;
  util::ReclaimableBuffer buffer;
  buffer.allocate(total_bytes, alignof(std::max_align_t));
  std::fill(buffer.data(), buffer.data() + buffer.size(), std::byte{0xa5});

  buffer.zero_and_reclaim(clear_offset, clear_bytes);

  for (size_t index = 0; index < clear_offset; ++index)
    ASSERT_EQ(buffer.data()[index], std::byte{0xa5}) << "leading byte " << index;
  for (size_t index = clear_offset; index < clear_offset + clear_bytes; ++index)
    ASSERT_EQ(buffer.data()[index], std::byte{0}) << "cleared byte " << index;
  for (size_t index = clear_offset + clear_bytes; index < buffer.size(); ++index)
    ASSERT_EQ(buffer.data()[index], std::byte{0xa5}) << "trailing byte " << index;
}

TEST(ReclaimableBuffer, MoveTransfersOwnershipWithoutMovingStorage) {
  util::ReclaimableBuffer source;
  source.allocate(/*bytes=*/4096, alignof(std::max_align_t));
  std::byte *original_address = source.data();
  source.data()[17] = std::byte{0x7b};

  util::ReclaimableBuffer destination(std::move(source));

  EXPECT_EQ(source.data(), nullptr);
  EXPECT_EQ(source.size(), 0u);
  EXPECT_EQ(destination.data(), original_address);
  EXPECT_EQ(destination.data()[17], std::byte{0x7b});
}

TEST(ReclaimableBuffer, MoveAssignmentReleasesDestinationAndHandlesSelfMove) {
  util::ReclaimableBuffer source;
  source.allocate(/*bytes=*/4096, alignof(std::max_align_t));
  std::fill(source.data(), source.data() + source.size(), std::byte{0x6a});
  std::byte *source_address = source.data();

  util::ReclaimableBuffer destination;
  destination.allocate(/*bytes=*/2048, alignof(std::max_align_t));
  destination.data()[0] = std::byte{0x1b};

  destination = std::move(source);

  EXPECT_EQ(source.data(), nullptr);
  EXPECT_EQ(source.size(), 0u);
  EXPECT_EQ(destination.data(), source_address);
  EXPECT_EQ(destination.size(), 4096u);
  EXPECT_TRUE(std::all_of(destination.data(), destination.data() + destination.size(),
                          [](std::byte value) { return value == std::byte{0x6a}; }));

  std::byte *destination_address = destination.data();
  util::ReclaimableBuffer *same_buffer = &destination;
  destination = std::move(*same_buffer);
  EXPECT_EQ(destination.data(), destination_address);
  EXPECT_EQ(destination.size(), 4096u);
  EXPECT_EQ(destination.data()[0], std::byte{0x6a});
}

TEST(ReclaimableBuffer, ZeroLengthRangeIsNoOp) {
  util::ReclaimableBuffer buffer;
  buffer.allocate(/*bytes=*/64, alignof(std::max_align_t));
  std::fill(buffer.data(), buffer.data() + buffer.size(), std::byte{0x3c});

  buffer.zero_and_reclaim(/*offset=*/buffer.size(), /*bytes=*/0);

  EXPECT_TRUE(std::all_of(buffer.data(), buffer.data() + buffer.size(),
                          [](std::byte value) { return value == std::byte{0x3c}; }));
}

TEST(ReclaimableBuffer, ClearsSmallRangeWithoutTouchingNeighbors) {
  util::ReclaimableBuffer buffer;
  buffer.allocate(/*bytes=*/64, alignof(std::max_align_t));
  std::fill(buffer.data(), buffer.data() + buffer.size(), std::byte{0x4d});

  constexpr size_t kClearOffset = 17;
  constexpr size_t kClearBytes = 13;
  buffer.zero_and_reclaim(kClearOffset, kClearBytes);

  EXPECT_TRUE(std::all_of(buffer.data(), buffer.data() + kClearOffset,
                          [](std::byte value) { return value == std::byte{0x4d}; }));
  EXPECT_TRUE(std::all_of(buffer.data() + kClearOffset, buffer.data() + kClearOffset + kClearBytes,
                          [](std::byte value) { return value == std::byte{0}; }));
  EXPECT_TRUE(std::all_of(buffer.data() + kClearOffset + kClearBytes, buffer.data() + buffer.size(),
                          [](std::byte value) { return value == std::byte{0x4d}; }));
}

TEST(ReclaimableBuffer, ReclaimsWhollyCoveredPages) {
#if defined(__linux__)
  const size_t page_size = system_page_size();
  util::ReclaimableBuffer buffer;
  buffer.allocate(page_size * 4, alignof(std::max_align_t));
  EXPECT_EQ(resident_page_count(buffer.data(), buffer.size()), 0u);

  std::fill(buffer.data(), buffer.data() + buffer.size(), std::byte{0x6d});
  const size_t resident_pages = resident_page_count(buffer.data(), buffer.size());
  EXPECT_GE(resident_pages, 4u);

  buffer.zero_and_reclaim(/*offset=*/0, buffer.size());
  EXPECT_EQ(resident_page_count(buffer.data(), buffer.size()), 0u);
#else
  GTEST_SKIP() << "physical-page residency is only observable on Linux";
#endif
}

} // namespace
