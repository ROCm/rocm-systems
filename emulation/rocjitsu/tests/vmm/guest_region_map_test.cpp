// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vmm/vfu/guest_region_map.h"

#include <gtest/gtest.h>

#include <limits>

#include <cstdint>

namespace {

constexpr uint64_t kPageSize = 0x1000;

simdojo::DmaRegion window(uint64_t base, uint64_t length) {
  return {.guest_phys = base, .length = length, .access = simdojo::DmaAccess::None};
}

TEST(GuestRegionMap, ReportsTheDistanceToTheEndOfAWindow) {
  rocjitsu::GuestRegionMap regions;
  ASSERT_TRUE(regions.insert(window(0x10000, kPageSize)));

  EXPECT_EQ(regions.bytes_until_end(0x10000), kPageSize);
  EXPECT_EQ(regions.bytes_until_end(0x10000 + 0x400), kPageSize - 0x400);
}

TEST(GuestRegionMap, ReportsNothingOutsideAnyWindow) {
  rocjitsu::GuestRegionMap regions;
  ASSERT_TRUE(regions.insert(window(0x10000, kPageSize)));

  EXPECT_EQ(regions.bytes_until_end(0x0fff0), 0u) << "below every window";
  EXPECT_EQ(regions.bytes_until_end(0x11000), 0u) << "one past the end";
  EXPECT_EQ(regions.bytes_until_end(0x90000), 0u) << "in a gap above";
}

TEST(GuestRegionMap, IgnoresAWindowSharedTwice) {
  rocjitsu::GuestRegionMap regions;

  EXPECT_TRUE(regions.insert(window(0x10000, kPageSize)));
  EXPECT_FALSE(regions.insert(window(0x10000, kPageSize)));
  EXPECT_EQ(regions.size(), 1u);
}

// A client may withdraw a window and share a differently sized one at the same
// address. Keeping the stale length would make a later transfer split on a
// boundary that no longer exists and refuse a range that is now valid.
TEST(GuestRegionMap, ReplacesAWindowResharedWithANewLength) {
  rocjitsu::GuestRegionMap regions;
  ASSERT_TRUE(regions.insert(window(0x10000, kPageSize)));

  EXPECT_TRUE(regions.insert(window(0x10000, 4 * kPageSize)));
  EXPECT_EQ(regions.size(), 1u);
  EXPECT_EQ(regions.bytes_until_end(0x10000), 4 * kPageSize);
}

TEST(GuestRegionMap, ForgetsWithdrawnWindows) {
  rocjitsu::GuestRegionMap regions;
  ASSERT_TRUE(regions.insert(window(0x10000, kPageSize)));

  regions.erase(window(0x10000, kPageSize));

  EXPECT_EQ(regions.size(), 0u);
  EXPECT_EQ(regions.bytes_until_end(0x10000), 0u);
}

TEST(GuestRegionMap, ForgetsEverythingWhenAClientDisconnects) {
  rocjitsu::GuestRegionMap regions;
  ASSERT_TRUE(regions.insert(window(0x10000, kPageSize)));
  ASSERT_TRUE(regions.insert(window(0x11000, kPageSize)));

  regions.clear();

  EXPECT_EQ(regions.size(), 0u);
}

// A guest whose IOMMU reflects memory page by page shares very many windows, and
// they do not arrive in address order. Walking a contiguous range across all of
// them is the streaming path a large transfer takes.
TEST(GuestRegionMap, WalksManyPageWindowsSharedOutOfOrder) {
  constexpr uint64_t kBase = 0x100000;
  constexpr int kWindows = 4096;
  rocjitsu::GuestRegionMap regions;

  for (int i = kWindows - 1; i >= 0; --i) {
    ASSERT_TRUE(regions.insert(window(kBase + static_cast<uint64_t>(i) * kPageSize, kPageSize)));
  }
  ASSERT_EQ(regions.size(), static_cast<std::size_t>(kWindows));

  uint64_t address = kBase;
  uint64_t covered = 0;
  while (true) {
    const std::size_t chunk = regions.bytes_until_end(address);
    if (chunk == 0) {
      break;
    }
    covered += chunk;
    address += chunk;
  }

  EXPECT_EQ(covered, static_cast<uint64_t>(kWindows) * kPageSize)
      << "a range spanning every shared window must be walkable end to end";
}

} // namespace

// Both branches of the sanity check on a window's length. These values arrive
// from a vfio-user client, so neither is hypothetical: a zero-length window
// contains nothing, and one whose length runs past the end of the address space
// would wrap its computed end and make the containment test succeed for an
// address the window does not hold.
TEST(GuestRegionMap, HoldsNothingForAZeroLengthWindow) {
  rocjitsu::GuestRegionMap regions;
  ASSERT_TRUE(regions.insert(window(0x100000, 0)));
  EXPECT_EQ(regions.bytes_until_end(0x100000), 0u);
}

TEST(GuestRegionMap, HoldsNothingForAWindowThatWrapsTheAddressSpace) {
  rocjitsu::GuestRegionMap regions;
  constexpr uint64_t kNearTop = std::numeric_limits<uint64_t>::max() - 0xfff;
  ASSERT_TRUE(regions.insert(window(kNearTop, 0x2000)));
  EXPECT_EQ(regions.bytes_until_end(kNearTop), 0u);
  EXPECT_EQ(regions.bytes_until_end(0x10), 0u) << "a wrapped window claimed low memory";
}
