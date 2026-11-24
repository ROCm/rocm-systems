// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vmm/vfu/guest_region_map.h"

#include <gtest/gtest.h>

#include <array>
#include <limits>

#include <cstddef>
#include <cstdint>

namespace {

constexpr uint64_t kPageSize = 0x1000;
constexpr simdojo::DmaAccess kReadWrite = simdojo::DmaAccess::Read | simdojo::DmaAccess::Write;

simdojo::DmaRegion window(uint64_t base, uint64_t length,
                          simdojo::DmaAccess access = simdojo::DmaAccess::None) {
  return {.guest_phys = base, .length = length, .access = access};
}

// Host memory standing in for a window the transport mapped. Nothing here
// reads or writes it: a lookup only does arithmetic on its address.
using HostPages = std::array<std::byte, 2 * kPageSize>;

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

// A copy that bypasses the transport is handed an address in this process, so
// what a lookup returns has to be exact: the window's own address, displaced by
// exactly as far as the guest address lies into the window.
TEST(GuestRegionMap, LocatesASpanInsideAMappedWindow) {
  rocjitsu::GuestRegionMap regions;
  HostPages host{};
  ASSERT_TRUE(regions.insert(window(0x10000, host.size(), kReadWrite), host.data()));

  EXPECT_EQ(regions.host_span(0x10000 + 0x400, 0x100, simdojo::DmaAccess::Read),
            host.data() + 0x400);
  EXPECT_EQ(regions.host_span(0x10000 + kPageSize, 8, simdojo::DmaAccess::Write),
            host.data() + kPageSize);
}

// The first and last bytes of a window belong to it and the bytes either side
// do not. A span is refused as soon as any byte of it is outside, because the
// address returned is used for the whole length without another check.
TEST(GuestRegionMap, LocatesSpansUpToBothEdgesOfAWindowAndNoFurther) {
  constexpr uint64_t kBase = 0x10000;
  rocjitsu::GuestRegionMap regions;
  HostPages host{};
  const uint64_t end = kBase + host.size();
  ASSERT_TRUE(regions.insert(window(kBase, host.size(), kReadWrite), host.data()));

  EXPECT_EQ(regions.host_span(kBase, host.size(), simdojo::DmaAccess::Read), host.data())
      << "the whole window";
  EXPECT_EQ(regions.host_span(kBase, 1, simdojo::DmaAccess::Read), host.data()) << "the first byte";
  EXPECT_EQ(regions.host_span(end - 1, 1, simdojo::DmaAccess::Read), host.data() + host.size() - 1)
      << "the last byte";

  EXPECT_EQ(regions.host_span(kBase - 1, 1, simdojo::DmaAccess::Read), nullptr)
      << "the byte below the window";
  EXPECT_EQ(regions.host_span(kBase - 1, 2, simdojo::DmaAccess::Read), nullptr)
      << "a span starting below the window";
  EXPECT_EQ(regions.host_span(end, 1, simdojo::DmaAccess::Read), nullptr)
      << "the byte past the window";
  EXPECT_EQ(regions.host_span(end - 1, 2, simdojo::DmaAccess::Read), nullptr)
      << "a span running one byte past the window";
  EXPECT_EQ(regions.host_span(kBase, host.size() + 1, simdojo::DmaAccess::Read), nullptr)
      << "a span one byte longer than the window";
  EXPECT_EQ(regions.host_span(kBase + 8, std::numeric_limits<std::size_t>::max(),
                              simdojo::DmaAccess::Read),
            nullptr)
      << "a length that would wrap if it were added to the address";
}

// Two windows that touch in guest memory are separate mappings, and nothing
// says they touch in this process. Here they do, which is the case a lookup
// could get away with joining them in, so it is the one that shows it does not.
TEST(GuestRegionMap, RefusesASpanCrossingTwoAdjacentWindows) {
  constexpr uint64_t kBase = 0x10000;
  rocjitsu::GuestRegionMap regions;
  HostPages host{};
  ASSERT_TRUE(regions.insert(window(kBase, kPageSize, kReadWrite), host.data()));
  ASSERT_TRUE(
      regions.insert(window(kBase + kPageSize, kPageSize, kReadWrite), host.data() + kPageSize));

  EXPECT_EQ(regions.host_span(kBase + kPageSize - 8, 16, simdojo::DmaAccess::Read), nullptr);
  EXPECT_EQ(regions.host_span(kBase, 2 * kPageSize, simdojo::DmaAccess::Read), nullptr);

  EXPECT_EQ(regions.host_span(kBase + kPageSize - 8, 8, simdojo::DmaAccess::Read),
            host.data() + kPageSize - 8)
      << "the part in the lower window";
  EXPECT_EQ(regions.host_span(kBase + kPageSize, 8, simdojo::DmaAccess::Read),
            host.data() + kPageSize)
      << "the part in the upper window";
}

// A window the transport could not map is still a window: transfers are split
// at its end like any other. It has no address here, so none may be returned.
TEST(GuestRegionMap, RefusesASpanInAWindowRecordedWithoutAHostAddress) {
  rocjitsu::GuestRegionMap regions;
  ASSERT_TRUE(regions.insert(window(0x10000, kPageSize, kReadWrite)));

  EXPECT_EQ(regions.host_span(0x10000, 8, simdojo::DmaAccess::Read), nullptr);
  EXPECT_EQ(regions.bytes_until_end(0x10000), kPageSize);
}

// A client may re-share a window it already holds, and the transport may have
// mapped it somewhere else in the meantime. The old address is then a mapping
// that no longer exists, so a lookup has to follow the latest report even
// though the window itself is reported as unchanged.
TEST(GuestRegionMap, FollowsTheHostAddressOfAWindowSharedAgain) {
  rocjitsu::GuestRegionMap regions;
  HostPages first{};
  HostPages second{};
  ASSERT_TRUE(regions.insert(window(0x10000, kPageSize, kReadWrite), first.data()));

  EXPECT_FALSE(regions.insert(window(0x10000, kPageSize, kReadWrite), second.data()))
      << "the same window is not a new one";
  EXPECT_EQ(regions.size(), 1u);
  EXPECT_EQ(regions.host_span(0x10000 + 4, 4, simdojo::DmaAccess::Read), second.data() + 4);

  EXPECT_FALSE(regions.insert(window(0x10000, kPageSize, kReadWrite)));
  EXPECT_EQ(regions.host_span(0x10000 + 4, 4, simdojo::DmaAccess::Read), nullptr)
      << "shared again with no mapping, the window has no address to give out";
}

TEST(GuestRegionMap, FollowsTheHostAddressOfAWindowResharedWithANewLength) {
  rocjitsu::GuestRegionMap regions;
  HostPages first{};
  HostPages second{};
  ASSERT_TRUE(regions.insert(window(0x10000, kPageSize, kReadWrite), first.data()));

  EXPECT_TRUE(regions.insert(window(0x10000, 2 * kPageSize, kReadWrite), second.data()));
  EXPECT_EQ(regions.host_span(0x10000 + kPageSize, kPageSize, simdojo::DmaAccess::Read),
            second.data() + kPageSize)
      << "the second page exists only in the window as it was shared again";
}

// The transport unmaps a window as soon as it has been withdrawn, so an address
// handed out afterwards would point at memory that is gone.
TEST(GuestRegionMap, LocatesNothingInAWithdrawnWindow) {
  rocjitsu::GuestRegionMap regions;
  HostPages host{};
  HostPages remapped{};
  ASSERT_TRUE(regions.insert(window(0x10000, kPageSize, kReadWrite), host.data()));
  ASSERT_TRUE(regions.insert(window(0x20000, kPageSize, kReadWrite), host.data() + kPageSize));

  regions.erase(window(0x10000, kPageSize));

  EXPECT_EQ(regions.host_span(0x10000, 8, simdojo::DmaAccess::Read), nullptr);
  EXPECT_EQ(regions.host_span(0x20000, 8, simdojo::DmaAccess::Read), host.data() + kPageSize)
      << "withdrawing one window must not disturb another";

  ASSERT_TRUE(regions.insert(window(0x10000, kPageSize, kReadWrite), remapped.data()));
  EXPECT_EQ(regions.host_span(0x10000, 8, simdojo::DmaAccess::Read), remapped.data())
      << "shared again after being withdrawn, the window is found where it is now";

  regions.clear();

  EXPECT_EQ(regions.host_span(0x10000, 8, simdojo::DmaAccess::Read), nullptr);
  EXPECT_EQ(regions.host_span(0x20000, 8, simdojo::DmaAccess::Read), nullptr);
}

// An empty span holds no byte that could be outside a window, so it is located
// wherever its address is. An address in no window is still in no window.
TEST(GuestRegionMap, LocatesAnEmptySpanOnlyAtAnAddressInsideAWindow) {
  constexpr uint64_t kBase = 0x10000;
  rocjitsu::GuestRegionMap regions;
  HostPages host{};
  ASSERT_TRUE(regions.insert(window(kBase, kPageSize, kReadWrite), host.data()));
  ASSERT_TRUE(regions.insert(window(0x40000, 0, kReadWrite), host.data() + kPageSize));

  EXPECT_EQ(regions.host_span(kBase, 0, simdojo::DmaAccess::Read), host.data());
  EXPECT_EQ(regions.host_span(kBase + kPageSize - 1, 0, simdojo::DmaAccess::Read),
            host.data() + kPageSize - 1);
  EXPECT_EQ(regions.host_span(kBase + kPageSize, 0, simdojo::DmaAccess::Read), nullptr)
      << "the address past the last byte is in no window";
  EXPECT_EQ(regions.host_span(0x40000, 0, simdojo::DmaAccess::Read), nullptr)
      << "a window with no bytes contains no address";
}

// The same wrap the distance lookup guards against: a window whose length runs
// past the top of the address space would otherwise be found to contain
// addresses it does not, and those would be turned into host addresses.
TEST(GuestRegionMap, LocatesNothingInAWindowThatWrapsTheAddressSpace) {
  rocjitsu::GuestRegionMap regions;
  HostPages host{};
  constexpr uint64_t kNearTop = std::numeric_limits<uint64_t>::max() - 0xfff;
  ASSERT_TRUE(regions.insert(window(kNearTop, 0x2000, kReadWrite), host.data()));

  EXPECT_EQ(regions.host_span(kNearTop, 8, simdojo::DmaAccess::Read), nullptr);
  EXPECT_EQ(regions.host_span(0x10, 8, simdojo::DmaAccess::Read), nullptr);
}

// A window is shared for reading, writing or both, and a mapping in this
// process does not enforce that by itself. A copy the window does not allow has
// to be turned away here so that the transport, which does enforce it, decides.
TEST(GuestRegionMap, RefusesAnAccessTheWindowDoesNotGrant) {
  rocjitsu::GuestRegionMap regions;
  HostPages host{};
  ASSERT_TRUE(regions.insert(window(0x10000, kPageSize, simdojo::DmaAccess::Read), host.data()));
  ASSERT_TRUE(regions.insert(window(0x20000, kPageSize, simdojo::DmaAccess::Write),
                             host.data() + kPageSize));
  ASSERT_TRUE(regions.insert(window(0x30000, kPageSize, kReadWrite), host.data()));
  ASSERT_TRUE(regions.insert(window(0x40000, kPageSize), host.data()));

  EXPECT_EQ(regions.host_span(0x10000, 8, simdojo::DmaAccess::Read), host.data());
  EXPECT_EQ(regions.host_span(0x10000, 8, simdojo::DmaAccess::Write), nullptr);
  EXPECT_EQ(regions.host_span(0x10000, 8, kReadWrite), nullptr)
      << "half of what is needed is not enough";

  EXPECT_EQ(regions.host_span(0x20000, 8, simdojo::DmaAccess::Write), host.data() + kPageSize);
  EXPECT_EQ(regions.host_span(0x20000, 8, simdojo::DmaAccess::Read), nullptr);

  EXPECT_EQ(regions.host_span(0x30000, 8, simdojo::DmaAccess::Read), host.data());
  EXPECT_EQ(regions.host_span(0x30000, 8, simdojo::DmaAccess::Write), host.data());
  EXPECT_EQ(regions.host_span(0x30000, 8, kReadWrite), host.data());

  EXPECT_EQ(regions.host_span(0x40000, 8, simdojo::DmaAccess::Read), nullptr)
      << "a window granting nothing";
  EXPECT_EQ(regions.host_span(0x40000, 8, simdojo::DmaAccess::Write), nullptr);
}
