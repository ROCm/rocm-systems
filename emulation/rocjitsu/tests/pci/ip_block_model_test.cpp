// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pci/ip_block_model.h"

#include "rocjitsu/vm/amdgpu/pci/register_aperture.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr uint64_t kApertureBytes = 512 * 1024;

TEST(RegisterAperture, RefusesAnOffsetWhoseRegisterWidthWraps) {
  rocjitsu::RegisterAperture registers("gpu", kApertureBytes);
  constexpr uint64_t kWrappingOffset = std::numeric_limits<uint64_t>::max() - 3;

  EXPECT_FALSE(registers.holds(kWrappingOffset));
  EXPECT_FALSE(registers.define(kWrappingOffset, 0x12345678));
  EXPECT_EQ(registers.value(kWrappingOffset), 0);
  EXPECT_FALSE(registers.store(kWrappingOffset, 0x87654321));
}

// Blocks legitimately share register segments on this family -- GC and SDMA0
// publish identical bases, as do the two management processors -- and stay
// apart only in the offsets they claim within them. So an absolute-dword map is
// well defined, and this is the check that says an overlap in it is never a
// shared segment showing through. Two models answering one register is silent:
// whichever defined it last wins, and the block that lost stalls the driver on
// a register reading as somebody else's.
TEST(RegisterClaims, AcceptsDisjointRangesAndRefusesOverlappingOnes) {
  // Adjacent, which is what two blocks in one segment look like: the second
  // begins exactly where the first ends.
  EXPECT_EQ(
      rocjitsu::overlapping_claim({{.owner = "a hub", .first_dword = 0x100, .count = 18},
                                   {.owner = "another hub", .first_dword = 0x112, .count = 18}}),
      "")
      << "two ranges that merely touch were reported as overlapping";

  // Out of order, because the device builds the map in whatever order it
  // happens to build models.
  EXPECT_EQ(rocjitsu::overlapping_claim({{.owner = "later", .first_dword = 0x2000, .count = 4},
                                         {.owner = "earlier", .first_dword = 0x10, .count = 4}}),
            "");

  EXPECT_NE(rocjitsu::overlapping_claim(
                {{.owner = "a hub", .first_dword = 0x100, .count = 18},
                 {.owner = "the interrupt handler", .first_dword = 0x111, .count = 1}}),
            "")
      << "a claim reaching one register into another's range was accepted";

  // Identical ranges, the way a profile that gave two instances of one block the
  // same segments would present.
  const std::string same =
      rocjitsu::overlapping_claim({{.owner = "instance 0", .first_dword = 0x50, .count = 8},
                                   {.owner = "instance 1", .first_dword = 0x50, .count = 8}});
  EXPECT_NE(same, "");
  EXPECT_NE(same.find("instance 0"), std::string::npos) << "the report names neither claimant";
  EXPECT_NE(same.find("instance 1"), std::string::npos);
}

// A model addresses its registers the way its own offset header does, and the
// window is what turns that into a byte offset. The segment is the dangerous
// half: it comes from a discovery profile, and one near the top of the range
// multiplies into a small byte offset that would pass an aperture bounds test
// and land on an unrelated register.
TEST(IpRegisterWindow, ResolvesThroughTheBlocksOwnSegmentsAndRefusesOnesThatWrap) {
  rocjitsu::RegisterAperture registers("gpu", kApertureBytes);
  const std::vector<uint64_t> segments = {0x10a0, 0x2000};
  const rocjitsu::IpRegisterWindow window(registers, segments, "gpu");

  ASSERT_TRUE(window.resolve(0, 0x80).has_value());
  EXPECT_EQ(*window.resolve(0, 0x80), (0x10a0 + 0x80) * 4);
  EXPECT_EQ(*window.resolve(1, 0x80), (0x2000 + 0x80) * 4)
      << "the second segment resolved against the first";

  EXPECT_FALSE(window.resolve(2, 0).has_value()) << "a segment the block never published";

  const std::vector<uint64_t> wrapping = {std::numeric_limits<uint64_t>::max() - 0x7f};
  rocjitsu::IpRegisterWindow wrapped(registers, wrapping, "gpu");
  EXPECT_FALSE(wrapped.resolve(0, 0x80).has_value())
      << "a segment 0x80 dwords short of 2^64 resolved onto byte zero";
  EXPECT_FALSE(wrapped.define(0, 0x80, 1)) << "and defined a register there";
  EXPECT_FALSE(registers.modelled(0)) << "byte zero was modelled by a block that cannot reach it";
}

} // namespace
