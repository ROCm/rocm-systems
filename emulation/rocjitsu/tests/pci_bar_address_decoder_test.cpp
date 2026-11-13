// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "simdojo/components/pci_bar_address_decoder.h"

#include <gtest/gtest.h>

#include <cstdint>

namespace {

TEST(PciBarAddressDecoder, ResolvesOnlyOneLivePeerBarWithTheRequestedRights) {
  simdojo::PciBarAddressDecoder decoder;
  ASSERT_TRUE(decoder.publish({.function = 0,
                               .bar = 0,
                               .base = 0xc080000000,
                               .size = 0x40000000,
                               .rights = simdojo::DmaAccess::Read | simdojo::DmaAccess::Write,
                               .generation = 7}));

  const auto match = decoder.resolve(1, 0xc080001230, 0x100, simdojo::DmaAccess::Read);
  ASSERT_TRUE(match.has_value());
  EXPECT_EQ(match->function, 0u);
  EXPECT_EQ(match->bar, 0u);
  EXPECT_EQ(match->offset, 0x1230u);
  EXPECT_EQ(match->generation, 7u);
  EXPECT_FALSE(decoder.resolve(0, 0xc080001230, 0x100, simdojo::DmaAccess::Read));
  EXPECT_FALSE(decoder.resolve(1, 0xc0bfffff80, 0x100, simdojo::DmaAccess::Read));
  EXPECT_FALSE(decoder.resolve(1, UINT64_MAX - 3, 8, simdojo::DmaAccess::Read));

  ASSERT_TRUE(decoder.publish({.function = 0,
                               .bar = 0,
                               .base = 0xc080000000,
                               .size = 0x40000000,
                               .rights = simdojo::DmaAccess::Read,
                               .generation = 7}));
  EXPECT_FALSE(decoder.resolve(1, 0xc080001230, 0x100, simdojo::DmaAccess::Write));
  decoder.revoke(0, 0);
  EXPECT_FALSE(decoder.resolve(1, 0xc080001230, 0x100, simdojo::DmaAccess::Read));
}

TEST(PciBarAddressDecoder, RejectsAmbiguousAndPartiallyOverlappingWindows) {
  simdojo::PciBarAddressDecoder decoder;
  ASSERT_TRUE(decoder.publish({.function = 0,
                               .bar = 0,
                               .base = 0x100000,
                               .size = 0x10000,
                               .rights = simdojo::DmaAccess::Read,
                               .generation = 1}));
  ASSERT_TRUE(decoder.publish({.function = 2,
                               .bar = 0,
                               .base = 0x108000,
                               .size = 0x10000,
                               .rights = simdojo::DmaAccess::Read,
                               .generation = 2}));
  EXPECT_FALSE(decoder.resolve(1, 0x108000, 0x1000, simdojo::DmaAccess::Read));
  EXPECT_FALSE(decoder.resolve(1, 0x107000, 0x2000, simdojo::DmaAccess::Read));
  const auto unique = decoder.resolve(1, 0x110000, 0x1000, simdojo::DmaAccess::Read);
  ASSERT_TRUE(unique);
  EXPECT_EQ(unique->function, 2u);
  EXPECT_EQ(unique->offset, 0x8000u);
}

} // namespace
