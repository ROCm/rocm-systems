/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#if defined(ENABLE_ROCSHMEM_GIN)

#include <cstring>

#include "ibvcore.h"
#include "ib_gid_utils.h"
#include "gin_rocshmem_gda_inspect.h"

namespace {

TEST(GinMtuNegotiation, IbvMtuToIntValidValues) {
  EXPECT_EQ(ginTestIbvMtuToInt(IBV_MTU_256), 256);
  EXPECT_EQ(ginTestIbvMtuToInt(IBV_MTU_512), 512);
  EXPECT_EQ(ginTestIbvMtuToInt(IBV_MTU_1024), 1024);
  EXPECT_EQ(ginTestIbvMtuToInt(IBV_MTU_2048), 2048);
  EXPECT_EQ(ginTestIbvMtuToInt(IBV_MTU_4096), 4096);
}

TEST(GinMtuNegotiation, IbvMtuToIntInvalidValue) {
  EXPECT_EQ(ginTestIbvMtuToInt(0), 0);
  EXPECT_EQ(ginTestIbvMtuToInt(6), 0);
}

TEST(GinMtuNegotiation, NegotiatePicksSmallerRemote) {
  int out = -1;
  ASSERT_EQ(ginTestNegotiatePathMtu(IBV_MTU_4096, IBV_MTU_1024, &out), 0);
  EXPECT_EQ(out, 1024);
}

TEST(GinMtuNegotiation, NegotiatePicksSmallerLocal) {
  int out = -1;
  ASSERT_EQ(ginTestNegotiatePathMtu(IBV_MTU_1024, IBV_MTU_4096, &out), 0);
  EXPECT_EQ(out, 1024);
}

TEST(GinMtuNegotiation, NegotiateEqualMtus) {
  int out = -1;
  ASSERT_EQ(ginTestNegotiatePathMtu(IBV_MTU_4096, IBV_MTU_4096, &out), 0);
  EXPECT_EQ(out, 4096);
}

TEST(GinMtuNegotiation, NegotiateRejectsInvalidRemoteMtu) {
  int out = -1;
  EXPECT_EQ(ginTestNegotiatePathMtu(IBV_MTU_4096, 0, &out), -1);
  EXPECT_EQ(out, -1);  // untouched on failure
}

TEST(GinMtuNegotiation, NegotiateRejectsInvalidLocalMtu) {
  int out = -1;
  EXPECT_EQ(ginTestNegotiatePathMtu(0, IBV_MTU_4096, &out), -1);
  EXPECT_EQ(out, -1);  // untouched on failure
}

// GID raw layout: bytes 4-5 hold the FLID (network order), bytes 6-7 the subnet prefix.
TEST(GinGidSelection, GidFlidAndSubnetExtraction) {
  uint8_t raw[16] = {0xfe, 0x80, 0, 0, 0x12, 0x34, 0xab, 0xcd};
  EXPECT_EQ(ncclIbGidFlid(raw), 0x1234);
  uint64_t prefix;
  memcpy(&prefix, raw, sizeof(prefix));
  EXPECT_EQ(ncclIbGidSubnetPrefix16(prefix), 0xabcd);
}

constexpr unsigned kGidIb = 0, kGidRoceV1 = 1, kGidRoceV2 = 2;

TEST(GinGidSelection, IbPrefersRoutableGidWithFlid) {
  unsigned char gids[2][16] = {{0xfe, 0x80}, {0xfe, 0x80, 0, 0, 0x00, 0x2a}};
  unsigned idx[2] = {0, 1}, port[2] = {1, 1}, type[2] = {kGidIb, kGidIb};
  EXPECT_EQ(ginTestSelectGidIndex(gids, idx, port, type, 2, 1, 1), 1);
}

TEST(GinGidSelection, IbFallsBackWhenRoutableFlidIsZero) {
  unsigned char gids[2][16] = {{0xfe, 0x80}, {0xfe, 0x80}};
  unsigned idx[2] = {0, 1}, port[2] = {1, 1}, type[2] = {kGidIb, kGidIb};
  EXPECT_EQ(ginTestSelectGidIndex(gids, idx, port, type, 2, 1, 1), 0);
}

TEST(GinGidSelection, IbFallsBackWhenRoutableIndexAbsent) {
  unsigned char gids[1][16] = {{0xfe, 0x80}};
  unsigned idx[1] = {0}, port[1] = {1}, type[1] = {kGidIb};
  EXPECT_EQ(ginTestSelectGidIndex(gids, idx, port, type, 1, 1, 1), 0);
  EXPECT_EQ(ginTestSelectGidIndex(gids, idx, port, type, 1, 1, -1), 0);
}

// Dual-port HCA: a port-2 decoy with gid_index 1 and a nonzero FLID precedes port 1's routable
// GID, whose FLID is zero. Without the port check the decoy would be picked (returning 1).
TEST(GinGidSelection, IbIgnoresOtherPortsRoutableGid) {
  unsigned char gids[3][16] = {{0xfe, 0x80}, {0xfe, 0x80, 0, 0, 0x00, 0x77}, {0xfe, 0x80}};
  unsigned idx[3] = {0, 1, 1}, port[3] = {1, 2, 1}, type[3] = {kGidIb, kGidIb, kGidIb};
  EXPECT_EQ(ginTestSelectGidIndex(gids, idx, port, type, 3, 1, 1), 0);
}

// Position 2 holds port 1 / gid_index 1; matching by position would pick the port-2 entry.
TEST(GinGidSelection, IbMatchesRoutableGidByPortAndIndexNotPosition) {
  unsigned char gids[3][16] = {{0xfe, 0x80}, {0xfe, 0x80, 0, 0, 0x00, 0x77}, {0xfe, 0x80, 0, 0, 0x00, 0x2a}};
  unsigned idx[3] = {0, 1, 1}, port[3] = {1, 2, 1}, type[3] = {kGidIb, kGidIb, kGidIb};
  EXPECT_EQ(ginTestSelectGidIndex(gids, idx, port, type, 3, 1, 1), 1);
}

// Position 1 holds gid_index 2, so the routable gid_index 1 does not exist.
TEST(GinGidSelection, IbTableHolesDoNotAliasRoutableIndex) {
  unsigned char gids[2][16] = {{0xfe, 0x80}, {0xfe, 0x80, 0, 0, 0x00, 0x2a}};
  unsigned idx[2] = {0, 2}, port[2] = {1, 1}, type[2] = {kGidIb, kGidIb};
  EXPECT_EQ(ginTestSelectGidIndex(gids, idx, port, type, 2, 1, 1), 0);
}

// The stored index is the entry's gid_index, not its position in the table.
TEST(GinGidSelection, IbReportsGidIndexNotPositionWhenPortOneStartsLate) {
  unsigned char gids[2][16] = {{0xfe, 0x80}, {0xfe, 0x80}};
  unsigned idx[2] = {0, 3}, port[2] = {2, 1}, type[2] = {kGidIb, kGidIb};
  EXPECT_EQ(ginTestSelectGidIndex(gids, idx, port, type, 2, 1, 1), 3);
}

TEST(GinGidSelection, RoceSkipsLinkLocalAndAllZeroAndPrefersV2) {
  unsigned char gids[4][16] = {{0}, {0xfe, 0x80, 1}, {0x20, 0x01, 1}, {0x20, 0x01, 2}};
  unsigned idx[4] = {0, 1, 2, 3}, port[4] = {1, 1, 1, 1}, type[4] = {kGidRoceV2, kGidRoceV2, kGidRoceV1, kGidRoceV2};
  EXPECT_EQ(ginTestSelectGidIndex(gids, idx, port, type, 4, 1, 1), 3);
}

TEST(GinGidSelection, RoceIgnoresOtherPortAndReportsGidIndexNotPosition) {
  unsigned char gids[2][16] = {{0x20, 0x01, 9}, {0x20, 0x01, 1}};
  unsigned idx[2] = {0, 5}, port[2] = {2, 1}, type[2] = {kGidRoceV2, kGidRoceV1};
  EXPECT_EQ(ginTestSelectGidIndex(gids, idx, port, type, 2, 1, 1), 5);
}

TEST(GinGidSelection, ReturnsMinusOneWhenNoUsableEntry) {
  unsigned char gids[2][16] = {{0}, {0xfe, 0x80}};
  unsigned idx[2] = {0, 1}, port[2] = {1, 1}, type[2] = {kGidRoceV2, kGidRoceV2};
  EXPECT_EQ(ginTestSelectGidIndex(gids, idx, port, type, 2, 1, 1), -1);
}

}  // namespace

#else  // !ENABLE_ROCSHMEM_GIN

TEST(GinMtuNegotiation, SkippedWithoutGinBuild) {
  GTEST_SKIP() << "GIN MTU negotiation tests require ENABLE_ROCSHMEM_GIN";
}

#endif  // ENABLE_ROCSHMEM_GIN
