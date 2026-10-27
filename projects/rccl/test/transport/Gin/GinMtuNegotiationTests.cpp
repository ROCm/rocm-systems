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

TEST(GinGidSelection, PrefersRoutableGidWithFlid) {
  unsigned char gids[2][16] = {{0xfe, 0x80}, {0xfe, 0x80, 0, 0, 0x00, 0x2a}};
  unsigned idx[2] = {0, 1}, port[2] = {1, 1};
  EXPECT_EQ(ginTestSelectIbGidIndex(gids, idx, port, 2, 1, 0, 1), 1);
}

TEST(GinGidSelection, FallsBackWhenRoutableFlidIsZero) {
  unsigned char gids[2][16] = {{0xfe, 0x80}, {0xfe, 0x80}};
  unsigned idx[2] = {0, 1}, port[2] = {1, 1};
  EXPECT_EQ(ginTestSelectIbGidIndex(gids, idx, port, 2, 1, 0, 1), 0);
}

TEST(GinGidSelection, FallsBackWhenRoutableIndexAbsent) {
  unsigned char gids[1][16] = {{0xfe, 0x80}};
  unsigned idx[1] = {0}, port[1] = {1};
  EXPECT_EQ(ginTestSelectIbGidIndex(gids, idx, port, 1, 1, 0, 1), 0);
  EXPECT_EQ(ginTestSelectIbGidIndex(gids, idx, port, 1, 1, 0, -1), 0);
}

// Dual-port HCA: port 2's entries precede port 1's routable GID, so array position 1 is not
// port 1 / gid_index 1.
TEST(GinGidSelection, MatchesRoutableGidByPortAndIndexNotPosition) {
  unsigned char gids[3][16] = {{0xfe, 0x80}, {0xfe, 0x80, 0, 0, 0x00, 0x77}, {0xfe, 0x80, 0, 0, 0x00, 0x2a}};
  unsigned idx[3] = {0, 1, 1}, port[3] = {1, 2, 1};
  EXPECT_EQ(ginTestSelectIbGidIndex(gids, idx, port, 3, 1, 0, 1), 1);
}

// Holes in the table: position 1 holds gid_index 2, so the routable gid_index 1 does not exist.
TEST(GinGidSelection, TableHolesDoNotAliasRoutableIndex) {
  unsigned char gids[2][16] = {{0xfe, 0x80}, {0xfe, 0x80, 0, 0, 0x00, 0x2a}};
  unsigned idx[2] = {0, 2}, port[2] = {1, 1};
  EXPECT_EQ(ginTestSelectIbGidIndex(gids, idx, port, 2, 1, 0, 1), 0);
}

}  // namespace

#else  // !ENABLE_ROCSHMEM_GIN

TEST(GinMtuNegotiation, SkippedWithoutGinBuild) {
  GTEST_SKIP() << "GIN MTU negotiation tests require ENABLE_ROCSHMEM_GIN";
}

#endif  // ENABLE_ROCSHMEM_GIN
