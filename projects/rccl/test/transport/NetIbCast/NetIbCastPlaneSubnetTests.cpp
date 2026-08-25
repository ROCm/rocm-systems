/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <cstdint>
#include <cstring>
#include <arpa/inet.h>
#include <gtest/gtest.h>

#include "net_ib_cast_inspect.h"

namespace {

// IPv4-mapped RoCE GID (::ffff:a.b.c.d).
void MakeGidV4(uint8_t g[16], uint8_t a, uint8_t b, uint8_t c, uint8_t d) {
  memset(g, 0, 16);
  g[10] = 0xff;
  g[11] = 0xff;
  g[12] = a; g[13] = b; g[14] = c; g[15] = d;
}

// Native IPv6 GID with a chosen 64-bit subnet prefix (g[15] = interface id).
void MakeGidV6(uint8_t g[16], uint8_t prefix0, uint8_t iface) {
  memset(g, 0, 16);
  g[0] = prefix0;
  g[1] = 0x01;
  g[15] = iface;
}

// plane/rail: IbCastGetPlaneIndex dedups plane IDs into a compact index space.
TEST(NetIbCastPlaneRail, GetPlaneIndexDedup) {
  int16_t count = 1;
  int16_t planes[14] = {-1};
  int16_t idx = -1;

  const int   seq[]    = {-1, 5, 5, 7, -1, 5};
  const int16_t expect[] = { 0, 1, 1, 2,  0, 1};
  for (size_t i = 0; i < sizeof(seq) / sizeof(seq[0]); i++) {
    ASSERT_EQ(ncclIbCastTestGetPlaneIndex(seq[i], &count, planes, &idx), ncclSuccess);
    EXPECT_EQ(idx, expect[i]) << "plane id " << seq[i];
  }
  EXPECT_EQ(count, 3);  // unique planes: {-1, 5, 7}
}

TEST(NetIbCastPlaneRail, GetPlaneIndexRejectsVirtBit) {
  int16_t count = 1, planes[14] = {-1}, idx = 0;
  EXPECT_NE(ncclIbCastTestGetPlaneIndex(0x4000, &count, planes, &idx), ncclSuccess);  // 0x4000 = VIRT_BIT
}

TEST(NetIbCastPlaneRail, GetPlaneIndexNullArgs) {
  int16_t count = 1, planes[14] = {-1}, idx = 0;
  EXPECT_NE(ncclIbCastTestGetPlaneIndex(0, nullptr, planes, &idx), ncclSuccess);
  EXPECT_NE(ncclIbCastTestGetPlaneIndex(0, &count, nullptr, &idx), ncclSuccess);
  EXPECT_NE(ncclIbCastTestGetPlaneIndex(0, &count, planes, nullptr), ncclSuccess);
}

// subnet: gidSameSubnet.
TEST(NetIbCastSubnet, GidSameSubnetIPv4) {
  uint8_t a[16], b[16];
  MakeGidV4(a, 192, 168, 1, 10);
  MakeGidV4(b, 192, 168, 1, 20);
  EXPECT_EQ(ncclIbCastTestGidSameSubnet(a, b, 24), 1);  // same /24

  MakeGidV4(b, 192, 168, 2, 20);
  EXPECT_EQ(ncclIbCastTestGidSameSubnet(a, b, 24), 0);  // different /24
  EXPECT_EQ(ncclIbCastTestGidSameSubnet(a, b, 16), 1);  // same /16
}

TEST(NetIbCastSubnet, GidSameSubnetIPv6) {
  uint8_t c[16], d[16];
  MakeGidV6(c, 0x20, 1);
  MakeGidV6(d, 0x20, 2);
  EXPECT_EQ(ncclIbCastTestGidSameSubnet(c, d, 64), 1);  // same 64-bit prefix

  MakeGidV6(d, 0x30, 2);
  EXPECT_EQ(ncclIbCastTestGidSameSubnet(c, d, 64), 0);  // different prefix
}

TEST(NetIbCastSubnet, GidSameSubnetFamilyMismatch) {
  uint8_t v4[16], v6[16];
  MakeGidV4(v4, 10, 0, 0, 1);
  MakeGidV6(v6, 0x20, 1);
  EXPECT_EQ(ncclIbCastTestGidSameSubnet(v4, v6, 24), 0);  // AF_INET vs AF_INET6
}

// subnet: subnetMatchesAny skips invalid/zero GIDs.
TEST(NetIbCastSubnet, SubnetMatchesAny) {
  uint8_t local[16];
  MakeGidV4(local, 10, 0, 5, 1);

  uint8_t zero[16];   memset(zero, 0, 16);            // invalid -> skipped
  uint8_t other[16];  MakeGidV4(other, 10, 0, 9, 9);
  uint8_t match[16];  MakeGidV4(match, 10, 0, 5, 200);

  uint8_t rem[3 * 16];
  memcpy(rem + 0,  zero,  16);
  memcpy(rem + 16, other, 16);
  memcpy(rem + 32, match, 16);
  EXPECT_EQ(ncclIbCastTestSubnetMatchesAny(local, rem, 3, 24), 1);

  memcpy(rem + 32, other, 16);  // no remote shares the subnet now
  EXPECT_EQ(ncclIbCastTestSubnetMatchesAny(local, rem, 3, 24), 0);
}

}  // namespace
