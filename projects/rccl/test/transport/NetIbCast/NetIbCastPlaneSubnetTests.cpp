/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only regression tests for the net_ib_cast features backported from
// NCCL v2.30.7 (ROCm/rocm-systems#8514):
//   - plane/rail detection : IbCastGetPlaneIndex
//   - subnet detection     : gidSameSubnet / subnetMatchesAny
//
// These exercise the *real* internal helpers through the test-only wrappers in
// net_ib_cast_inspect.h (no RDMA HW / MPI / GPU required). The GIN_IB_TC traffic
// class and the GRH addressing + device-override paths are exercised end-to-end
// by the MPI suite (transport/NetIbMPI/*), since they require live QPs.

#include <cstdint>
#include <cstring>
#include <arpa/inet.h>
#include <gtest/gtest.h>

// net_ib_cast_inspect.h already wraps its declarations in extern "C" and pulls
// in nccl.h (with its HIP/C++ templates) *outside* that guard, so it must be
// included directly — wrapping it in an extra extern "C" forces the HIP
// templates into C linkage and breaks the build.
#include "net_ib_cast_inspect.h"

namespace {

// Build an IPv4-mapped RoCE GID (::ffff:a.b.c.d).
void MakeGidV4(uint8_t g[16], uint8_t a, uint8_t b, uint8_t c, uint8_t d) {
  memset(g, 0, 16);
  g[10] = 0xff;
  g[11] = 0xff;
  g[12] = a; g[13] = b; g[14] = c; g[15] = d;
}

// Build a native (non IPv4-mapped) IPv6 GID with a chosen 64-bit subnet prefix.
void MakeGidV6(uint8_t g[16], uint8_t prefix0, uint8_t iface) {
  memset(g, 0, 16);
  g[0] = prefix0;
  g[1] = 0x01;   // ensure it is not an all-zero / IPv4-mapped pattern
  g[15] = iface; // interface id (not part of the subnet prefix)
}

// ---------------------------------------------------------------------------
// plane/rail: IbCastGetPlaneIndex dedups plane IDs into a compact index space.
// ---------------------------------------------------------------------------
TEST(NetIbCastPlaneRail, GetPlaneIndexDedup) {
  int16_t count = 1;
  int16_t planes[14] = {-1};  // slot 0 seeded with NCCL_NET_ID_UNDEF
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
  // 0x4000 == NCCL_IB_PLANE_VIRT_BIT: reserved, must be rejected.
  EXPECT_NE(ncclIbCastTestGetPlaneIndex(0x4000, &count, planes, &idx), ncclSuccess);
}

TEST(NetIbCastPlaneRail, GetPlaneIndexNullArgs) {
  int16_t count = 1, planes[14] = {-1}, idx = 0;
  EXPECT_NE(ncclIbCastTestGetPlaneIndex(0, nullptr, planes, &idx), ncclSuccess);
  EXPECT_NE(ncclIbCastTestGetPlaneIndex(0, &count, nullptr, &idx), ncclSuccess);
  EXPECT_NE(ncclIbCastTestGetPlaneIndex(0, &count, planes, nullptr), ncclSuccess);
}

// ---------------------------------------------------------------------------
// subnet detection: gidSameSubnet
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// subnet detection: subnetMatchesAny (skips invalid/zero GIDs)
// ---------------------------------------------------------------------------
TEST(NetIbCastSubnet, SubnetMatchesAny) {
  uint8_t local[16];
  MakeGidV4(local, 10, 0, 5, 1);

  uint8_t zero[16];   memset(zero, 0, 16);        // invalid GID -> skipped
  uint8_t other[16];  MakeGidV4(other, 10, 0, 9, 9);  // different /24
  uint8_t match[16];  MakeGidV4(match, 10, 0, 5, 200); // same /24

  uint8_t rem[3 * 16];
  memcpy(rem + 0,  zero,  16);
  memcpy(rem + 16, other, 16);
  memcpy(rem + 32, match, 16);
  EXPECT_EQ(ncclIbCastTestSubnetMatchesAny(local, rem, 3, 24), 1);

  memcpy(rem + 32, other, 16);  // now no remote shares the subnet
  EXPECT_EQ(ncclIbCastTestSubnetMatchesAny(local, rem, 3, 24), 0);
}

}  // namespace
