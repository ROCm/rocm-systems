/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*
 * Regression tests for the IPv4 subnet-match used by ncclFindInterfaceMatchSubnet()
 * (rcclMatchSubnetV4 in socket.h); guards against re-inverting it (upstream NCCL PR #2047).
 */

#include "socket.h"
#include "gtest/gtest.h"

#include <arpa/inet.h>
#include <cstring>
#include <netinet/in.h>
#include <sys/socket.h>

namespace RcclUnitTesting {

namespace {

// Parse dotted-quad text (e.g. "192.168.1.10") into an in_addr. A parse failure is
// fatal: otherwise it would fall through with a zeroed 0.0.0.0 and could make a
// later assertion pass for the wrong reason.
void ParseV4(const char* s, struct in_addr* out) {
  memset(out, 0, sizeof(*out));
  ASSERT_EQ(inet_pton(AF_INET, s, out), 1) << "inet_pton failed for '" << s << "'";
}

} // namespace

// Declare an in_addr named 'var' holding the address parsed from 'str'.
#define V4(var, str)                                                                                                   \
  struct in_addr var;                                                                                                  \
  ASSERT_NO_FATAL_FAILURE(ParseV4(str, &var))

// Same /24 subnet -> must match. This is the core assertion that would FAIL if
// the boolean were re-inverted (the PR #2047 defect).
TEST(MatchSubnetTests, IPv4SameSubnetMatches) {
  V4(mask, "255.255.255.0");
  V4(a1, "192.168.1.10");
  V4(a2, "192.168.1.20");
  V4(b1, "10.0.5.1");
  V4(b2, "10.0.5.254");
  EXPECT_TRUE(rcclMatchSubnetV4(a1, a2, mask));
  EXPECT_TRUE(rcclMatchSubnetV4(b1, b2, mask));
}

// Different /24 subnets -> must NOT match.
TEST(MatchSubnetTests, IPv4DifferentSubnetDoesNotMatch) {
  V4(mask, "255.255.255.0");
  V4(a1, "192.168.1.10");
  V4(a2, "192.168.2.20");
  V4(b1, "10.0.5.1");
  V4(b2, "10.0.6.1");
  EXPECT_FALSE(rcclMatchSubnetV4(a1, a2, mask));
  EXPECT_FALSE(rcclMatchSubnetV4(b1, b2, mask));
}

// Netmask width changes the answer: a wider /16 makes .1.x and .2.x match.
TEST(MatchSubnetTests, IPv4NetmaskWidthMatters) {
  V4(mask24, "255.255.255.0");
  V4(mask16, "255.255.0.0");
  V4(a, "172.16.1.5");
  V4(b, "172.16.2.5");
  EXPECT_FALSE(rcclMatchSubnetV4(a, b, mask24));
  EXPECT_TRUE(rcclMatchSubnetV4(a, b, mask16));
}

// Mask corner cases: all-zero matches everything, all-ones matches only an exact
// address, and a /25 splits a /24 in half.
TEST(MatchSubnetTests, IPv4MaskEdgeCases) {
  V4(maskAny, "0.0.0.0");
  V4(maskExact, "255.255.255.255");
  V4(mask25, "255.255.255.128");
  V4(mask24, "255.255.255.0");

  // 0.0.0.0: every address is on the same "subnet".
  V4(far1, "192.168.1.1");
  V4(far2, "10.0.0.1");
  EXPECT_TRUE(rcclMatchSubnetV4(far1, far2, maskAny));
  EXPECT_TRUE(rcclMatchSubnetV4(far1, far1, maskAny));

  // 255.255.255.255: only an exact address match qualifies.
  V4(host, "10.0.5.1");
  V4(hostSame, "10.0.5.1");
  V4(hostNext, "10.0.5.2");
  EXPECT_TRUE(rcclMatchSubnetV4(host, hostSame, maskExact));
  EXPECT_FALSE(rcclMatchSubnetV4(host, hostNext, maskExact));

  // /25 boundary: .10 and .120 share the lower half, .200 is in the upper half.
  V4(lowA, "10.0.5.10");
  V4(lowB, "10.0.5.120");
  V4(high, "10.0.5.200");
  EXPECT_TRUE(rcclMatchSubnetV4(lowA, lowB, mask25));
  EXPECT_FALSE(rcclMatchSubnetV4(lowA, high, mask25));
  // The same pair does match once the mask widens to /24.
  EXPECT_TRUE(rcclMatchSubnetV4(lowA, high, mask24));

  // Network and broadcast addresses of a /24 are still on that subnet.
  V4(network, "10.0.5.0");
  V4(broadcast, "10.0.5.255");
  EXPECT_TRUE(rcclMatchSubnetV4(network, broadcast, mask24));
}

// Interface-selection scenario: scanning candidates must pick exactly the one on
// the remote's subnet (as ncclFindInterfaceMatchSubnet does).
TEST(MatchSubnetTests, IPv4SelectsCorrectInterface) {
  V4(mask, "255.255.255.0");
  V4(remote, "10.0.5.42");

  // Candidate local interface addresses.
  V4(eth0, "192.168.1.1"); // wrong subnet
  V4(eth1, "10.0.5.1");    // correct subnet
  V4(eth2, "172.16.0.1");  // wrong subnet

  EXPECT_FALSE(rcclMatchSubnetV4(eth0, remote, mask));
  EXPECT_TRUE(rcclMatchSubnetV4(eth1, remote, mask));
  EXPECT_FALSE(rcclMatchSubnetV4(eth2, remote, mask));

  // Emulate the scan loop picking the first matching interface.
  struct in_addr candidates[] = {eth0, eth1, eth2};
  int selected = -1;
  for (int i = 0; i < 3; i++) {
    if (rcclMatchSubnetV4(candidates[i], remote, mask)) {
      selected = i;
      break;
    }
  }
  EXPECT_EQ(selected, 1) << "Expected to select eth1 (the interface on the remote's subnet)";
}

} // namespace RcclUnitTesting
