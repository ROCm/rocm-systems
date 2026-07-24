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

// Build an in_addr from dotted-quad text (e.g. "192.168.1.10").
struct in_addr V4(const char* s) {
  struct in_addr a;
  memset(&a, 0, sizeof(a));
  EXPECT_EQ(inet_pton(AF_INET, s, &a), 1) << "inet_pton failed for '" << s << "'";
  return a;
}

} // namespace

// Same /24 subnet -> must match. This is the core assertion that would FAIL if
// the boolean were re-inverted (the PR #2047 defect).
TEST(MatchSubnetTests, IPv4SameSubnetMatches) {
  struct in_addr mask = V4("255.255.255.0");
  EXPECT_TRUE(rcclMatchSubnetV4(V4("192.168.1.10"), V4("192.168.1.20"), mask));
  EXPECT_TRUE(rcclMatchSubnetV4(V4("10.0.5.1"), V4("10.0.5.254"), mask));
}

// Different /24 subnets -> must NOT match.
TEST(MatchSubnetTests, IPv4DifferentSubnetDoesNotMatch) {
  struct in_addr mask = V4("255.255.255.0");
  EXPECT_FALSE(rcclMatchSubnetV4(V4("192.168.1.10"), V4("192.168.2.20"), mask));
  EXPECT_FALSE(rcclMatchSubnetV4(V4("10.0.5.1"), V4("10.0.6.1"), mask));
}

// Netmask width changes the answer: a wider /16 makes .1.x and .2.x match.
TEST(MatchSubnetTests, IPv4NetmaskWidthMatters) {
  EXPECT_FALSE(rcclMatchSubnetV4(V4("172.16.1.5"), V4("172.16.2.5"), V4("255.255.255.0")));
  EXPECT_TRUE(rcclMatchSubnetV4(V4("172.16.1.5"), V4("172.16.2.5"), V4("255.255.0.0")));
}

// Interface-selection scenario: scanning candidates must pick exactly the one on
// the remote's subnet (as ncclFindInterfaceMatchSubnet does).
TEST(MatchSubnetTests, IPv4SelectsCorrectInterface) {
  struct in_addr mask = V4("255.255.255.0");
  struct in_addr remote = V4("10.0.5.42");

  // Candidate local interface addresses.
  struct in_addr eth0 = V4("192.168.1.1"); // wrong subnet
  struct in_addr eth1 = V4("10.0.5.1");    // correct subnet
  struct in_addr eth2 = V4("172.16.0.1");  // wrong subnet

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
