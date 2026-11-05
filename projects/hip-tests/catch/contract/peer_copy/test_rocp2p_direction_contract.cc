/*
 * Copyright Advanced Micro Devices, Inc.
 *
 * SPDX-License-Identifier: MIT
 */

#include <array>

#include <hip_test_common.hh>
#include "rocp2p_direction.hpp"

namespace {
struct RoleCase {
  bool copy_on_src;
  bool copy_on_dst;
  roc::p2p::CopyRole role;
};

struct DecisionCase {
  roc::p2p::CopyRole role;
  bool can_reach_src;
  bool can_reach_dst;
  bool can_blit;
};

constexpr std::array<RoleCase, 4> kRoleCases{{
    {false, false, roc::p2p::CopyRole::kThirdParty},
    {true, false, roc::p2p::CopyRole::kLocalToPeer},
    {false, true, roc::p2p::CopyRole::kPeerToLocal},
    {true, true, roc::p2p::CopyRole::kLocalToLocal},
}};

constexpr std::array<DecisionCase, 16> kCases{{
    // Local read, peer write: only the destination has to be addressable.
    {roc::p2p::CopyRole::kLocalToPeer, false, false, false},
    {roc::p2p::CopyRole::kLocalToPeer, false, true, true},
    {roc::p2p::CopyRole::kLocalToPeer, true, false, false},
    {roc::p2p::CopyRole::kLocalToPeer, true, true, true},
    // Peer read, local write: only the source has to be addressable.
    {roc::p2p::CopyRole::kPeerToLocal, false, false, false},
    {roc::p2p::CopyRole::kPeerToLocal, false, true, false},
    {roc::p2p::CopyRole::kPeerToLocal, true, false, true},
    {roc::p2p::CopyRole::kPeerToLocal, true, true, true},
    // Both buffers are local: no peer address is touched.
    {roc::p2p::CopyRole::kLocalToLocal, false, false, true},
    {roc::p2p::CopyRole::kLocalToLocal, false, true, true},
    {roc::p2p::CopyRole::kLocalToLocal, true, false, true},
    {roc::p2p::CopyRole::kLocalToLocal, true, true, true},
    // Third device: both peer buffers have to be addressable.
    {roc::p2p::CopyRole::kThirdParty, false, false, false},
    {roc::p2p::CopyRole::kThirdParty, false, true, false},
    {roc::p2p::CopyRole::kThirdParty, true, false, false},
    {roc::p2p::CopyRole::kThirdParty, true, true, true},
}};
}  // namespace

HIP_TEST_CASE(Contract_P2P_Direction_BlitDecision) {
  for (const auto& c : kRoleCases) {
    INFO("copy_on_src=" << c.copy_on_src << " copy_on_dst=" << c.copy_on_dst);
    REQUIRE(roc::p2p::SelectRole(c.copy_on_src, c.copy_on_dst) == c.role);
  }

  for (const auto& c : kCases) {
    INFO("role=" << static_cast<int>(c.role) << " src=" << c.can_reach_src
                 << " dst=" << c.can_reach_dst);
    REQUIRE(roc::p2p::CanBlit(c.role, c.can_reach_src, c.can_reach_dst) == c.can_blit);
    REQUIRE(roc::p2p::NeedsStaging(c.role, c.can_reach_src, c.can_reach_dst) == !c.can_blit);
  }
}
