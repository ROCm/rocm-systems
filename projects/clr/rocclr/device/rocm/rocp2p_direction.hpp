/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

namespace roc {
namespace p2p {

// Which buffers the device executing a P2P blit has to reach.
enum class CopyRole {
  kLocalToPeer,  // local read, peer write
  kPeerToLocal,  // peer read, local write
  kLocalToLocal,
  kThirdParty,  // neither buffer is local
};

constexpr CopyRole SelectRole(bool copy_on_src, bool copy_on_dst) {
  if (copy_on_src && !copy_on_dst) {
    return CopyRole::kLocalToPeer;
  }
  if (copy_on_dst && !copy_on_src) {
    return CopyRole::kPeerToLocal;
  }
  if (copy_on_src) {
    return CopyRole::kLocalToLocal;
  }
  return CopyRole::kThirdParty;
}

// A direct P2P blit is possible only when the executing device can address
// every peer buffer it touches. KFD records peer access per direction, so this
// is not symmetric and must not be decided from a single link property.
constexpr bool CanBlit(CopyRole role, bool can_reach_src, bool can_reach_dst) {
  switch (role) {
    case CopyRole::kLocalToPeer:
      return can_reach_dst;
    case CopyRole::kPeerToLocal:
      return can_reach_src;
    case CopyRole::kLocalToLocal:
      return true;
    case CopyRole::kThirdParty:
      return can_reach_src && can_reach_dst;
  }
  return false;
}

constexpr bool NeedsStaging(CopyRole role, bool can_reach_src, bool can_reach_dst) {
  return !CanBlit(role, can_reach_src, can_reach_dst);
}

// Compile-time truth table. Keeping it next to the helper makes the decision
// testable without pulling a private ROCclr header into hip-tests.
static_assert(SelectRole(false, false) == CopyRole::kThirdParty);
static_assert(SelectRole(true, false) == CopyRole::kLocalToPeer);
static_assert(SelectRole(false, true) == CopyRole::kPeerToLocal);
static_assert(SelectRole(true, true) == CopyRole::kLocalToLocal);

// Local read, peer write: only the destination has to be addressable.
static_assert(!CanBlit(CopyRole::kLocalToPeer, false, false));
static_assert(CanBlit(CopyRole::kLocalToPeer, false, true));
static_assert(!CanBlit(CopyRole::kLocalToPeer, true, false));
static_assert(CanBlit(CopyRole::kLocalToPeer, true, true));

// Peer read, local write: only the source has to be addressable.
static_assert(!CanBlit(CopyRole::kPeerToLocal, false, false));
static_assert(!CanBlit(CopyRole::kPeerToLocal, false, true));
static_assert(CanBlit(CopyRole::kPeerToLocal, true, false));
static_assert(CanBlit(CopyRole::kPeerToLocal, true, true));

// Both buffers are local: no peer address is touched.
static_assert(CanBlit(CopyRole::kLocalToLocal, false, false));
static_assert(CanBlit(CopyRole::kLocalToLocal, false, true));
static_assert(CanBlit(CopyRole::kLocalToLocal, true, false));
static_assert(CanBlit(CopyRole::kLocalToLocal, true, true));

// Third device: both peer buffers have to be addressable.
static_assert(!CanBlit(CopyRole::kThirdParty, false, false));
static_assert(!CanBlit(CopyRole::kThirdParty, false, true));
static_assert(!CanBlit(CopyRole::kThirdParty, true, false));
static_assert(CanBlit(CopyRole::kThirdParty, true, true));

}  // namespace p2p
}  // namespace roc
