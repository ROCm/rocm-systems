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

}  // namespace p2p
}  // namespace roc
