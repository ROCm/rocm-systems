// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_CLUSTER_LDS_MULTICAST_H_
#define ROCJITSU_VM_AMDGPU_CLUSTER_LDS_MULTICAST_H_

#include <array>
#include <cstdint>
#include <vector>

namespace rocjitsu {
namespace amdgpu {

class ComputeUnitCore;
struct VectorMemState;
class Wavefront;

struct ClusterLdsTarget {
  ComputeUnitCore *cu = nullptr;
  uint32_t wg_id = 0;
  uint32_t lds_base = 0;
  /// Placement rank used to identify the issuing workgroup's destination.
  uint32_t cluster_rank = 0;
};

/// @brief LDS writeback request produced by an async cluster load.
///
/// @details Each transaction represents one workgroup's participation in a
/// cluster async-to-LDS operation. The mask still records the selected cluster
/// ranks, but functional writeback uses the issuing workgroup's own LDS
/// destination metadata; peers only receive data when they issue their own
/// matching transaction.
struct ClusterLdsMulticastTransaction {
  uint32_t source_wg_id = 0;
  uint32_t source_cluster_rank = 0;
  uint32_t source_lds_base = 0;
  uint32_t mcast_mask = 0;
  uint32_t bytes_per_lane = 0;
  uint32_t wf_size = 0;
  uint64_t lane_mask = 0;
  bool per_lane_addr = false;
  std::array<uint32_t, 64> per_lane_lds_addr = {};
  std::vector<uint8_t> payload;
  std::vector<ClusterLdsTarget> targets;
};

/// @brief Remap a source WG LDS address into an equivalent target WG LDS window.
uint64_t remap_cluster_lds_addr(uint32_t source_lds_base, uint32_t target_lds_base,
                                uint64_t source_lds_addr);

/// @brief Return the lane LDS address remapped into one target LDS window.
uint64_t cluster_lds_lane_addr(const ClusterLdsMulticastTransaction &txn, uint32_t lane,
                               uint32_t target_lds_base);

/// @brief Return true when the transaction mask selects the issuing workgroup.
bool cluster_lds_source_rank_selected(const ClusterLdsMulticastTransaction &txn);

ClusterLdsMulticastTransaction
make_cluster_lds_multicast_transaction(VectorMemState &state, const Wavefront &wf,
                                       std::vector<ClusterLdsTarget> targets);

/// @brief Write the issuing participant's own LDS payload.
/// @details This deliberately does not fan out one requester payload into peer
/// LDS windows. Peers selected by M0 are eligible participants, but each peer's
/// LDS destination is taken from that peer's own issued transaction.
void write_cluster_lds_multicast(const ClusterLdsMulticastTransaction &txn);

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_CLUSTER_LDS_MULTICAST_H_
