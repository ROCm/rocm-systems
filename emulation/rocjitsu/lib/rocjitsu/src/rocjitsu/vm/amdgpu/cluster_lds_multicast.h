// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_CLUSTER_LDS_MULTICAST_H_
#define ROCJITSU_VM_AMDGPU_CLUSTER_LDS_MULTICAST_H_

#include "rocjitsu/vm/amdgpu/wait_counters.h"

#include <array>
#include <cstdint>
#include <functional>
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
  /// Informational placement rank for diagnostics and future rank-sensitive
  /// cluster operations; current multicast writes need only CU and LDS base.
  uint32_t cluster_rank = 0;
};

/// @brief Fully described LDS multicast writeback produced by an async cluster load.
///
/// @details The immediate functional backend consumes this synchronously today.
/// A future timing backend can instead retain the transaction, model delivery
/// latency/contention, and complete ASYNCCNT only after all target LDS writes
/// have been delivered. The transaction is the boundary where a future fabric
/// model can preserve source/target rank, LDS windows, payload, and lane mask
/// before scheduling per-target delivery events.
struct ClusterLdsMulticastTransaction {
  uint32_t dispatch_id = 0;
  uint32_t source_wg_id = 0;
  uint32_t source_cluster_rank = 0;
  uint32_t source_lds_base = 0;
  uint32_t mcast_mask = 0;
  WaitCounterType wait_counter_type = WaitCounterType::ASYNCCNT;
  uint32_t bytes_per_lane = 0;
  uint32_t wf_size = 0;
  uint64_t lane_mask = 0;
  bool per_lane_addr = false;
  std::array<uint32_t, 64> per_lane_lds_addr = {};
  std::vector<uint8_t> payload;
  std::vector<ClusterLdsTarget> targets;
};

enum class [[nodiscard]] ClusterLdsMulticastResult {
  Complete,
  Deferred,
};

using ClusterLdsMulticastCompletion = std::function<void()>;

/// @brief Remap a source WG LDS address into an equivalent target WG LDS window.
uint32_t remap_cluster_lds_addr(uint32_t source_lds_base, uint32_t target_lds_base,
                                uint32_t source_lds_addr);

/// @brief Return the target LDS address for a lane in one multicast target.
uint32_t cluster_lds_lane_addr(const ClusterLdsMulticastTransaction &txn, uint32_t lane,
                               uint32_t target_lds_base);

ClusterLdsMulticastTransaction
make_cluster_lds_multicast_transaction(VectorMemState &state, const Wavefront &wf,
                                       std::vector<ClusterLdsTarget> targets);

/// @brief Execution boundary for cluster-load async-to-LDS fan-out.
class ClusterLdsMulticastEngine {
public:
  virtual ~ClusterLdsMulticastEngine() = default;

  /// @brief Submit one owned multicast transaction.
  ///
  /// @details Backends returning Complete must perform all writes before return
  /// and must not invoke complete. Backends returning Deferred take ownership of
  /// txn and must invoke complete exactly once after all writes are visible.
  virtual ClusterLdsMulticastResult submit(ClusterLdsMulticastTransaction txn,
                                           ClusterLdsMulticastCompletion complete) = 0;
};

/// @brief Functional backend: immediately writes the multicast payload to target LDS.
class ImmediateClusterLdsMulticastEngine final : public ClusterLdsMulticastEngine {
public:
  ClusterLdsMulticastResult submit(ClusterLdsMulticastTransaction txn,
                                   ClusterLdsMulticastCompletion complete) override;
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_CLUSTER_LDS_MULTICAST_H_
