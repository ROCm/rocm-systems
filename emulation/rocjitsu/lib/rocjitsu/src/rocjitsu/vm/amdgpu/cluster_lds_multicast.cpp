// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/cluster_lds_multicast.h"

#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/lds.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <format>
#include <stdexcept>
#include <utility>

namespace rocjitsu {
namespace amdgpu {

namespace {

void write_cluster_lds_target(const ClusterLdsMulticastTransaction &txn,
                              const ClusterLdsTarget &target) {
  if (!target.cu)
    return;

  auto &lds = target.cu->lds();
  for (uint32_t lane = 0; lane < txn.wf_size; ++lane) {
    if ((txn.lane_mask & (1ULL << lane)) == 0)
      continue;
    uint32_t data_offset = lane * txn.bytes_per_lane;
    if (data_offset + txn.bytes_per_lane > txn.payload.size()) {
      throw std::runtime_error(std::format(
          "cluster LDS multicast payload too small: lane={} offset={} bytes={} payload={}", lane,
          data_offset, txn.bytes_per_lane, txn.payload.size()));
    }
    uint32_t lds_addr = cluster_lds_lane_addr(txn, lane, target.lds_base);
    if (uint64_t(lds_addr) + txn.bytes_per_lane > lds.size_bytes()) {
      throw std::runtime_error(std::format(
          "cluster LDS multicast target address out of range: lane={} target_wg={} addr={:#x} "
          "bytes={} lds_size={}",
          lane, target.wg_id, lds_addr, txn.bytes_per_lane, lds.size_bytes()));
    }
    for (uint32_t b = 0; b < txn.bytes_per_lane; ++b)
      lds.write8(lds_addr + b, txn.payload[data_offset + b]);
  }
}

} // namespace

uint32_t remap_cluster_lds_addr(uint32_t source_lds_base, uint32_t target_lds_base,
                                uint32_t source_lds_addr) {
  if (source_lds_addr < source_lds_base) {
    throw std::runtime_error(std::format("cluster LDS source address {:#x} precedes base {:#x}",
                                         source_lds_addr, source_lds_base));
  }
  return target_lds_base + (source_lds_addr - source_lds_base);
}

uint32_t cluster_lds_lane_addr(const ClusterLdsMulticastTransaction &txn, uint32_t lane,
                               uint32_t target_lds_base) {
  uint32_t source_lds_addr = txn.source_lds_base + lane * txn.bytes_per_lane;
  if (txn.per_lane_addr)
    source_lds_addr = txn.per_lane_lds_addr[lane];
  return remap_cluster_lds_addr(txn.source_lds_base, target_lds_base, source_lds_addr);
}

ClusterLdsMulticastTransaction
make_cluster_lds_multicast_transaction(VectorMemState &state, const Wavefront &wf,
                                       std::vector<ClusterLdsTarget> targets) {
  assert(state.cluster_multicast && "cluster LDS multicast transaction requires cluster_multicast");
  assert(state.lds_base == wf.lds_base() &&
         "cluster LDS multicast source base must be the source WG allocation base");
  ClusterLdsMulticastTransaction txn{};
  txn.dispatch_id = wf.dispatch_id();
  txn.source_wg_id = wf.wg_id();
  txn.source_cluster_rank = wf.cluster_rank();
  txn.source_lds_base = state.lds_base;
  txn.mcast_mask = state.cluster_mcast_mask;
  txn.wait_counter_type = state.wait_counter_type;
  txn.bytes_per_lane = state.num_elems * state.elem_size;
  txn.wf_size = state.wf_size;
  txn.lane_mask = state.lane_mask;
  txn.per_lane_addr = state.lds_per_lane_addr;
  txn.per_lane_global_addr = state.per_lane_addr;
  txn.per_lane_lds_addr = state.per_lane_lds_addr;
  txn.payload = std::move(state.response_data);
  txn.targets = std::move(targets);
  return txn;
}

bool cluster_lds_source_rank_selected(const ClusterLdsMulticastTransaction &txn) {
  return txn.mcast_mask == 0 ||
         (txn.mcast_mask & cluster_multicast_rank_mask(txn.source_cluster_rank)) != 0;
}

ClusterLdsMulticastResult
ImmediateClusterLdsMulticastEngine::submit(ClusterLdsMulticastTransaction txn,
                                           ClusterLdsMulticastCompletion /*complete*/) {
  if (!cluster_lds_source_rank_selected(txn))
    return ClusterLdsMulticastResult::Complete;

  auto target_it = std::find_if(txn.targets.begin(), txn.targets.end(), [&](const auto &target) {
    return target.wg_id == txn.source_wg_id && target.cluster_rank == txn.source_cluster_rank;
  });

  if (target_it != txn.targets.end())
    write_cluster_lds_target(txn, *target_it);
  return ClusterLdsMulticastResult::Complete;
}

} // namespace amdgpu
} // namespace rocjitsu
