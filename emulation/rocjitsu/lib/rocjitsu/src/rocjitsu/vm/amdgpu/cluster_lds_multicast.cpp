// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/cluster_lds_multicast.h"

#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/lds.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <format>
#include <stdexcept>
#include <utility>

namespace rocjitsu {
namespace amdgpu {

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
make_cluster_lds_multicast_transaction(const VectorMemState &state, const Wavefront &wf,
                                       std::vector<ClusterLdsTarget> targets) {
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
  txn.per_lane_lds_addr = state.per_lane_lds_addr;
  txn.payload = state.response_data;
  txn.targets = std::move(targets);
  return txn;
}

ClusterLdsMulticastResult
ImmediateClusterLdsMulticastEngine::submit(ClusterLdsMulticastTransaction txn,
                                           ClusterLdsMulticastCompletion /*complete*/) {
  for (const auto &target : txn.targets) {
    if (!target.cu)
      continue;
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
      for (uint32_t b = 0; b < txn.bytes_per_lane; ++b)
        lds.write8(lds_addr + b, txn.payload[data_offset + b]);
    }
  }
  return ClusterLdsMulticastResult::Complete;
}

} // namespace amdgpu
} // namespace rocjitsu
