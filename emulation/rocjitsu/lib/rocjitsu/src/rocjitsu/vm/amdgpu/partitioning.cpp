// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/partitioning.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rocjitsu {
namespace amdgpu {

namespace {

uint32_t total_xcds(std::span<SoC *> socs) {
  uint32_t num_xcds = 0;
  for (SoC *soc : socs) {
    if (soc)
      num_xcds += soc->num_xcds();
  }
  return num_xcds;
}

} // namespace

uint32_t default_xcd_partition_count(std::span<SoC *> socs) {
  // hardware_concurrency() returns 0 when the host thread count is
  // indeterminate; fall back to a single partition rather than guessing.
  const uint32_t host_threads = std::thread::hardware_concurrency();
  if (host_threads == 0)
    return 1;
  return std::max(std::min(host_threads, total_xcds(socs)), 1u);
}

uint32_t default_xcd_partition_count(SoC *soc) {
  std::array<SoC *, 1> socs = {soc};
  return default_xcd_partition_count(std::span<SoC *>(socs));
}

uint32_t clamp_xcd_partition_count(std::span<SoC *> socs, uint32_t requested_partitions) {
  return std::clamp(requested_partitions, 1u, std::max(total_xcds(socs), 1u));
}

uint32_t clamp_xcd_partition_count(SoC *soc, uint32_t requested_partitions) {
  std::array<SoC *, 1> socs = {soc};
  return clamp_xcd_partition_count(std::span<SoC *>(socs), requested_partitions);
}

bool partition_topology_by_xcds(simdojo::Topology &topology, std::span<SoC *> socs,
                                uint32_t num_partitions) {
  if (num_partitions == 0)
    return false;

  const std::vector<simdojo::Component *> components = topology.collect_all_components();
  const std::unordered_set<simdojo::Component *> topology_components(components.begin(),
                                                                     components.end());
  std::unordered_map<simdojo::Component *, simdojo::PartitionID> xcd_partitions;
  uint32_t global_xcd_index = 0;
  for (SoC *soc : socs) {
    if (!soc)
      continue;
    for (uint32_t xcd_index = 0; xcd_index < soc->num_xcds(); ++xcd_index) {
      simdojo::Component *xcd = soc->xcd(xcd_index);
      if (!topology_components.contains(xcd))
        return false;
      xcd_partitions[xcd] = global_xcd_index % num_partitions;
      ++global_xcd_index;
    }
  }

  if (xcd_partitions.empty())
    return false;

  topology.partition_manual(num_partitions, [&](simdojo::Component *component) {
    for (auto *candidate = component; candidate != nullptr;) {
      auto it = xcd_partitions.find(candidate);
      if (it != xcd_partitions.end())
        return it->second;

      auto *parent = candidate->parent();
      candidate = dynamic_cast<simdojo::Component *>(parent);
      assert((parent == nullptr || candidate != nullptr) &&
             "component parents must also be components");
    }
    return simdojo::PartitionID{0};
  });
  return true;
}

bool partition_topology_by_xcds(simdojo::Topology &topology, SoC *soc, uint32_t num_partitions) {
  std::array<SoC *, 1> socs = {soc};
  return partition_topology_by_xcds(topology, std::span<SoC *>(socs), num_partitions);
}

} // namespace amdgpu
} // namespace rocjitsu
