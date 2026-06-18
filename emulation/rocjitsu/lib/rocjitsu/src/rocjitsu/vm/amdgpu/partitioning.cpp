// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/partitioning.h"

#include <array>
#include <unordered_map>

namespace rocjitsu {
namespace amdgpu {

bool partition_topology_by_xcds(simdojo::Topology &topology, std::span<SoC *> socs,
                                uint32_t num_partitions) {
  if (num_partitions <= 1)
    return false;

  std::unordered_map<simdojo::Component *, simdojo::PartitionID> xcd_partitions;
  uint32_t global_xcd_index = 0;
  for (auto *soc : socs) {
    if (!soc)
      continue;
    for (uint32_t xcd_index = 0; xcd_index < soc->num_xcds(); ++xcd_index) {
      xcd_partitions[soc->xcd(xcd_index)] = global_xcd_index % num_partitions;
      ++global_xcd_index;
    }
  }

  if (xcd_partitions.empty())
    return false;

  topology.partition_manual(num_partitions, [&](simdojo::Component *component) {
    for (auto *candidate = component; candidate != nullptr;
         candidate = static_cast<simdojo::Component *>(candidate->parent())) {
      auto it = xcd_partitions.find(candidate);
      if (it != xcd_partitions.end())
        return it->second;
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
