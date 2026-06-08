// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file partitioning.h
/// @brief Deterministic AMDGPU simulator topology partitioning helpers.

#ifndef ROCJITSU_VM_AMDGPU_PARTITIONING_H_
#define ROCJITSU_VM_AMDGPU_PARTITIONING_H_

#include "rocjitsu/vm/soc.h"

#include "simdojo/sim/topology.h"

#include <cstdint>
#include <span>

namespace rocjitsu {
namespace amdgpu {

/// @brief Partition an AMDGPU topology by whole XCD subtrees.
///
/// @details If @p num_partitions is greater than one and at least one XCD is
/// present, this installs a manual topology partition where each XCD subtree is
/// assigned to global_xcd_index % num_partitions. Components outside XCD
/// subtrees stay in partition 0.
/// @returns true when a manual partition was installed, false for no-op cases.
bool partition_topology_by_xcds(simdojo::Topology &topology, std::span<SoC *> socs,
                                uint32_t num_partitions);

/// @brief Convenience overload for a single SoC.
bool partition_topology_by_xcds(simdojo::Topology &topology, SoC *soc, uint32_t num_partitions);

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_PARTITIONING_H_
