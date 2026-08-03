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

/// @brief Clamp a requested partition count to the visible XCD count.
///
/// @details Counts XCDs across all non-null SoCs and clamps
/// @p requested_partitions to the inclusive range [1, max(total XCDs, 1)].
/// @returns The usable partition count.
[[nodiscard]] uint32_t clamp_xcd_partition_count(std::span<SoC *> socs,
                                                 uint32_t requested_partitions);

/// @brief Convenience overload for a single SoC.
[[nodiscard]] uint32_t clamp_xcd_partition_count(SoC *soc, uint32_t requested_partitions);

/// @brief Partition an AMDGPU topology by whole XCD subtrees.
///
/// @details If @p num_partitions is nonzero and at least one XCD is present,
/// this installs a manual topology partition where each XCD subtree is
/// assigned to global_xcd_index % num_partitions. Components outside XCD
/// subtrees stay in partition 0.
/// @returns true when a manual partition was installed; false when
/// @p num_partitions is zero, no XCDs are present, or any supplied XCD is not
/// a member of @p topology. Failure leaves existing partition state unchanged.
[[nodiscard]] bool partition_topology_by_xcds(simdojo::Topology &topology, std::span<SoC *> socs,
                                              uint32_t num_partitions);

/// @brief Convenience overload for a single SoC.
[[nodiscard]] bool partition_topology_by_xcds(simdojo::Topology &topology, SoC *soc,
                                              uint32_t num_partitions);

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_PARTITIONING_H_
