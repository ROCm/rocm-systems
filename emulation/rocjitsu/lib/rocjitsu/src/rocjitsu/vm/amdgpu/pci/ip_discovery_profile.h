// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file ip_discovery_profile.h
/// @brief The blocks a simulated gfx1250 reports to a guest driver.
///
/// @details One place decides what the device says it contains, because two
/// places would eventually disagree: the device publishes this table into its
/// own memory, and the offline tool writes the same bytes to a file for a guest
/// booted with `amdgpu.discovery=2`. A guest that sees different hardware
/// depending on which path delivered the table would be worse than either.

#pragma once

#include "rocjitsu/vm/amdgpu/pci/ip_discovery.h"

#include <cstdint>

namespace rocjitsu {

/// @brief How much of the part the discovery table advertises.
///
/// @details Instance counts in the table are not decoration: the driver derives
/// its XCC mask from the number of graphics instances, and on this family it
/// then derives the SDMA mask from that (`soc_v1_0_init_soc_config` recomputes
/// `sdma_mask` as two engines per XCC, discarding whatever the SDMA records
/// said). So the graphics instance count is the one knob that decides the shape
/// of the machine the guest believes it has.
struct GpuDiscoveryTopology {
  /// @brief Graphics instances, which become the guest's XCC mask.
  ///
  /// @details Deliberately one during bring-up, which is **narrower than the
  /// eight XCDs the part's own config describes**. Advertising all eight would
  /// give the guest sixteen SDMA engines and two AIDs, pulling in XCP partition
  /// management and exhausting the MMHUB invalidation-engine budget, none of
  /// which the device can answer yet. Raise this when the blocks that consume
  /// it are modelled, and not before.
  uint8_t graphics_instances = 1;
};

/// @brief The blocks a simulated gfx1250 reports.
///
/// @details A short list on purpose. Naming a block is what makes the driver
/// instantiate a driver for it, so leaving one out is how the device says it
/// does not have that hardware yet.
///
/// @param[in] topology How much of the part to advertise.
/// @returns The spec to serialize with @ref build_ip_discovery_table.
[[nodiscard]] IpDiscoverySpec gfx1250_discovery_spec(const GpuDiscoveryTopology &topology = {});

} // namespace rocjitsu
