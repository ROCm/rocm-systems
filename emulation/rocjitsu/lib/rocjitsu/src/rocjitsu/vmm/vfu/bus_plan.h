// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file bus_plan.h
/// @brief Turns what a device declares into the arguments a transport registers.
///
/// @details The neutral @ref simdojo::BarSpec and the vfio-user region API do
/// not agree on how "the guest may not map this" is spelled. libvfio-user treats
/// a region that has a backing descriptor but no explicit windows as fully
/// mappable, whereas a @ref simdojo::BarSpec with no windows means every access
/// must trap so the device sees it. Reconciling the two by hand at the call site
/// is how a register aperture silently becomes shared memory, so the policy
/// lives here where it can be tested on its own.

#pragma once

#include "simdojo/components/pci_device.h"

#include <cstdint>
#include <vector>

namespace rocjitsu {

/// @brief How one BAR should be registered with a transport.
struct BarRegionPlan {
  bool valid = false;     ///< False if the declaration is self-inconsistent.
  int backing_fd = -1;    ///< Descriptor to share, or negative to trap every access.
  uint64_t fd_offset = 0; ///< Offset into @ref backing_fd of the region base.
  std::vector<simdojo::MmapArea> mmap_areas; ///< Windows the guest may map; may be empty.
};

/// @brief Decide how @p bar should be registered.
/// @param[in] bar The BAR as the device declared it.
/// @returns The plan; @ref BarRegionPlan::valid is false when @p bar is
///          inconsistent, such as windows that fall outside the region or
///          windows with no descriptor to back them.
[[nodiscard]] BarRegionPlan plan_bar_region(const simdojo::BarSpec &bar);

/// @brief How a device's interrupt declaration should be registered.
struct InterruptPlan {
  bool supported = false;  ///< False if the transport cannot advertise this kind yet.
  uint32_t intx_count = 0; ///< Legacy interrupt pins to advertise; zero for none.
};

/// @brief Decide how @p spec should be advertised.
/// @param[in] spec The interrupt capability the device declared.
/// @returns The plan; @ref InterruptPlan::supported is false for capabilities
///          this transport does not implement, which the caller must refuse
///          rather than quietly substitute something else for.
/// @details A device that raises nothing is advertised with nothing: telling a
/// guest about a pin the device never asserts invites a driver to wait on it.
[[nodiscard]] InterruptPlan plan_interrupts(const simdojo::InterruptSpec &spec);

} // namespace rocjitsu
