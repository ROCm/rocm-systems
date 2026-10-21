// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file guest_region_map.h
/// @brief The guest memory windows a client has shared, ordered by address.
///
/// @details A transport has to know where one shared window ends, because a
/// transfer crossing two of them must be split at the boundary and the
/// scatter-gather entries the protocol hands back do not say where that is. The
/// windows arrive one map notification at a time, so the transport keeps its own
/// record of them.
///
/// Order matters for cost, not just tidiness. A guest whose IOMMU reflects
/// memory page by page can share hundreds of thousands of windows, so lookup and
/// insertion have to stay logarithmic; scanning a list per window would make
/// registering N windows quadratic and copying across them worse.

#pragma once

#include "simdojo/components/pci_device.h"

#include <cstddef>
#include <cstdint>
#include <map>

namespace rocjitsu {

/// @brief Address-ordered set of the guest memory windows currently shared.
class GuestRegionMap {
public:
  /// @brief Record a window the client shared.
  /// @param[in] region The window.
  /// @retval true The window was not already recorded.
  /// @retval false An identical window was already present, so nothing changed.
  /// @details A client may re-share a window it already holds, which the
  /// protocol reports as a fresh mapping; treating that as new would drift this
  /// record and the device's view of it apart.
  bool insert(const simdojo::DmaRegion &region);

  /// @brief Forget a window the client withdrew.
  /// @param[in] region The window being withdrawn.
  void erase(const simdojo::DmaRegion &region);

  /// @brief Forget every window, as when a client disconnects.
  void clear();

  /// @brief Return how far @p guest_phys is from the end of its window.
  /// @param[in] guest_phys Guest-physical address to locate.
  /// @returns Bytes from @p guest_phys to the end of the window containing it,
  ///          or zero if no shared window contains it.
  [[nodiscard]] std::size_t bytes_until_end(uint64_t guest_phys) const;

  /// @brief Return how many windows are recorded.
  [[nodiscard]] std::size_t size() const { return regions_.size(); }

private:
  std::map<uint64_t, simdojo::DmaRegion> regions_;
};

} // namespace rocjitsu
