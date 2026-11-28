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
///
/// A window the transport has mapped into this process is recorded with the
/// address it is mapped at, so a transfer lying inside one window can be
/// resolved to host memory from this record alone.

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
  /// @param[in] host_address Where this process can address the window's first
  ///            byte, with all @c region.length bytes addressable from there,
  ///            or nullptr when the window is not mapped here.
  /// @retval true The window was not already recorded.
  /// @retval false A window with the same address and length was already
  ///         present; only its host address was replaced by @p host_address.
  /// @details A client may re-share a window it already holds, which the
  /// protocol reports as a fresh mapping; treating that as new would drift this
  /// record and the device's view of it apart.
  bool insert(const simdojo::DmaRegion &region, std::byte *host_address = nullptr);

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

  /// @brief Locate a span that lies wholly inside one window mapped here.
  /// @param[in] guest_phys Guest-physical address of the span's first byte.
  /// @param[in] length Number of bytes in the span.
  /// @param[in] needed Access the window must grant.
  /// @returns Where the span starts in this process, or nullptr unless a single
  ///          recorded window contains @p guest_phys and the @p length bytes
  ///          from it, was recorded with a host address, and grants every
  ///          access in @p needed.
  /// @details The address is only as durable as the mapping behind it, so the
  /// caller must keep the window from being erased while it uses the result.
  [[nodiscard]] std::byte *host_span(uint64_t guest_phys, std::size_t length,
                                     simdojo::DmaAccess needed) const;

  /// @brief Return how many windows are recorded.
  [[nodiscard]] std::size_t size() const { return regions_.size(); }

private:
  /// @brief One shared window, and where this process can address it.
  struct Window {
    simdojo::DmaRegion region;
    std::byte *host_address = nullptr;
  };

  /// @brief Return the well-formed window containing @p guest_phys, or nullptr.
  [[nodiscard]] const Window *containing(uint64_t guest_phys) const;

  std::map<uint64_t, Window> regions_;
};

} // namespace rocjitsu
