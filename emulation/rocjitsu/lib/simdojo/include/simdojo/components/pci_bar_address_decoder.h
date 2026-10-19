// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "simdojo/components/pci_device.h"

#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <utility>

namespace simdojo {

/// Decode an address in one PCI bus address space to an enabled endpoint BAR.
///
/// The bus adapter publishes windows after ordinary PCI configuration writes.
/// Callers provide a translated bus address, so an upstream IOMMU remains a
/// separate block. A request that intersects more than one window or crosses a
/// boundary has no owner; guessing one would grant access to the wrong device.
/// External synchronization protects publication and resolution together.
class PciBarAddressDecoder {
public:
  struct Window {
    uint32_t function = 0;
    uint32_t bar = 0;
    uint64_t base = 0;
    uint64_t size = 0;
    DmaAccess rights = DmaAccess::None;
    uint64_t generation = 0;

    friend bool operator==(const Window &, const Window &) = default;
  };

  struct Match {
    uint32_t function = 0;
    uint32_t bar = 0;
    uint64_t offset = 0;
    uint64_t generation = 0;
  };

  [[nodiscard]] bool publish(Window window) {
    if (window.base == 0 || window.size == 0 || window.generation == 0 ||
        window.rights == DmaAccess::None ||
        window.base > std::numeric_limits<uint64_t>::max() - window.size)
      return false;
    windows_.insert_or_assign({window.function, window.bar}, window);
    return true;
  }

  void revoke(uint32_t function, uint32_t bar) { windows_.erase({function, bar}); }

  [[nodiscard]] std::optional<Window> window(uint32_t function, uint32_t bar) const {
    const auto found = windows_.find({function, bar});
    if (found == windows_.end())
      return std::nullopt;
    return found->second;
  }

  [[nodiscard]] std::optional<Match> resolve(uint32_t requester, uint64_t address, uint64_t length,
                                             DmaAccess wanted) const {
    if (length == 0 || wanted == DmaAccess::None ||
        address > std::numeric_limits<uint64_t>::max() - length)
      return std::nullopt;

    const uint64_t end = address + length;
    std::optional<Match> match;
    for (const auto &entry : windows_) {
      const Window &window = entry.second;
      const uint64_t window_end = window.base + window.size;
      if (address >= window_end || end <= window.base)
        continue;
      if (match || address < window.base || end > window_end || window.function == requester ||
          !contains(window.rights, wanted))
        return std::nullopt;
      match = Match{.function = window.function,
                    .bar = window.bar,
                    .offset = address - window.base,
                    .generation = window.generation};
    }
    return match;
  }

private:
  std::map<std::pair<uint32_t, uint32_t>, Window> windows_;
};

} // namespace simdojo
