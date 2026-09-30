// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vmm/vfu/guest_region_map.h"

#include <limits>

namespace rocjitsu {

bool GuestRegionMap::insert(const simdojo::DmaRegion &region) {
  const auto [entry, inserted] = regions_.try_emplace(region.guest_phys, region);
  if (inserted) {
    return true;
  }
  // A window at a known address but a different length is a different window,
  // so it replaces the stale record rather than being ignored: a later transfer
  // splitting on the old boundary would refuse a range that is now valid.
  if (entry->second.length != region.length) {
    entry->second = region;
    return true;
  }
  return false;
}

void GuestRegionMap::erase(const simdojo::DmaRegion &region) { regions_.erase(region.guest_phys); }

void GuestRegionMap::clear() { regions_.clear(); }

std::size_t GuestRegionMap::bytes_until_end(uint64_t guest_phys) const {
  const auto after = regions_.upper_bound(guest_phys);
  if (after == regions_.begin()) {
    return 0;
  }
  const simdojo::DmaRegion &candidate = std::prev(after)->second;
  // A window's length reaches this from a vfio-user client, so it is not trusted
  // to be sane: a length that wraps past the end of the address space would make
  // `end` small and the containment test below succeed for an address the window
  // does not hold. A zero-length window contains nothing either.
  if (candidate.length == 0 ||
      candidate.length > std::numeric_limits<uint64_t>::max() - candidate.guest_phys) {
    return 0;
  }
  const uint64_t end = candidate.guest_phys + candidate.length;
  if (guest_phys < candidate.guest_phys || guest_phys >= end) {
    return 0;
  }
  return static_cast<std::size_t>(end - guest_phys);
}

} // namespace rocjitsu
