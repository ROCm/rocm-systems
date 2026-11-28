// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vmm/vfu/guest_region_map.h"

#include <iterator>
#include <limits>

namespace rocjitsu {

bool GuestRegionMap::insert(const simdojo::DmaRegion &region, std::byte *host_address) {
  const auto [entry, inserted] = regions_.try_emplace(
      region.guest_phys, Window{.region = region, .host_address = host_address});
  if (inserted) {
    return true;
  }
  // The address always follows the latest report, even for a window that is
  // otherwise unchanged, so a lookup can never hand out one the transport has
  // since replaced.
  entry->second.host_address = host_address;
  // A window at a known address but a different length is a different window,
  // so it replaces the stale record rather than being ignored: a later transfer
  // splitting on the old boundary would refuse a range that is now valid.
  if (entry->second.region.length != region.length) {
    entry->second.region = region;
    return true;
  }
  return false;
}

void GuestRegionMap::erase(const simdojo::DmaRegion &region) { regions_.erase(region.guest_phys); }

void GuestRegionMap::clear() { regions_.clear(); }

const GuestRegionMap::Window *GuestRegionMap::containing(uint64_t guest_phys) const {
  const std::map<uint64_t, Window>::const_iterator after = regions_.upper_bound(guest_phys);
  if (after == regions_.begin()) {
    return nullptr;
  }
  const Window &candidate = std::prev(after)->second;
  const simdojo::DmaRegion &region = candidate.region;
  // A window's length reaches this from a vfio-user client, so it is not trusted
  // to be sane: a length that wraps past the end of the address space would make
  // `end` small and the containment test below succeed for an address the window
  // does not hold. A zero-length window contains nothing either.
  if (region.length == 0 ||
      region.length > std::numeric_limits<uint64_t>::max() - region.guest_phys) {
    return nullptr;
  }
  const uint64_t end = region.guest_phys + region.length;
  if (guest_phys < region.guest_phys || guest_phys >= end) {
    return nullptr;
  }
  return &candidate;
}

std::size_t GuestRegionMap::bytes_until_end(uint64_t guest_phys) const {
  const Window *window = containing(guest_phys);
  if (window == nullptr) {
    return 0;
  }
  return static_cast<std::size_t>(window->region.guest_phys + window->region.length - guest_phys);
}

std::byte *GuestRegionMap::host_span(uint64_t guest_phys, std::size_t length,
                                     simdojo::DmaAccess needed) const {
  const Window *window = containing(guest_phys);
  if (window == nullptr || window->host_address == nullptr ||
      !simdojo::contains(window->region.access, needed)) {
    return nullptr;
  }
  const uint64_t offset = guest_phys - window->region.guest_phys;
  if (length > window->region.length - offset) {
    return nullptr;
  }
  return window->host_address + offset;
}

} // namespace rocjitsu
