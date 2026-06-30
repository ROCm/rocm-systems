/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdint>

namespace amd {

//! Range-aware erase from an address-keyed ordered map.
//!
//! The map is keyed by an allocation's base address and each stored value spans
//! [base, base + size). Lookups (MemObjMap::FindMemObj) resolve *any* pointer
//! that falls inside that interval, so removal must apply the same range test.
//! Erasing by the exact key alone is wrong: a pointer that a lookup happily
//! resolves -- an interior address, or a base that was adjusted for a
//! per-device / host mapping -- would not match its own entry, and the caller
//! would wrongly conclude the allocation is missing. Keeping find and remove
//! symmetric is what lets MemObjMap::RemoveMemObj treat a miss as a genuine
//! corruption signal again instead of a routine false alarm.
//!
//! Returns the removed mapped value (a pointer) on success, or a
//! default-constructed value (nullptr) when no entry covers \a key. \a size_of
//! maps a stored value to its byte span. Isolating this here keeps the range
//! logic -- duplicated across the FindMemObj* helpers -- in one unit-tested
//! place.
template <typename Map, typename SizeFn>
inline typename Map::mapped_type EraseCoveringMemObj(Map& map, uintptr_t key, SizeFn size_of) {
  // upper_bound(key) is the first entry strictly above key; the entry that may
  // cover key is its immediate predecessor.
  auto it = map.upper_bound(key);
  if (it == map.begin()) {
    return nullptr;
  }
  --it;
  const uintptr_t base = it->first;
  if (key >= base && key < base + size_of(it->second)) {
    typename Map::mapped_type value = it->second;
    map.erase(it);
    return value;
  }
  return nullptr;
}

}  // namespace amd
