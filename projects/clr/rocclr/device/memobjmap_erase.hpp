/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdint>

namespace amd {

//! Range-aware, predicate-guarded erase from an address-keyed ordered map.
//!
//! The map is keyed by an allocation's base address and each stored value spans
//! [base, base + size). Lookups (MemObjMap::FindMemObj) resolve *any* pointer
//! that falls inside that interval, so a removal that wants to stay symmetric
//! with lookup must apply the same range test rather than erasing by the exact
//! key alone.
//!
//! The entry is erased only if \a pred accepts its value. Range coverage alone
//! does not prove the entry is the one the caller means to remove: on Windows,
//! per-device VA ranges can numerically overlap an unrelated allocation's
//! [base, base + size) in the global map, and erasing whatever happens to
//! cover the pointer would de-index a live allocation. Callers that hold the
//! amd::Memory* being freed pass an identity predicate; callers that only have
//! the pointer accept any covering entry via EraseCoveringMemObj.
//!
//! Returns the removed mapped value (a pointer) on success, or a
//! default-constructed value (nullptr) when no entry covers \a key or \a pred
//! rejects the covering entry (nothing is erased). \a size_of maps a stored
//! value to its byte span. Isolating this here keeps the range logic --
//! duplicated across the FindMemObj* helpers -- in one unit-tested place.
template <typename Map, typename SizeFn, typename Pred>
inline typename Map::mapped_type EraseCoveringMemObjIf(Map& map, uintptr_t key, SizeFn size_of,
                                                       Pred pred) {
  // upper_bound(key) is the first entry strictly above key; the entry that may
  // cover key is its immediate predecessor.
  auto it = map.upper_bound(key);
  if (it == map.begin()) {
    return nullptr;
  }
  --it;
  const uintptr_t base = it->first;
  if (key < base || key >= base + size_of(it->second) || !pred(it->second)) {
    return nullptr;
  }
  typename Map::mapped_type value = it->second;
  map.erase(it);
  return value;
}

//! EraseCoveringMemObjIf accepting any covering entry.
template <typename Map, typename SizeFn>
inline typename Map::mapped_type EraseCoveringMemObj(Map& map, uintptr_t key, SizeFn size_of) {
  return EraseCoveringMemObjIf(map, key, size_of,
                               [](const typename Map::mapped_type&) { return true; });
}

}  // namespace amd
