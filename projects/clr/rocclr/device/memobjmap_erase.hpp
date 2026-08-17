/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace amd {

//! Range-aware erase from an address-keyed ordered map.
//!
//! The map is keyed by an allocation's base address and each stored value spans
//! [base, base + size). Lookups (MemObjMap::FindMemObj) resolve *any* pointer
//! that falls inside that interval, so a removal that wants to stay symmetric
//! with lookup (MemObjMap::FindAndRemoveMemObj) must apply the same range test
//! rather than erasing by the exact key alone.
//!
//! Returns the removed mapped value (a pointer) on success, or a
//! default-constructed value (nullptr) when no entry covers \a key (nothing is
//! erased). \a size_of maps a stored value to its byte span. Isolating this
//! here keeps the range logic -- duplicated across the FindMemObj* helpers --
//! in one unit-tested place.
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
  if (key < base || key >= base + size_of(it->second)) {
    return nullptr;
  }
  typename Map::mapped_type value = it->second;
  map.erase(it);
  return value;
}

//! Erase every entry whose mapped value is \a value; returns the number erased.
//!
//! An allocation can be indexed under several keys at once (its base address,
//! its host pointer, per-device virtual addresses), and a free must drop every
//! one of them before the object is released -- any entry left behind would
//! dangle. Erasing by identity rather than by key needs no knowledge of which
//! aliases exist and can never touch another allocation's entry.
template <typename Map, typename Value>
inline size_t EraseEntriesWithValue(Map& map, const Value& value) {
  size_t erased = 0;
  for (auto it = map.begin(); it != map.end();) {
    if (it->second == value) {
      it = map.erase(it);
      ++erased;
    } else {
      ++it;
    }
  }
  return erased;
}

//! True if \a map holds any entry whose mapped value is \a value.
template <typename Map, typename Value>
inline bool ContainsValue(const Map& map, const Value& value) {
  for (const auto& entry : map) {
    if (entry.second == value) {
      return true;
    }
  }
  return false;
}

//! True if \a map maps exactly \a key to \a value.
template <typename Map, typename Value>
inline bool ContainsKeyWithValue(const Map& map, typename Map::key_type key, const Value& value) {
  auto it = map.find(key);
  return it != map.end() && it->second == value;
}

}  // namespace amd
