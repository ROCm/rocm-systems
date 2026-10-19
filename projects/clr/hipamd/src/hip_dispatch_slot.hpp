/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "utils/macros.hpp"

#include <atomic>
#include <cstddef>

// Puts a layer of wrappers into a live HIP dispatch table and takes it out
// again. HRR capture and the CLR profiler both do this. Other threads keep
// calling through the table meanwhile, so it is written one slot at a time with
// atomic operations and never copied over as a whole. Outside Windows,
// function-pointer slots are accessed through a may_alias view so the store is
// well-defined against the typed table fields; MSVC does no type-based alias
// analysis and clang-cl disables it by default.
namespace hip {
namespace dispatch_slot {

template <typename Table>
constexpr size_t kSlots = (sizeof(Table) - sizeof(size_t)) / sizeof(void*);

#if IS_WINDOWS
struct Slot {
  void* value;
};

inline void Store(Slot* slot, void* value) {
  std::atomic_ref<void*>(slot->value).store(value, std::memory_order_release);
}

inline void Replace(Slot* slot, void* expected, void* desired) {
  std::atomic_ref<void*>(slot->value).compare_exchange_strong(
      expected, desired, std::memory_order_release, std::memory_order_relaxed);
}
#elif defined(__GNUC__) || defined(__clang__)
struct Slot {
  void* value;
} __attribute__((__may_alias__));

inline void Store(Slot* slot, void* value) {
  __atomic_store_n(&slot->value, value, __ATOMIC_RELEASE);
}

inline void Replace(Slot* slot, void* expected, void* desired) {
  __atomic_compare_exchange_n(&slot->value, &expected, desired, false, __ATOMIC_RELEASE,
                              __ATOMIC_RELAXED);
}
#else
#error "Outside Windows, HIP dispatch slots need GCC or Clang"
#endif

template <typename Table> constexpr void CheckLayout() {
  static_assert(sizeof(void (*)()) == sizeof(void*),
                "dispatch slot atomics need function pointers the size of void*");
  static_assert(offsetof(Table, size) == 0 && sizeof(Table::size) == sizeof(size_t) &&
                    (sizeof(Table) - sizeof(size_t)) % sizeof(void*) == 0,
                "layout must match HIP dispatch tables: size_t size then void* slots");
  static_assert(sizeof(Slot) == sizeof(void*), "Slot must be a single pointer");
}

template <typename Table> Slot* Slots(Table& table) {
  CheckLayout<Table>();
  return reinterpret_cast<Slot*>(reinterpret_cast<char*>(&table) + sizeof(size_t));
}

template <typename Table> const Slot* Slots(const Table& table) {
  CheckLayout<Table>();
  return reinterpret_cast<const Slot*>(reinterpret_cast<const char*>(&table) + sizeof(size_t));
}

}  // namespace dispatch_slot

// Stores wrappers[i] in every slot of `live` where it differs from next[i], the
// table the wrappers call on to. Neither the size field nor any other slot is
// written. Unlike ReplaceDispatchSlots() this does not compare first: a slot that
// changed since `next` was taken still gets its wrapper.
template <typename Table>
void InstallDispatchSlots(Table* live, const Table& wrappers, const Table& next) {
  dispatch_slot::Slot* slot = dispatch_slot::Slots(*live);
  const dispatch_slot::Slot* wrapper = dispatch_slot::Slots(wrappers);
  const dispatch_slot::Slot* orig = dispatch_slot::Slots(next);
  for (size_t i = 0; i < dispatch_slot::kSlots<Table>; ++i) {
    if (wrapper[i].value != orig[i].value) dispatch_slot::Store(&slot[i], wrapper[i].value);
  }
}

// Puts to[i] in every slot where from[i] and to[i] differ, but only while the
// slot still holds from[i]. A slot that another component changed keeps its new
// value. With from = wrappers and to = next it takes a layer out; the other way
// round it puts one in.
template <typename Table>
void ReplaceDispatchSlots(Table* live, const Table& from, const Table& to) {
  dispatch_slot::Slot* slot = dispatch_slot::Slots(*live);
  const dispatch_slot::Slot* old_value = dispatch_slot::Slots(from);
  const dispatch_slot::Slot* new_value = dispatch_slot::Slots(to);
  for (size_t i = 0; i < dispatch_slot::kSlots<Table>; ++i) {
    if (old_value[i].value != new_value[i].value)
      dispatch_slot::Replace(&slot[i], old_value[i].value, new_value[i].value);
  }
}

}  // namespace hip
