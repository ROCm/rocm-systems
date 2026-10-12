/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <type_traits>
#if defined(_WIN32)
#include <atomic>
#endif

namespace hip {

// Loads one function-pointer slot of a live HIP dispatch table.
//
// Tools such as HRR capture write slots of the live table while other threads call
// through it. Each write is a single release store of one aligned pointer (see
// install_shims in hrr/hip_capture.cpp). The HIP entry points in hip_table_interface.cpp
// read each slot with the matching acquire load, so a caller gets either the old function
// or the new one, and the new function sees everything its tool wrote before installing
// it. An acquire load of a pointer is a plain load on x86-64 and LDAR on AArch64.
template <typename Fn> inline Fn LoadDispatchSlot(const Fn& slot) {
  static_assert(std::is_pointer_v<Fn> && sizeof(Fn) == sizeof(void*),
                "dispatch slots are function pointers the size of void*");
#if defined(_WIN32)
  // Same view as the Windows store in hrr/hip_capture.cpp.
  void*& raw = reinterpret_cast<void*&>(const_cast<Fn&>(slot));
  return reinterpret_cast<Fn>(std::atomic_ref<void*>(raw).load(std::memory_order_acquire));
#else
  return __atomic_load_n(&slot, __ATOMIC_ACQUIRE);
#endif
}

}  // namespace hip
