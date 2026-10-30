/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#if defined(_WIN32)
#include <atomic>
#endif

namespace hip {

// Acquire load of one slot of a live HIP dispatch table, pairing with the release store
// HRR capture installs its shims with (projects/hrr/DESIGN.md, "Core Mechanism").
template <typename Fn> inline Fn LoadDispatchSlot(const Fn& slot) {
#if defined(_WIN32)
  // Same view as the Windows store in hrr/hip_capture.cpp.
  void*& raw = reinterpret_cast<void*&>(const_cast<Fn&>(slot));
  return reinterpret_cast<Fn>(std::atomic_ref<void*>(raw).load(std::memory_order_acquire));
#else
  return __atomic_load_n(&slot, __ATOMIC_ACQUIRE);
#endif
}

}  // namespace hip
