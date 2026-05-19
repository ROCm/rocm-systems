/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifdef HIP_GRAPH_DISPATCH_OPTIMIZED

#include "device/rocm/rocdevice.hpp"
#include "device/rocm/rocsignal.hpp"

namespace amd::roc {

// ================================================================================================
void Device::ResetHwEvents(std::vector<void*>& hw_events) const {
  for (auto* hw_event : hw_events) {
    if (hw_event != nullptr) {
      Hsa::signal_silent_store_relaxed(
          reinterpret_cast<ProfilingSignal*>(hw_event)->signal_, kInitSignalValueOne);
    }
  }
}

// ================================================================================================
void Device::ClearHwEvent(void* hw_event) const {
  if (hw_event != nullptr) {
    Hsa::signal_store_relaxed(
        reinterpret_cast<ProfilingSignal*>(hw_event)->signal_, 0);
  }
}

}  // namespace amd::roc

#endif  // HIP_GRAPH_DISPATCH_OPTIMIZED
