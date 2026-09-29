// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string_view>

namespace rocjitsu {

/// @brief Internal wait counters tracked by waitcheck.
///
/// @details GFX12 targets use the split counter names directly. Legacy gfx9
/// style targets reuse Load for vmcnt, Ds for lgkmcnt, and Exp for expcnt.
enum class WaitCounterKind : uint8_t {
  Load = 0,
  Store,
  Ds,
  Km,
  Sample,
  Bvh,
  Exp,
  X,
  Async,
  Tensor,
  VmVsrc,
  VaVdst,
  Depctr,
  Count,
};

/// @brief Human-readable split counter name, e.g. "loadcnt".
[[nodiscard]] inline std::string_view wait_counter_name(WaitCounterKind counter) {
  switch (counter) {
  case WaitCounterKind::Load:
    return "loadcnt";
  case WaitCounterKind::Store:
    return "storecnt";
  case WaitCounterKind::Ds:
    return "dscnt";
  case WaitCounterKind::Km:
    return "kmcnt";
  case WaitCounterKind::Sample:
    return "samplecnt";
  case WaitCounterKind::Bvh:
    return "bvhcnt";
  case WaitCounterKind::Exp:
    return "expcnt";
  case WaitCounterKind::X:
    return "xcnt";
  case WaitCounterKind::Async:
    return "asynccnt";
  case WaitCounterKind::Tensor:
    return "tensorcnt";
  case WaitCounterKind::VmVsrc:
    return "depctr_vm_vsrc";
  case WaitCounterKind::VaVdst:
    return "wait_va_vdst";
  case WaitCounterKind::Depctr:
    return "depctr";
  case WaitCounterKind::Count:
    break;
  }
  return "unknown";
}

} // namespace rocjitsu
