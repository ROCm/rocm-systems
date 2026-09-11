// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_WAIT_COUNTER_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_WAIT_COUNTER_H_

#include <cstdint>

namespace rocjitsu::amdgpu {

/// @brief Counter types tracked by s_waitcnt and related instructions.
///
/// GFX9 (CDNA1-4) uses VMCNT/LGKMCNT/EXPCNT. GFX10 and GFX11 add VSCNT
/// for store-only tracking. On GFX11, the internal LOADCNT/STORECNT and
/// DSCNT/KMCNT names preserve event subsets covered by its aggregate
/// VMCNT/VSCNT and LGKMCNT waits. GFX12 exposes those fine-grained counters
/// directly.
/// GFX12.5 also adds TENSORCNT for tensor data mover operations and ASYNCCNT
/// for async global/cluster transfers to or from LDS.
enum class WaitCounterType : uint8_t {
  VMCNT,     ///< Vector-memory count (loads; GFX9 stores too).
  LGKMCNT,   ///< LDS/GDS/K(Constant)/Message count.
  EXPCNT,    ///< Export count.
  VSCNT,     ///< Vector store count (GFX10/11 -- S_WAITCNT_VSCNT).
  LOADCNT,   ///< Vector-load subtype on GFX11; architectural counter on GFX12+.
  STORECNT,  ///< Vector-store subtype on GFX11; architectural counter on GFX12+.
  DSCNT,     ///< DS subtype on GFX11; architectural counter on GFX12+.
  KMCNT,     ///< Scalar-memory subtype on GFX11; architectural counter on GFX12+.
  TENSORCNT, ///< Tensor data mover count (GFX12.5).
  ASYNCCNT,  ///< Async global/cluster LDS transfer count (GFX12.5).
};

/// @brief Return whether waiting on @p wait_type also constrains an event
/// tracked by @p event_type.
/// @details Monolithic counters cover the corresponding split-counter
/// families. Split waits constrain only their exact counter.
[[nodiscard]] constexpr bool wait_counter_covers(WaitCounterType wait_type,
                                                 WaitCounterType event_type) {
  switch (wait_type) {
  case WaitCounterType::VMCNT:
    return event_type == WaitCounterType::VMCNT || event_type == WaitCounterType::LOADCNT;
  case WaitCounterType::LGKMCNT:
    return event_type == WaitCounterType::LGKMCNT || event_type == WaitCounterType::DSCNT ||
           event_type == WaitCounterType::KMCNT;
  case WaitCounterType::VSCNT:
    return event_type == WaitCounterType::VSCNT || event_type == WaitCounterType::STORECNT;
  case WaitCounterType::EXPCNT:
  case WaitCounterType::LOADCNT:
  case WaitCounterType::STORECNT:
  case WaitCounterType::DSCNT:
  case WaitCounterType::KMCNT:
  case WaitCounterType::TENSORCNT:
  case WaitCounterType::ASYNCCNT:
    return event_type == wait_type;
  }
  return false;
}

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_WAIT_COUNTER_H_
