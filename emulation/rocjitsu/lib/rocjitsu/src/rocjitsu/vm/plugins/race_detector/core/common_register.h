// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "rocjitsu/isa/arch/amdgpu/shared/memory_issue.h"

#include <cassert>
#include <span>

namespace rocjitsu::plugins::race_detector {

/// Categorizes in-flight memory operations for race detection.
enum class MemoryEventType {
  GLOBAL_TO_VGPR = 0, ///< Load from global memory to VGPR.
  VGPR_TO_GLOBAL,     ///< Store from VGPR to global memory.
  LDS_TO_VGPR,        ///< Load from LDS to VGPR.
  VGPR_TO_LDS,        ///< Store from VGPR to LDS.

  /// Direct-to-LDS (DTL): `buffer_load ... lds` bypasses VGPRs and writes
  /// global memory data directly to LDS.
  ///
  /// The event records the target-specific VMEM/async counter family separately.
  /// The corresponding zero wait must complete before the owning wave can read
  /// the LDS bytes. Cross-wave access additionally requires a barrier, like a
  /// DS write followed by a barrier.
  GLOBAL_TO_LDS,

  GLOBAL_TO_SGPR,   ///< Scalar load; the event stores its counter family separately.
  SCALAR_TO_GLOBAL, ///< Scalar store; retained to preserve partial-wait ordering.

  N
};

/// Race-detector name for the decoded hardware completion-order classification.
using MemoryOrderClass = amdgpu::MemoryCompletionClass;

/// Derive the ordering usable for whole-event hazards such as same-wave LDS
/// overlap. Per-counter ordering remains useful for partial waits, but an event
/// spanning different completion classes is not ordered as a whole.
inline MemoryOrderClass
memoryOrderForObligations(std::span<const amdgpu::MemoryCounterObligation> obligations) {
  if (obligations.empty())
    return MemoryOrderClass::UNORDERED;

  const MemoryOrderClass order = obligations.front().completion_class();
  for (const auto obligation : obligations) {
    if (obligation.completion_class() != order)
      return MemoryOrderClass::UNORDERED;
  }
  return order;
}

/// True if the event writes LDS bytes.
inline bool isToLds(MemoryEventType t) {
  return t == MemoryEventType::VGPR_TO_LDS || t == MemoryEventType::GLOBAL_TO_LDS;
}

/// True if the event touches LDS (read or write).
inline bool isLdsInvolved(MemoryEventType t) {
  return isToLds(t) || t == MemoryEventType::LDS_TO_VGPR;
}

/// True if the event doesn't touch LDS — safe to trim at WAVE_COMPLETE.
inline bool isWaveLocal(MemoryEventType t) { return !isLdsInvolved(t); }

/// Default combined-counter assignment for core callers without dynamic
/// instruction state. Runtime integration passes the exact counter explicitly.
inline amdgpu::WaitCounterType defaultWaitCounterType(MemoryEventType t) {
  switch (t) {
  case MemoryEventType::GLOBAL_TO_VGPR:
  case MemoryEventType::VGPR_TO_GLOBAL:
  case MemoryEventType::GLOBAL_TO_LDS:
    return amdgpu::WaitCounterType::VMCNT;
  case MemoryEventType::LDS_TO_VGPR:
  case MemoryEventType::VGPR_TO_LDS:
  case MemoryEventType::GLOBAL_TO_SGPR:
  case MemoryEventType::SCALAR_TO_GLOBAL:
    return amdgpu::WaitCounterType::LGKMCNT;
  case MemoryEventType::N:
    break;
  }
  assert(false && "invalid memory event type");
  return amdgpu::WaitCounterType::VMCNT;
}

/// Ordering used by architecture-neutral unit-test helpers. Runtime callers
/// pass instruction-specific ordering explicitly.
inline MemoryOrderClass defaultMemoryOrder(MemoryEventType t) {
  switch (t) {
  case MemoryEventType::GLOBAL_TO_VGPR:
  case MemoryEventType::VGPR_TO_GLOBAL:
  case MemoryEventType::GLOBAL_TO_LDS:
    return MemoryOrderClass::VMEM;
  case MemoryEventType::LDS_TO_VGPR:
  case MemoryEventType::VGPR_TO_LDS:
    return MemoryOrderClass::LDS;
  case MemoryEventType::GLOBAL_TO_SGPR:
  case MemoryEventType::SCALAR_TO_GLOBAL:
    return MemoryOrderClass::UNORDERED;
  case MemoryEventType::N:
    break;
  }
  assert(false && "invalid memory event type");
  return MemoryOrderClass::UNORDERED;
}

} // namespace rocjitsu::plugins::race_detector
