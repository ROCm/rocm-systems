// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "rocjitsu/isa/arch/amdgpu/shared/memory_issue.h"
#include "rocjitsu/isa/arch/amdgpu/shared/wait_counter_policy.h"

#include <array>
#include <cassert>
#include <cstddef>
#include <optional>
#include <span>

namespace rocjitsu::plugins::race_detector {

struct ConditionalVgprWrite {
  uint32_t reg = 0;
  uint64_t lanes = 0;
};

/// Categorizes in-flight memory operations for race detection.
enum class MemoryEventType {
  GLOBAL_TO_VGPR = 0, ///< Load from global memory to VGPR.
  VGPR_TO_GLOBAL,     ///< Store from VGPR to global memory.
  LDS_TO_VGPR,        ///< Load from LDS to VGPR.
  VGPR_TO_LDS,        ///< Store from VGPR to LDS.

  /// Direct-to-LDS (DTL): `buffer_load ... lds` bypasses VGPRs and writes
  /// global memory data directly to LDS. Unlike VGPR_TO_LDS (which has VGPR
  /// registers), GLOBAL_TO_LDS events have no VGPR registers -- only LDS
  /// intervals.
  ///
  /// The event records the target-specific VMEM/async counter family separately.
  /// The corresponding zero wait must complete before the owning wave can read
  /// the LDS bytes. Cross-wave access additionally requires a barrier, like a
  /// DS write followed by a barrier.
  GLOBAL_TO_LDS,

  GLOBAL_TO_SGPR,   ///< Scalar load to an SGPR; the event stores its counter family separately.
  GLOBAL_TO_TTMP,   ///< Scalar load to a TTMP; the event stores its counter family separately.
  SCALAR_TO_GLOBAL, ///< Scalar store; retained to preserve partial-wait ordering.

  /// LDS-to-global transfer, such as a tensor DMA store: reads LDS and writes
  /// global memory, with no VGPR registers -- only LDS intervals. The reverse
  /// direction is GLOBAL_TO_LDS.
  ///
  /// The event records its counter family and completion order separately.
  /// defaultWaitCounterType and defaultMemoryOrder return those of a tensor DMA
  /// store (TENSORCNT, TENSOR).
  LDS_TO_GLOBAL,

  /// A synchronous VALU operand read visible to peers until a barrier.
  LDS_OPERAND_READ,

};

/// Race-detector name for the decoded hardware completion-order classification.
using MemoryOrderClass = amdgpu::MemoryCompletionClass;

/// VMEM result writes can use distinct hardware FIFOs while sharing a counter.
class VgprWriteOrder {
public:
  constexpr VgprWriteOrder(MemoryOrderClass completion,
                           amdgpu::VmemWriteback type = amdgpu::VmemWriteback::Ordinary)
      : completion(completion), type(type) {}
  bool operator==(const VgprWriteOrder &) const = default;
  MemoryOrderClass completion;
  amdgpu::VmemWriteback type;
};

struct LdsAccess {
  bool read_modify_write = false;
  uint8_t aligned_atomic_bytes = 0;
};

/// Derive the ordering usable for whole-event hazards such as destination WAW.
/// Per-counter ordering remains useful for partial waits, but an event spanning
/// different completion classes is not ordered as a whole.
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

/// True if the event loads into a VGPR.
inline bool isToVgpr(MemoryEventType t) {
  return t == MemoryEventType::GLOBAL_TO_VGPR || t == MemoryEventType::LDS_TO_VGPR;
}

inline bool isToLds(MemoryEventType t) {
  return t == MemoryEventType::VGPR_TO_LDS || t == MemoryEventType::GLOBAL_TO_LDS;
}

inline bool isToSgpr(MemoryEventType t) { return t == MemoryEventType::GLOBAL_TO_SGPR; }

inline bool isToTtmp(MemoryEventType t) { return t == MemoryEventType::GLOBAL_TO_TTMP; }

inline bool isFromLds(MemoryEventType t) {
  return t == MemoryEventType::LDS_TO_VGPR || t == MemoryEventType::LDS_TO_GLOBAL ||
         t == MemoryEventType::LDS_OPERAND_READ;
}

/// True if the event touches LDS (read or write).
inline bool isLdsInvolved(MemoryEventType t) { return isToLds(t) || isFromLds(t); }

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
  case MemoryEventType::GLOBAL_TO_TTMP:
  case MemoryEventType::SCALAR_TO_GLOBAL:
    return amdgpu::WaitCounterType::LGKMCNT;
  case MemoryEventType::LDS_TO_GLOBAL:
    return amdgpu::WaitCounterType::TENSORCNT;
  case MemoryEventType::LDS_OPERAND_READ:
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
  case MemoryEventType::GLOBAL_TO_TTMP:
  case MemoryEventType::SCALAR_TO_GLOBAL:
    return MemoryOrderClass::UNORDERED;
  case MemoryEventType::LDS_TO_GLOBAL:
    return MemoryOrderClass::TENSOR;
  case MemoryEventType::LDS_OPERAND_READ:
    break;
  }
  assert(false && "invalid memory event type");
  return MemoryOrderClass::UNORDERED;
}

/// @brief Owns one or two unit counter increments with the same completion order.
/// @details The converted span borrows this object's storage. Event registration
/// consumes it before returning, so passing a temporary directly is safe. Keep
/// the owner alive when storing a span for later use. Use an explicit obligation
/// array for different completion orders or non-unit counter increments.
class CounterObligations {
public:
  CounterObligations(
      amdgpu::WaitCounterType wait_counter_type, MemoryOrderClass memory_order,
      std::optional<amdgpu::WaitCounterType> additional_wait_counter_type = std::nullopt)
      : obligations_{amdgpu::MemoryCounterObligation{wait_counter_type, memory_order}, {}} {
    if (additional_wait_counter_type)
      obligations_[count_++] = {*additional_wait_counter_type, memory_order};
  }

  /// @brief The default counter and order for an event type.
  explicit CounterObligations(MemoryEventType type)
      : CounterObligations(defaultWaitCounterType(type), defaultMemoryOrder(type)) {}

  operator std::span<const amdgpu::MemoryCounterObligation>() const {
    return {obligations_.data(), count_};
  }

private:
  std::array<amdgpu::MemoryCounterObligation, 2> obligations_;
  std::size_t count_ = 1;
};

} // namespace rocjitsu::plugins::race_detector
