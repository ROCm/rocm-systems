// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once
#include "rocjitsu/vm/amdgpu/wait_counters.h"

#include <ostream>
#include <sstream>
#include <string>

namespace rocjitsu::plugins::race_detector {

/// Categorizes in-flight memory operations for race detection.
enum class MemoryEventType {
  GLOBAL_TO_VGPR = 0, ///< Load from global memory to VGPR (counted by vmcnt).
  VGPR_TO_GLOBAL,     ///< Store from VGPR to global memory (counted by vmcnt).
  LDS_TO_VGPR,        ///< Load from LDS to VGPR (counted by lgkmcnt).
  VGPR_TO_LDS,        ///< Store from VGPR to LDS (counted by lgkmcnt).

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

  N
};

/// Event direction helpers: "to VGPR" means a load writing into a VGPR,
/// "from VGPR" means a store reading out of a VGPR.
inline bool isToVgpr(MemoryEventType t) {
  return t == MemoryEventType::GLOBAL_TO_VGPR || t == MemoryEventType::LDS_TO_VGPR;
}

inline bool isToLds(MemoryEventType t) {
  return t == MemoryEventType::VGPR_TO_LDS || t == MemoryEventType::GLOBAL_TO_LDS;
}

inline bool isToSgpr(MemoryEventType t) { return t == MemoryEventType::GLOBAL_TO_SGPR; }

inline bool isToTtmp(MemoryEventType t) { return t == MemoryEventType::GLOBAL_TO_TTMP; }

inline bool isToScalar(MemoryEventType t) { return isToSgpr(t) || isToTtmp(t); }

/// True if the event touches LDS (read or write).
inline bool isLdsInvolved(MemoryEventType t) {
  return isToLds(t) || t == MemoryEventType::LDS_TO_VGPR;
}

inline bool isFromVgpr(MemoryEventType t) {
  return t == MemoryEventType::VGPR_TO_GLOBAL || t == MemoryEventType::VGPR_TO_LDS;
}

/// True if the event doesn't touch LDS — safe to trim at WAVE_COMPLETE.
inline bool isWaveLocal(MemoryEventType t) { return !isLdsInvolved(t); }

/// Legacy wait-counter assignment for core callers without dynamic
/// instruction state. Runtime integration passes the exact counter explicitly.
inline amdgpu::WaitCounterType defaultWaitCounterType(MemoryEventType t) {
  if (t == MemoryEventType::GLOBAL_TO_VGPR || t == MemoryEventType::VGPR_TO_GLOBAL ||
      t == MemoryEventType::GLOBAL_TO_LDS)
    return amdgpu::WaitCounterType::VMCNT;
  return amdgpu::WaitCounterType::LGKMCNT;
}

/// A register reference (type + index).
class CommonRegister {
public:
  enum class Type { SGPR, VGPR, UNKNOWN };
  Type type;
  int index;

  static CommonRegister getVgpr(int idx) { return CommonRegister{Type::VGPR, idx}; }

  void appendStr(std::ostream &os) const {
    char prefix = (type == Type::SGPR) ? 's' : (type == Type::VGPR) ? 'v' : '?';
    os << prefix << index;
  }

  std::string str() const {
    std::ostringstream oss;
    appendStr(oss);
    return oss.str();
  }
};

inline std::ostream &operator<<(std::ostream &os, const CommonRegister &reg) {
  reg.appendStr(os);
  return os;
}

} // namespace rocjitsu::plugins::race_detector
