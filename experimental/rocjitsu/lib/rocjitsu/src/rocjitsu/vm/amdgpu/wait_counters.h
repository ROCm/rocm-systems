// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_WAIT_COUNTERS_H_
#define ROCJITSU_VM_AMDGPU_WAIT_COUNTERS_H_

#include <algorithm>
#include <cassert>
#include <cstdint>

namespace rocjitsu {
namespace amdgpu {

/// @brief Counter types tracked by s_waitcnt.
enum class WaitCounterType : uint8_t {
  VMCNT,   ///< Vector memory count (global loads/stores).
  LGKMCNT, ///< LDS/GDS/K(Constant)/Message count (scalar + local memory).
  EXPCNT,  ///< Export count (VGPR availability after export).
};

/// @brief Outstanding memory operation counters for a wavefront.
///
/// Each counter tracks the number of in-flight operations of a given type.
/// The hardware decrements automatically on completion; s_waitcnt stalls
/// the wavefront until counters fall to the specified thresholds.
struct WaitCounters {
  uint8_t vmcnt = 0;   ///< VMEM count.
  uint8_t lgkmcnt = 0; ///< Local, global, constant count.
  uint8_t expcnt = 0;  ///< Export count.

  /// Hardware saturation limits for each counter type.
  static constexpr uint8_t VMCNT_MAX = 63;
  static constexpr uint8_t LGKMCNT_MAX = 31;
  static constexpr uint8_t EXPCNT_MAX = 7;

  void increment(WaitCounterType type) {
    switch (type) {
    case WaitCounterType::VMCNT:
      vmcnt = std::min<uint8_t>(vmcnt + 1, VMCNT_MAX);
      break;
    case WaitCounterType::LGKMCNT:
      lgkmcnt = std::min<uint8_t>(lgkmcnt + 1, LGKMCNT_MAX);
      break;
    case WaitCounterType::EXPCNT:
      expcnt = std::min<uint8_t>(expcnt + 1, EXPCNT_MAX);
      break;
    }
  }

  void decrement(WaitCounterType type) {
    switch (type) {
    case WaitCounterType::VMCNT:
      assert(vmcnt > 0 && "VMCNT underflow");
      --vmcnt;
      break;
    case WaitCounterType::LGKMCNT:
      assert(lgkmcnt > 0 && "LGKMCNT underflow");
      --lgkmcnt;
      break;
    case WaitCounterType::EXPCNT:
      assert(expcnt > 0 && "EXPCNT underflow");
      --expcnt;
      break;
    }
  }
};

/// @brief Target counter thresholds for s_waitcnt.
///
/// A wavefront stalls until all of its WaitCounters are at or below the
/// corresponding target value. Max values mean "don't wait on this counter".
struct WaitTarget {
  uint8_t vmcnt = 63;
  uint8_t lgkmcnt = 31;
  uint8_t expcnt = 7;

  /// @brief Check whether the given counters satisfy all thresholds.
  bool satisfied(const WaitCounters &c) const {
    return c.vmcnt <= vmcnt && c.lgkmcnt <= lgkmcnt && c.expcnt <= expcnt;
  }
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_WAIT_COUNTERS_H_
