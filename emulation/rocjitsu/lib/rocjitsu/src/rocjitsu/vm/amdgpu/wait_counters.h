// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_VM_AMDGPU_WAIT_COUNTERS_H_
#define ROCJITSU_VM_AMDGPU_WAIT_COUNTERS_H_

#include <algorithm>
#include <cassert>
#include <cstdint>

namespace rocjitsu {
namespace amdgpu {

/// @brief Counter types tracked by s_waitcnt and related instructions.
///
/// GFX9 (CDNA1-4) and GFX10 (RDNA1/2) use VMCNT/LGKMCNT/EXPCNT.
/// GFX10 adds VSCNT for store-only tracking (S_WAITCNT_VSCNT).
/// GFX11 (RDNA3/3.5) and GFX12 (RDNA4) use fine-grained counters:
///   LOADCNT (≡ VMCNT for loads), STORECNT (≡ VSCNT), DSCNT (DS subset
///   of LGKMCNT), KMCNT (scalar/constant subset of LGKMCNT), EXPCNT.
enum class WaitCounterType : uint8_t {
  VMCNT,    ///< Vector memory count (global loads; GFX9/10 stores too).
  LGKMCNT,  ///< LDS/GDS/K(Constant)/Message count.
  EXPCNT,   ///< Export count.
  VSCNT,    ///< Vector store count (GFX10 only — S_WAITCNT_VSCNT).
  LOADCNT,  ///< Vector load count (GFX11+; alias of VMCNT for loads).
  STORECNT, ///< Vector store count (GFX11+; alias of VSCNT).
  DSCNT,    ///< DS (LDS/GDS) count (GFX11+; subset of LGKMCNT).
  KMCNT,    ///< Scalar/constant memory count (GFX11+; subset of LGKMCNT).
};

/// @brief Outstanding memory operation counters for a wavefront.
///
/// Unified counter set that covers all ISA families.  CDNA1-4 use only
/// vmcnt/lgkmcnt/expcnt.  RDNA1/2 add vscnt.  RDNA3/3.5/4 use the
/// fine-grained counters (loadcnt/storecnt/dscnt/kmcnt/expcnt) which
/// are stored in the same fields as aliases:
///   loadcnt → vmcnt, storecnt → vscnt, dscnt + kmcnt → lgkmcnt.
///
/// For RDNA3+, the split counters (dscnt, kmcnt) are tracked independently
/// but their sum is also reflected in lgkmcnt for backward compatibility
/// with the monolithic S_WAITCNT instruction that RDNA3/3.5 still support.
struct WaitCounters {
  // GFX9/10 counters (primary)
  uint8_t vmcnt = 0;   ///< VMEM load count (GFX9/10) / loadcnt alias (GFX11+).
  uint8_t lgkmcnt = 0; ///< LDS+GDS+K+M count (GFX9/10) / sum of dscnt+kmcnt (GFX11+).
  uint8_t expcnt = 0;  ///< Export count (all ISAs).
  uint8_t vscnt = 0;   ///< Vector store count (GFX10) / storecnt alias (GFX11+).

  // GFX11+ fine-grained counters
  uint8_t dscnt = 0; ///< DS (LDS/GDS) count (GFX11+).
  uint8_t kmcnt = 0; ///< Scalar/constant memory count (GFX11+).

  /// @brief Check whether all counters are zero (no outstanding memory ops).
  bool empty() const {
    return vmcnt == 0 && lgkmcnt == 0 && expcnt == 0 && vscnt == 0 && dscnt == 0 && kmcnt == 0;
  }

  /// Hardware saturation limits for each counter type.
  static constexpr uint8_t VMCNT_MAX = 63;
  static constexpr uint8_t LGKMCNT_MAX = 63;
  static constexpr uint8_t EXPCNT_MAX = 7;
  static constexpr uint8_t VSCNT_MAX = 63;
  static constexpr uint8_t DSCNT_MAX = 63;
  static constexpr uint8_t KMCNT_MAX = 31;

  void increment(WaitCounterType type) {
    switch (type) {
    case WaitCounterType::VMCNT:
    case WaitCounterType::LOADCNT:
      vmcnt = std::min<uint8_t>(vmcnt + 1, VMCNT_MAX);
      break;
    case WaitCounterType::LGKMCNT:
      lgkmcnt = std::min<uint8_t>(lgkmcnt + 1, LGKMCNT_MAX);
      break;
    case WaitCounterType::EXPCNT:
      expcnt = std::min<uint8_t>(expcnt + 1, EXPCNT_MAX);
      break;
    case WaitCounterType::VSCNT:
    case WaitCounterType::STORECNT:
      vscnt = std::min<uint8_t>(vscnt + 1, VSCNT_MAX);
      break;
    case WaitCounterType::DSCNT:
      dscnt = std::min<uint8_t>(dscnt + 1, DSCNT_MAX);
      lgkmcnt = std::min<uint8_t>(lgkmcnt + 1, LGKMCNT_MAX);
      break;
    case WaitCounterType::KMCNT:
      kmcnt = std::min<uint8_t>(kmcnt + 1, KMCNT_MAX);
      lgkmcnt = std::min<uint8_t>(lgkmcnt + 1, LGKMCNT_MAX);
      break;
    }
  }

  void decrement(WaitCounterType type) {
    switch (type) {
    case WaitCounterType::VMCNT:
    case WaitCounterType::LOADCNT:
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
    case WaitCounterType::VSCNT:
    case WaitCounterType::STORECNT:
      assert(vscnt > 0 && "VSCNT underflow");
      --vscnt;
      break;
    case WaitCounterType::DSCNT:
      assert(dscnt > 0 && "DSCNT underflow");
      --dscnt;
      assert(lgkmcnt > 0 && "LGKMCNT underflow from DSCNT");
      --lgkmcnt;
      break;
    case WaitCounterType::KMCNT:
      assert(kmcnt > 0 && "KMCNT underflow");
      --kmcnt;
      assert(lgkmcnt > 0 && "LGKMCNT underflow from KMCNT");
      --lgkmcnt;
      break;
    }
  }
};

/// @brief Target counter thresholds for s_waitcnt and split-wait instructions.
///
/// A wavefront stalls until all of its WaitCounters are at or below the
/// corresponding target value. Max values mean "don't wait on this counter".
struct WaitTarget {
  uint8_t vmcnt = WaitCounters::VMCNT_MAX;
  uint8_t lgkmcnt = WaitCounters::LGKMCNT_MAX;
  uint8_t expcnt = WaitCounters::EXPCNT_MAX;
  uint8_t vscnt = WaitCounters::VSCNT_MAX;
  uint8_t dscnt = WaitCounters::DSCNT_MAX;
  uint8_t kmcnt = WaitCounters::KMCNT_MAX;

  /// @brief Check whether the given counters satisfy all thresholds.
  bool satisfied(const WaitCounters &c) const {
    return c.vmcnt <= vmcnt && c.lgkmcnt <= lgkmcnt && c.expcnt <= expcnt && c.vscnt <= vscnt &&
           c.dscnt <= dscnt && c.kmcnt <= kmcnt;
  }
};

} // namespace amdgpu
} // namespace rocjitsu

#endif // ROCJITSU_VM_AMDGPU_WAIT_COUNTERS_H_
