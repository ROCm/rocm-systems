//===-- aegisbit/RegisterHelper.h - Register Number Mapping -----*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Helper for mapping register names to LLVM register numbers.
/// AMDGPU register numbers are not sequential (VGPR0 != 0).
///
//===----------------------------------------------------------------------===//

#ifndef AEGISBIT_REGISTER_HELPER_H
#define AEGISBIT_REGISTER_HELPER_H

#include <cstdint>

namespace aegisbit {

/// Register number mapping for AMDGPU
///
/// LLVM uses non-sequential register numbering. These values are from
/// llvm/lib/Target/AMDGPU/SIRegisterInfo.td and may vary by LLVM version.
/// Validated against LLVM 23.x (gfx942 target).
///
/// Physical register ranges (LLVM 23.x):
///   VGPR0 = 486, VGPR1 = 487, ..., VGPR255 = 741  (256 VGPRs)
///   SGPR0 = 324, SGPR1 = 325, ..., SGPR103 = 427  (104 SGPRs)
///
/// If upgrading LLVM versions, verify these with:
///   llvm-tblgen --dump-json AMDGPU.td | jq '.registers[] | select(.name | startswith("VGPR"))'
///
/// This helper provides the mapping for test construction and analysis.
class RegisterHelper {
public:
  // Constants for register ranges (LLVM 23.x validated)
  static constexpr unsigned VGPR_BASE = 486;   // VGPR0 enum value
  static constexpr unsigned VGPR_COUNT = 512;   // v0-v511 (gfx90a+ unified file)
  static constexpr unsigned SGPR_BASE = 324;
  static constexpr unsigned SGPR_COUNT = 104;
  static constexpr unsigned AGPR_BASE = 51;    // AGPR0 enum value
  static constexpr unsigned AGPR_COUNT = 256;  // a0-a255

  /// Get LLVM register number for VGPR (0-511)
  static constexpr unsigned getVGPR(unsigned Index) {
    return VGPR_BASE + Index;
  }

  /// Get LLVM register number for SGPR (0-103)
  static constexpr unsigned getSGPR(unsigned Index) {
    return SGPR_BASE + Index;
  }

  /// Check if register number is a VGPR
  static constexpr bool isVGPR(unsigned RegNum) {
    return RegNum >= VGPR_BASE && RegNum < VGPR_BASE + VGPR_COUNT;
  }

  /// Check if register number is a SGPR
  static constexpr bool isSGPR(unsigned RegNum) {
    return RegNum >= SGPR_BASE && RegNum < SGPR_BASE + SGPR_COUNT;
  }

  /// Get VGPR index from register number (inverse of getVGPR)
  static constexpr unsigned getVGPRIndex(unsigned RegNum) {
    return RegNum - VGPR_BASE;
  }

  /// Get SGPR index from register number (inverse of getSGPR)
  static constexpr unsigned getSGPRIndex(unsigned RegNum) {
    return RegNum - SGPR_BASE;
  }

  /// Get LLVM register number for AGPR (0-255)
  static constexpr unsigned getAGPR(unsigned Index) {
    return AGPR_BASE + Index;
  }

  /// Check if register number is an AGPR
  static constexpr bool isAGPR(unsigned RegNum) {
    return RegNum >= AGPR_BASE && RegNum < AGPR_BASE + AGPR_COUNT;
  }

  /// Get AGPR index from register number (inverse of getAGPR)
  static constexpr unsigned getAGPRIndex(unsigned RegNum) {
    return RegNum - AGPR_BASE;
  }
};

} // namespace aegisbit

#endif // AEGISBIT_REGISTER_HELPER_H
