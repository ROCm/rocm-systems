// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WaveASM_TARGET_AMDGCN_REGISTERINFO_H
#define WaveASM_TARGET_AMDGCN_REGISTERINFO_H

#include "llvm/ADT/StringRef.h"
#include <optional>
#include <string>

namespace waveasm {

//===----------------------------------------------------------------------===//
// Register Formatting
//===----------------------------------------------------------------------===//

/// Format a single VGPR
inline std::string formatVGPR(int64_t index) {
  return "v" + std::to_string(index);
}

/// Format a VGPR range
inline std::string formatVGPRRange(int64_t baseIndex, int64_t size) {
  if (size == 1) {
    return formatVGPR(baseIndex);
  }
  return "v[" + std::to_string(baseIndex) + ":" +
         std::to_string(baseIndex + size - 1) + "]";
}

/// Format a single SGPR
inline std::string formatSGPR(int64_t index) {
  return "s" + std::to_string(index);
}

/// Format an SGPR range
inline std::string formatSGPRRange(int64_t baseIndex, int64_t size) {
  if (size == 1) {
    return formatSGPR(baseIndex);
  }
  return "s[" + std::to_string(baseIndex) + ":" +
         std::to_string(baseIndex + size - 1) + "]";
}

/// Format a single AGPR
inline std::string formatAGPR(int64_t index) {
  return "a" + std::to_string(index);
}

/// Format an AGPR range
inline std::string formatAGPRRange(int64_t baseIndex, int64_t size) {
  if (size == 1) {
    return formatAGPR(baseIndex);
  }
  return "a[" + std::to_string(baseIndex) + ":" +
         std::to_string(baseIndex + size - 1) + "]";
}

//===----------------------------------------------------------------------===//
// Special Register Names
//===----------------------------------------------------------------------===//

/// Get the name of a special register
inline std::string formatSpecialReg(llvm::StringRef name) {
  // Special registers are used as-is
  return name.str();
}

/// Well-known special register names
namespace SpecialRegs {
constexpr llvm::StringLiteral VCC = "vcc";
constexpr llvm::StringLiteral VCC_LO = "vcc_lo";
constexpr llvm::StringLiteral VCC_HI = "vcc_hi";
constexpr llvm::StringLiteral EXEC = "exec";
constexpr llvm::StringLiteral EXEC_LO = "exec_lo";
constexpr llvm::StringLiteral EXEC_HI = "exec_hi";
constexpr llvm::StringLiteral SCC = "scc";
constexpr llvm::StringLiteral M0 = "m0";
constexpr llvm::StringLiteral FLAT_SCRATCH = "flat_scratch";
constexpr llvm::StringLiteral FLAT_SCRATCH_LO = "flat_scratch_lo";
constexpr llvm::StringLiteral FLAT_SCRATCH_HI = "flat_scratch_hi";
} // namespace SpecialRegs

//===----------------------------------------------------------------------===//
// ABI Register Assignments
//===----------------------------------------------------------------------===//

/// Standard ABI register assignments for kernels
struct KernelABIRegs {
  // VGPR assignments
  static constexpr int64_t WORKITEM_ID_X = 0; // v0 = workitem_id_x
  static constexpr int64_t WORKITEM_ID_Y = 1; // v1 = workitem_id_y (if used)
  static constexpr int64_t WORKITEM_ID_Z = 2; // v2 = workitem_id_z (if used)

  // SGPR assignments (typical, may vary)
  static constexpr int64_t KERNARG_SEGMENT_PTR = 0; // s[0:1]
  static constexpr int64_t DISPATCH_PTR = 2;        // s[2:3] (if enabled)
  static constexpr int64_t WORKGROUP_ID_X = 4;      // s4 or later
  static constexpr int64_t WORKGROUP_ID_Y = 5;      // s5 or later
  static constexpr int64_t WORKGROUP_ID_Z = 6;      // s6 or later
  static constexpr int64_t FLAT_SCRATCH_INIT = 4;   // s[4:5] when enabled
};

//===----------------------------------------------------------------------===//
// Register Constraints
//===----------------------------------------------------------------------===//

/// Get the alignment requirement for a register range size
inline int64_t getAlignmentForSize(int64_t size) {
  // AMDGCN register alignment rules:
  // - Size 1: align 1
  // - Size 2: align 2
  // - Size 3: align 4 (rounded up)
  // - Size 4: align 4
  // - Size 5-8: align 4
  // - Size > 8: align 4 (for most cases)
  // - MFMA accumulators: may require align 4 or higher
  if (size <= 1)
    return 1;
  if (size == 2)
    return 2;
  return 4; // 4-wide alignment for larger ranges
}

/// Check if a register index is valid for a given size and alignment
inline bool isValidRegisterAllocation(int64_t index, int64_t size,
                                      int64_t alignment, int64_t maxRegs) {
  // Check alignment
  if (index % alignment != 0)
    return false;
  // Check bounds
  if (index + size > maxRegs)
    return false;
  return true;
}

//===----------------------------------------------------------------------===//
// Register Class Helpers
//===----------------------------------------------------------------------===//

/// Parse a register string to extract class and index
/// Returns (isVGPR, baseIndex, size) or nullopt if invalid
std::optional<std::tuple<bool, int64_t, int64_t>>
parseRegisterString(llvm::StringRef regStr);

/// Check if a string represents a special register
bool isSpecialRegister(llvm::StringRef regStr);

} // namespace waveasm

#endif // WaveASM_TARGET_AMDGCN_REGISTERINFO_H
