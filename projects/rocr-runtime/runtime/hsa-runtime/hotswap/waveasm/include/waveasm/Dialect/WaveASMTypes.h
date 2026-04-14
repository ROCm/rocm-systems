// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WaveASM_DIALECT_WAVEASMTYPES_H
#define WaveASM_DIALECT_WAVEASMTYPES_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Types.h"

// Include the generated enum declarations
#include "waveasm/Dialect/WaveASMEnums.h.inc"

#define GET_TYPEDEF_CLASSES
#include "waveasm/Dialect/WaveASMTypes.h.inc"

namespace waveasm {

// Helper functions for register type classification

/// Check if a type is a virtual register type
inline bool isVirtualRegType(mlir::Type type) {
  return mlir::isa<VRegType, SRegType, ARegType>(type);
}

/// Check if a type is a physical register type
inline bool isPhysicalRegType(mlir::Type type) {
  return mlir::isa<PVRegType, PSRegType, PARegType>(type);
}

/// Check if a type is any register type
inline bool isRegType(mlir::Type type) {
  return mlir::isa<VRegType, SRegType, ARegType, PVRegType, PSRegType,
                   PARegType>(type);
}

/// Check if a type is a VGPR type (virtual or physical)
inline bool isVGPRType(mlir::Type type) {
  return mlir::isa<VRegType, PVRegType>(type);
}

/// Check if a type is an AGPR type (virtual or physical)
inline bool isAGPRType(mlir::Type type) {
  return mlir::isa<ARegType, PARegType>(type);
}

/// Check if a type is an SGPR type (virtual or physical)
inline bool isSGPRType(mlir::Type type) {
  return mlir::isa<SRegType, PSRegType>(type);
}

/// Get the register class for a type
inline std::optional<RegClass> getRegClass(mlir::Type type) {
  if (mlir::isa<VRegType, PVRegType>(type))
    return RegClass::VGPR;
  if (mlir::isa<SRegType, PSRegType>(type))
    return RegClass::SGPR;
  if (mlir::isa<ARegType, PARegType>(type))
    return RegClass::AGPR;
  return std::nullopt;
}

/// Get the register size for a type
inline int64_t getRegSize(mlir::Type type) {
  if (auto vreg = mlir::dyn_cast<VRegType>(type))
    return vreg.getSize();
  if (auto sreg = mlir::dyn_cast<SRegType>(type))
    return sreg.getSize();
  if (auto pvreg = mlir::dyn_cast<PVRegType>(type))
    return pvreg.getSize();
  if (auto psreg = mlir::dyn_cast<PSRegType>(type))
    return psreg.getSize();
  if (auto areg = mlir::dyn_cast<ARegType>(type))
    return areg.getSize();
  if (auto pareg = mlir::dyn_cast<PARegType>(type))
    return pareg.getSize();
  return 1;
}

/// Get register alignment for a type
inline int64_t getRegAlignment(mlir::Type type) {
  if (auto vreg = mlir::dyn_cast<VRegType>(type))
    return vreg.getAlignment();
  if (auto sreg = mlir::dyn_cast<SRegType>(type))
    return sreg.getAlignment();
  if (auto areg = mlir::dyn_cast<ARegType>(type))
    return areg.getAlignment();
  // Physical registers don't have alignment (already allocated)
  return 1;
}

/// Check if type is an immediate
inline bool isImmType(mlir::Type type) { return mlir::isa<ImmType>(type); }

/// Check if type is the SCC flag type
inline bool isSCCType(mlir::Type type) { return mlir::isa<SCCType>(type); }

/// Check if two types are structurally compatible for control flow.
/// After register allocation, virtual types (vreg, sreg) become physical
/// types (pvreg, psreg) which include a register index. For control flow
/// verification, we only care that the register class and size match,
/// not the specific physical register index.
inline bool typesCompatible(mlir::Type a, mlir::Type b) {
  if (a == b)
    return true;

  // Both are SCC types (parameterless, so a == b already covers this,
  // but be explicit for clarity).
  if (mlir::isa<SCCType>(a) && mlir::isa<SCCType>(b))
    return true;

  // Both are VReg types (virtual)
  if (auto va = mlir::dyn_cast<VRegType>(a))
    if (auto vb = mlir::dyn_cast<VRegType>(b))
      return va.getSize() == vb.getSize();

  // Both are PVReg types (physical) - compare size only, not index
  if (auto pa = mlir::dyn_cast<PVRegType>(a))
    if (auto pb = mlir::dyn_cast<PVRegType>(b))
      return pa.getSize() == pb.getSize();

  // Both are SReg types (virtual)
  if (auto sa = mlir::dyn_cast<SRegType>(a))
    if (auto sb = mlir::dyn_cast<SRegType>(b))
      return sa.getSize() == sb.getSize();

  // Both are PSReg types (physical) - compare size only, not index
  if (auto pa = mlir::dyn_cast<PSRegType>(a))
    if (auto pb = mlir::dyn_cast<PSRegType>(b))
      return pa.getSize() == pb.getSize();

  // Both are AReg types (virtual)
  if (auto aa = mlir::dyn_cast<ARegType>(a))
    if (auto ab = mlir::dyn_cast<ARegType>(b))
      return aa.getSize() == ab.getSize();

  // Both are PAReg types (physical) - compare size only, not index
  if (auto aa = mlir::dyn_cast<PARegType>(a))
    if (auto ab = mlir::dyn_cast<PARegType>(b))
      return aa.getSize() == ab.getSize();

  // Cross virtual/physical: vreg vs pvreg (same register class)
  if ((mlir::isa<VRegType>(a) && mlir::isa<PVRegType>(b)) ||
      (mlir::isa<PVRegType>(a) && mlir::isa<VRegType>(b)))
    return true; // Allow during mixed states (mid-regalloc)

  if ((mlir::isa<SRegType>(a) && mlir::isa<PSRegType>(b)) ||
      (mlir::isa<PSRegType>(a) && mlir::isa<SRegType>(b)))
    return true; // Allow during mixed states (mid-regalloc)

  if ((mlir::isa<ARegType>(a) && mlir::isa<PARegType>(b)) ||
      (mlir::isa<PARegType>(a) && mlir::isa<ARegType>(b)))
    return true; // Allow during mixed states (mid-regalloc)

  return false;
}

/// Check that two type ranges are pairwise compatible (same size and each
/// element pair passes typesCompatible).
inline bool typesCompatible(mlir::TypeRange a, mlir::TypeRange b) {
  if (a.size() != b.size())
    return false;
  for (auto [ta, tb] : llvm::zip(a, b))
    if (!typesCompatible(ta, tb))
      return false;
  return true;
}

} // namespace waveasm

#endif // WaveASM_DIALECT_WAVEASMTYPES_H
