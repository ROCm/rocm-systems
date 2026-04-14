// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WaveASM_DIALECT_WAVEASMINTERFACES_H
#define WaveASM_DIALECT_WAVEASMINTERFACES_H

#include "mlir/IR/OpDefinition.h"

// Include generated op interfaces (MFMAOpInterface, etc.)
#include "waveasm/Dialect/WaveASMOpInterfaces.h.inc"

namespace mlir {
namespace OpTrait {

//===----------------------------------------------------------------------===//
// WaveASM Operation Traits
//
// These traits categorize operations for passes like CSE, liveness, etc.
// Using traits instead of string matching provides type safety.
// Defined in mlir::OpTrait namespace as required by NativeOpTrait<>.
//===----------------------------------------------------------------------===//

/// Trait for arithmetic/ALU operations that are pure and CSE-eligible.
/// Includes VALU and SALU operations like v_add_u32, s_add_u32, etc.
template <typename ConcreteType>
class ArithmeticOp : public TraitBase<ConcreteType, ArithmeticOp> {};

// Note: MFMAOp is now an OpInterface (MFMAOpInterface) defined in
// WaveASMInterfaces.td. The interface provides getA(), getB(), getAcc(),
// getDst(), and getTiedOperandIndex() methods for MFMA operations.

/// Trait for non-emitting operations (register aliases, eliminated entirely).
/// These ops do not produce assembly instructions and do not create real
/// inter-instruction delays. Used by hazard mitigation to skip over them.
template <typename ConcreteType>
class NonEmittingOp : public TraitBase<ConcreteType, NonEmittingOp> {};

/// Trait for memory operations (loads/stores).
/// These are NOT eligible for CSE.
template <typename ConcreteType>
class MemoryOp : public TraitBase<ConcreteType, MemoryOp> {};

/// Trait for control flow operations (branches, barriers, waitcnt).
/// These are NOT eligible for CSE.
template <typename ConcreteType>
class ControlFlowOp : public TraitBase<ConcreteType, ControlFlowOp> {};

/// Trait for special register operations (M0, VCC, EXEC).
/// These are NOT eligible for CSE.
template <typename ConcreteType>
class SpecialRegOp : public TraitBase<ConcreteType, SpecialRegOp> {};

/// Trait for operations that implicitly write the SCC (Scalar Condition Code)
/// flag as a hardware side effect.  Ops with this trait must not be placed
/// between an SCC-producing op and its consumer (ConditionOp, s_cselect,
/// s_addc_u32).  The SCC verifier pass checks this invariant.
template <typename ConcreteType>
class SCCDef : public TraitBase<ConcreteType, SCCDef> {};

/// Trait for operations that implicitly read the SCC flag.
/// The result depends on the current SCC value, so these ops are NOT
/// eligible for CSE (two identical operands can produce different results
/// if SCC differs).
template <typename ConcreteType>
class SCCUse : public TraitBase<ConcreteType, SCCUse> {};

} // namespace OpTrait
} // namespace mlir

#endif // WaveASM_DIALECT_WAVEASMINTERFACES_H
