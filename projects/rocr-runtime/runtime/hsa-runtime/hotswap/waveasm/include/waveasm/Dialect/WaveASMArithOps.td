// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WAVEASM_DIALECT_WAVEASMARITHOPS
#define WAVEASM_DIALECT_WAVEASMARITHOPS

include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "waveasm/Dialect/WaveASMDialect.td"
include "waveasm/Dialect/WaveASMTypes.td"
include "waveasm/Dialect/WaveASMInterfaces.td"

//===----------------------------------------------------------------------===//
// Generic Arithmetic Pseudo-Ops
//
// These ops are emitted during translation from MLIR/LLVM dialect and are
// lowered to concrete SALU/VALU machine ops by legalization passes. They
// defer register file (SGPR vs VGPR) and type width (i32 vs i64) decisions.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Base Class
//===----------------------------------------------------------------------===//

// Generic arithmetic pseudo-op. Register-file-agnostic: accepts any operand
// type (SGPR, VGPR, immediate) and produces any register type. Legalization
// replaces these with concrete SALU/VALU ops.
class ArithPseudoOp<string mnemonic, list<Trait> traits = []>
    : WAVEASMOp<"arith." # mnemonic, !listconcat([Pure, WaveASM_ArithmeticOp], traits)>;

//===----------------------------------------------------------------------===//
// Binary Arithmetic
//===----------------------------------------------------------------------===//

def WaveASM_ArithAddOp : ArithPseudoOp<"add", [Commutative]> {
  let summary = "Generic integer add (legalized to s_add_u32 or v_add_u32)";
  let arguments = (ins WaveASM_AnyOperand:$lhs, WaveASM_AnyOperand:$rhs);
  let results = (outs WaveASM_AnyReg:$dst);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def WaveASM_ArithMulOp : ArithPseudoOp<"mul", [Commutative]> {
  let summary = "Generic integer multiply (legalized to s_mul_i32 or v_mul_lo_u32)";
  let arguments = (ins WaveASM_AnyOperand:$lhs, WaveASM_AnyOperand:$rhs);
  let results = (outs WaveASM_AnyReg:$dst);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Bitwise
//===----------------------------------------------------------------------===//

def WaveASM_ArithOrOp : ArithPseudoOp<"or", [Commutative]> {
  let summary = "Generic bitwise OR (legalized to s_or_b32/b64 or v_or_b32/b64)";
  let arguments = (ins WaveASM_AnyOperand:$lhs, WaveASM_AnyOperand:$rhs);
  let results = (outs WaveASM_AnyReg:$dst);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

def WaveASM_ArithAndOp : ArithPseudoOp<"and", [Commutative]> {
  let summary = "Generic bitwise AND (legalized to s_and_b32/b64 or v_and_b32/b64)";
  let arguments = (ins WaveASM_AnyOperand:$lhs, WaveASM_AnyOperand:$rhs);
  let results = (outs WaveASM_AnyReg:$dst);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Comparison
//===----------------------------------------------------------------------===//

def WaveASM_ArithCmpOp : ArithPseudoOp<"cmp", []> {
  let summary = "Generic integer compare (legalized to s_cmp_* or v_cmp_*)";
  let description = [{
    Compares two integer operands using the given predicate. Produces a
    condition result that can be consumed by arith.select or control flow.

    Legalization lowers this to:
    - SALU: s_cmp_*_i32/u32 (result is SCC in an SGPR).
    - VALU: v_cmp_*_i32/u32 (sets VCC implicitly, no explicit result).
  }];
  let arguments = (ins WaveASM_CmpPred:$predicate,
                       WaveASM_AnyOperand:$lhs, WaveASM_AnyOperand:$rhs);
  let results = (outs WaveASM_AnyReg:$dst);
  let assemblyFormat = "$predicate `,` $lhs `,` $rhs attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Select
//===----------------------------------------------------------------------===//

def WaveASM_ArithSelectOp : ArithPseudoOp<"select"> {
  let summary = "Generic conditional select (legalized to v_cndmask_b32 or s_cselect_b32)";
  let description = [{
    Selects between two values based on a condition. The condition is
    typically produced by arith.cmp.

    Legalization lowers this to:
    - VALU: v_cndmask_b32 (condition via VCC from preceding v_cmp).
    - SALU: s_cselect_b32 (condition via SCC from preceding s_cmp).
  }];
  let arguments = (ins WaveASM_AnyOperand:$falseVal,
                       WaveASM_AnyOperand:$trueVal,
                       WaveASM_AnyReg:$condition);
  let results = (outs WaveASM_AnyReg:$dst);
  let assemblyFormat = "$condition `,` $trueVal `,` $falseVal attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Type Conversion
//===----------------------------------------------------------------------===//

def WaveASM_ArithReadFirstLaneOp : ArithPseudoOp<"readfirstlane"> {
  let summary = "Generic VGPR to SGPR lane read (legalized to v_readfirstlane_b32)";
  let description = [{
    Reads the value from the first active lane of a VGPR and produces an SGPR.
    For i64, legalization splits into two v_readfirstlane_b32 ops.
    If the source is already an SGPR, legalization elides the op.
  }];
  let arguments = (ins WaveASM_AnyOperand:$src);
  let results = (outs WaveASM_AnyReg:$dst);
  let assemblyFormat = "$src attr-dict `:` functional-type(operands, results)";
}

def WaveASM_ArithTruncOp : ArithPseudoOp<"trunc"> {
  let summary = "Generic integer truncation (e.g., i64 -> i32)";
  let description = [{
    Truncates a wide integer to a narrower width by extracting the low
    sub-register. For i64->i32 on SGPR pairs, extracts the low SGPR.
  }];
  let arguments = (ins WaveASM_AnyOperand:$src);
  let results = (outs WaveASM_AnyReg:$dst);
  let assemblyFormat = "$src attr-dict `:` functional-type(operands, results)";
}

def WaveASM_ArithSExtOp : ArithPseudoOp<"sext"> {
  let summary = "Generic sign extension (e.g., i32 -> i64)";
  let description = [{
    Sign-extends a narrow integer to a wider width. For i32->i64, the
    high word is filled with the sign bit (arithmetic shift right by 31).
  }];
  let arguments = (ins WaveASM_AnyOperand:$src);
  let results = (outs WaveASM_AnyReg:$dst);
  let assemblyFormat = "$src attr-dict `:` functional-type(operands, results)";
}

def WaveASM_ArithZExtOp : ArithPseudoOp<"zext"> {
  let summary = "Generic zero extension (e.g., i32 -> i64)";
  let description = [{
    Zero-extends a narrow integer to a wider width. For i32->i64, the
    high word is set to zero.
  }];
  let arguments = (ins WaveASM_AnyOperand:$src);
  let results = (outs WaveASM_AnyReg:$dst);
  let assemblyFormat = "$src attr-dict `:` functional-type(operands, results)";
}

#endif // WAVEASM_DIALECT_WAVEASMARITHOPS
