// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WAVEASM_DIALECT_WAVEASMCONTROLFLOWOPS
#define WAVEASM_DIALECT_WAVEASMCONTROLFLOWOPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "waveasm/Dialect/WaveASMDialect.td"
include "waveasm/Dialect/WaveASMTypes.td"

//===----------------------------------------------------------------------===//
// Structured Control Flow Operations
//===----------------------------------------------------------------------===//

def WaveASM_LoopOp : WAVEASMOp<"loop", [
    DeclareOpInterfaceMethods<RegionBranchOpInterface, ["getSuccessorInputs", "getEntrySuccessorOperands", "areTypesCompatible"]>,
    DeclareOpInterfaceMethods<LoopLikeOpInterface, ["getInitsMutable", "getRegionIterArgs", "getYieldedValuesMutable"]>,
    RecursiveMemoryEffects,
    SingleBlockImplicitTerminator<"ConditionOp">
  ]> {
  let summary = "Structured loop with trailing condition (do-while semantics)";
  let description = [{
    Executes the body region at least once, then re-enters if the trailing
    waveasm.condition evaluates to true.  This is a do-while style loop:
    the condition is checked after the body, not before.
    Uses block arguments for loop-carried values (SSA φ-nodes).

    The body region must terminate with a waveasm.condition operation
    that computes the loop condition and yields updated values.

    See test/Dialect/region-control-flow.mlir for round-trip examples
    (simple loops, multi-iter_arg loops, MFMA accumulation, nesting).
  }];

  let arguments = (ins Variadic<WaveASM_AnyReg>:$initArgs);
  let results = (outs Variadic<WaveASM_AnyReg>:$results);
  let regions = (region SizedRegion<1>:$body);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "::mlir::ValueRange":$initArgs)>
  ];

  let extraClassDeclaration = [{
    // Get the loop body block
    ::mlir::Block &getBodyBlock();

    // Get the number of loop-carried values
    unsigned getNumRegionIterArgs();

    // RegionBranchOpInterface will be implemented in .cpp
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def WaveASM_IfOp : WAVEASMOp<"if", [
    DeclareOpInterfaceMethods<RegionBranchOpInterface, ["getSuccessorInputs", "areTypesCompatible"]>,
    RecursiveMemoryEffects,
    SingleBlockImplicitTerminator<"YieldOp">
  ]> {
  let summary = "Conditional execution (if-then-else)";
  let description = [{
    Conditionally executes then or else region based on condition.
    Both regions must terminate with waveasm.yield.

    See test/Dialect/region-control-flow.mlir for round-trip examples
    (simple if, if-then-else, multi-result if, if nested in loop).
  }];

  let arguments = (ins WaveASM_AnySCCOrSGPR:$condition);
  let results = (outs Variadic<WaveASM_AnyReg>:$results);
  let regions = (region SizedRegion<1>:$thenRegion,
                        AnyRegion:$elseRegion);

  let skipDefaultBuilders = 1;
  let builders = [
    OpBuilder<(ins "::mlir::Value":$condition, "bool":$withElseRegion)>,
    OpBuilder<(ins "::mlir::TypeRange":$resultTypes, "::mlir::Value":$condition,
                   "bool":$withElseRegion)>
  ];

  let extraClassDeclaration = [{
    // Get blocks
    ::mlir::Block &getThenBlock();
    ::mlir::Block *getElseBlock();

    // Check if else region exists
    bool hasElse();
  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def WaveASM_ConditionOp : WAVEASMOp<"condition", [
    Terminator,
    HasParent<"LoopOp">,
    DeclareOpInterfaceMethods<RegionBranchTerminatorOpInterface,
      ["getSuccessorRegions", "getMutableSuccessorOperands"]>
  ]> {
  let summary = "Loop condition terminator";
  let description = [{
    Terminates a loop body with a condition check and updated
    loop-carried values. If condition is true, loops back to the start.
    Otherwise, exits the loop with the final values.

    TODO: Model this potentially using side effects.
    IMPORTANT: The condition value must be produced by an s_cmp instruction
    that sets the SCC flag. The assembly emitter emits `s_cbranch_scc1` to
    implement the loop back-edge. No SCC-clobbering instructions (s_add,
    s_and, s_or, etc.) may be inserted between the s_cmp and this operation.
    The waitcnt and hazard mitigation passes must respect this constraint
    (s_waitcnt and s_nop do not clobber SCC and are safe).

    See test/Dialect/region-control-flow.mlir for usage examples.
  }];

  let arguments = (ins WaveASM_AnySCCOrSGPR:$condition,
                       Variadic<WaveASM_AnyReg>:$iterArgs);

  let assemblyFormat = [{
    $condition `:` type($condition)
    (`iter_args` `(` $iterArgs `)` `:` type($iterArgs)^)?
    attr-dict
  }];

  let hasVerifier = 1;
}

def WaveASM_YieldOp : WAVEASMOp<"yield", [
    Pure,
    ReturnLike,
    Terminator,
    WaveASM_NonEmittingOp,
    ParentOneOf<["IfOp", "ProgramOp"]>
  ]> {
  let summary = "Yield values from a region";
  let description = [{
    Yields values from a region (if/then/else) back to the parent operation.

    See test/Dialect/region-control-flow.mlir for usage examples.
  }];

  let arguments = (ins Variadic<WaveASM_AnyReg>:$results);
  let assemblyFormat = "attr-dict ($results^ `:` type($results))?";
}

#endif // WAVEASM_DIALECT_WAVEASMCONTROLFLOWOPS
