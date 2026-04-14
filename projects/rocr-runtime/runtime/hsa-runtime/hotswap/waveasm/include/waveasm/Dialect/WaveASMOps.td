// Copyright 2025 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WaveASM_DIALECT_WAVEASMOPS
#define WaveASM_DIALECT_WAVEASMOPS

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "waveasm/Dialect/WaveASMDialect.td"
include "waveasm/Dialect/WaveASMTypes.td"
include "waveasm/Dialect/WaveASMAttrs.td"
include "waveasm/Dialect/WaveASMInterfaces.td"

//===----------------------------------------------------------------------===//
// Base Classes for WAVEASM Operations
//===----------------------------------------------------------------------===//

// Base class for all WAVEASM operations
class WAVEASMOp<string mnemonic, list<Trait> traits = []>
    : Op<WaveASMDialect, mnemonic, traits>;

// Note: Tied operands (result tied to an input for in-place update) are handled
// via the 'tied_operand' attribute on MFMA ops. Register allocation uses this
// to allocate the result to the same physical register as the tied input.

//===----------------------------------------------------------------------===//
// Base Classes for Instruction Categories
//===----------------------------------------------------------------------===//

// VALU Unary: dst = op(src)
class VALUUnaryOp<string mnemonic, list<Trait> traits = []>
    : WAVEASMOp<mnemonic, !listconcat([Pure, WaveASM_ArithmeticOp], traits)> {
  let arguments = (ins WaveASM_VRegOrImm:$src);
  let results = (outs WaveASM_AnyVGPR:$dst);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($dst)";
}

// VALU Binary: dst = op(src0, src1)
// Note: VALU ops can read from SGPR (broadcast to all lanes) or immediate
class VALUBinaryOp<string mnemonic, list<Trait> traits = []>
    : WAVEASMOp<mnemonic, !listconcat([Pure, WaveASM_ArithmeticOp], traits)> {
  let arguments = (ins WaveASM_VALUSrc:$src0, WaveASM_VALUSrc:$src1);
  let results = (outs WaveASM_AnyVGPR:$dst);
  let assemblyFormat = "$src0 `,` $src1 attr-dict `:` type($src0) `,` type($src1) `->` type($dst)";
}

// VALU Binary Flexible: alias for VALUBinaryOp (both accept VGPR, SGPR, or immediate)
// Kept for backwards compatibility
class VALUBinaryFlexOp<string mnemonic, list<Trait> traits = []>
    : WAVEASMOp<mnemonic, !listconcat([Pure, WaveASM_ArithmeticOp], traits)> {
  let arguments = (ins WaveASM_VALUSrc:$src0, WaveASM_VALUSrc:$src1);
  let results = (outs WaveASM_AnyVGPR:$dst);
  let assemblyFormat = "$src0 `,` $src1 attr-dict `:` type($src0) `,` type($src1) `->` type($dst)";
}

// VALU Binary with carry out: dst, carry = op(src0, src1)
class VALUBinaryCarryOp<string mnemonic, list<Trait> traits = []>
    : WAVEASMOp<mnemonic, !listconcat([Pure, WaveASM_ArithmeticOp], traits)> {
  let arguments = (ins WaveASM_VALUSrc:$src0, WaveASM_VALUSrc:$src1);
  let results = (outs WaveASM_AnyVGPR:$dst, WaveASM_AnySGPR:$carry);
  let assemblyFormat = "$src0 `,` $src1 attr-dict `:` type($src0) `,` type($src1) `->` type($dst) `,` type($carry)";
}

// VALU Ternary: dst = op(src0, src1, src2)
class VALUTernaryOp<string mnemonic, list<Trait> traits = []>
    : WAVEASMOp<mnemonic, !listconcat([Pure, WaveASM_ArithmeticOp], traits)> {
  let arguments = (ins WaveASM_VALUSrc:$src0, WaveASM_VALUSrc:$src1, WaveASM_VALUSrc:$src2);
  let results = (outs WaveASM_AnyVGPR:$dst);
  let assemblyFormat = "$src0 `,` $src1 `,` $src2 attr-dict `:` type($src0) `,` type($src1) `,` type($src2) `->` type($dst)";
}

// VALU Compare: sets vcc, no explicit result
class VALUCmpOp<string mnemonic, list<Trait> traits = []>
    : WAVEASMOp<mnemonic, traits> {
  let arguments = (ins WaveASM_VALUSrc:$src0, WaveASM_VALUSrc:$src1);
  let results = (outs);  // implicitly sets vcc
  let assemblyFormat = "$src0 `,` $src1 attr-dict `:` type($src0) `,` type($src1)";
}

// SALU Unary: dst = op(src)
// Pure -- does NOT clobber SCC on hardware.
// Used for: s_mov_b32, s_mov_b64, s_sext_i32_i8, s_sext_i32_i16.
class SALUUnaryOp<string mnemonic, list<Trait> traits = []>
    : WAVEASMOp<mnemonic, !listconcat([Pure, WaveASM_ArithmeticOp], traits)> {
  let arguments = (ins WaveASM_SRegOrImm:$src);
  let results = (outs WaveASM_AnySGPR:$dst);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($dst)";
}

// SALU Unary with SCC clobber: dst = op(src), SCC = f(result)
// Writes SCC on hardware but result depends only on operands.
class SALUUnaryWithSCCOp<string mnemonic, list<Trait> traits = []>
    : WAVEASMOp<mnemonic, !listconcat([
        Pure, WaveASM_ArithmeticOp, WaveASM_SCCDef
      ], traits)> {
  let arguments = (ins WaveASM_SRegOrImm:$src);
  let results = (outs WaveASM_AnySGPR:$dst, WaveASM_SCC:$scc);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($dst) `,` type($scc)";
}

// SALU Binary: dst = op(src0, src1)
// Pure -- does NOT clobber SCC on hardware.
// Used for: s_mul_i32, s_mul_hi_*, s_bfm_*, s_pack_*.
class SALUBinaryOp<string mnemonic, list<Trait> traits = []>
    : WAVEASMOp<mnemonic, !listconcat([Pure, WaveASM_ArithmeticOp], traits)> {
  let arguments = (ins WaveASM_AnySGPR:$src0, WaveASM_SRegOrImm:$src1);
  let results = (outs WaveASM_AnySGPR:$dst);
  let assemblyFormat = "$src0 `,` $src1 attr-dict `:` type($src0) `,` type($src1) `->` type($dst)";
}

// SALU Binary with SCC clobber: dst = op(src0, src1), SCC = f(result)
// Writes SCC on hardware but result depends only on operands.
class SALUBinaryWithSCCOp<string mnemonic, list<Trait> traits = []>
    : WAVEASMOp<mnemonic, !listconcat([
        Pure, WaveASM_ArithmeticOp, WaveASM_SCCDef
      ], traits)> {
  let arguments = (ins WaveASM_AnySGPR:$src0, WaveASM_SRegOrImm:$src1);
  let results = (outs WaveASM_AnySGPR:$dst, WaveASM_SCC:$scc);
  let assemblyFormat = "$src0 `,` $src1 attr-dict `:` type($src0) `,` type($src1) `->` type($dst) `,` type($scc)";
}

// SALU Binary with carry-out: dst, scc = op(src0, src1)
// Sets SCC as a side effect (carry-out). Used for S_ADD_U32, S_SUB_U32, etc.
class SALUBinaryWithCarryOp<string mnemonic, list<Trait> traits = []>
    : WAVEASMOp<mnemonic, !listconcat([WaveASM_ArithmeticOp, WaveASM_SCCDef], traits)> {
  let arguments = (ins WaveASM_AnySGPR:$src0, WaveASM_SRegOrImm:$src1);
  let results = (outs WaveASM_AnySGPR:$dst, WaveASM_SCC:$scc);
  let assemblyFormat = "$src0 `,` $src1 attr-dict `:` type($src0) `,` type($src1) `->` type($dst) `,` type($scc)";
}

// SALU Binary with carry-in AND carry-out: dst, scc = op(scc_in, src0, src1)
// Reads SCC (carry-in) and writes SCC (carry-out). Used for S_ADDC_U32.
// The explicit $scc_in operand creates a true SSA dependency for the
// carry chain (s_add_u32 -> s_addc_u32), enabling correct CSE behavior.
class SALUBinaryWithCarryInOp<string mnemonic, list<Trait> traits = []>
    : WAVEASMOp<mnemonic, !listconcat([WaveASM_ArithmeticOp, WaveASM_SCCDef, WaveASM_SCCUse], traits)> {
  let arguments = (ins WaveASM_SCC:$scc_in, WaveASM_AnySGPR:$src0, WaveASM_SRegOrImm:$src1);
  let results = (outs WaveASM_AnySGPR:$dst, WaveASM_SCC:$scc);
  let assemblyFormat = "$scc_in `,` $src0 `,` $src1 attr-dict `:` type($scc_in) `,` type($src0) `,` type($src1) `->` type($dst) `,` type($scc)";
}

// SALU Compare: returns condition in SCC.
// On hardware, s_cmp sets the implicit SCC flag with no destination register.
// We model SCC as an explicit SCCType result so that ConditionOp / IfOp can
// reference it as an SSA Value. The assembly emitter skips SCC results and
// operands during emission. CSE-eligible; the SCC spill/reload pass handles
// extended live ranges created by CSE.
class SALUCmpOp<string mnemonic, list<Trait> traits = []>
    : WAVEASMOp<mnemonic, !listconcat([Pure, WaveASM_ArithmeticOp,
        WaveASM_SCCDef], traits)> {
  let arguments = (ins WaveASM_AnySGPR:$src0, WaveASM_SRegOrImm:$src1);
  let results = (outs WaveASM_SCC:$result);  // SCC flag (not a real SGPR)
  let assemblyFormat = "$src0 `,` $src1 attr-dict `:` type($src0) `,` type($src1) `->` type($result)";
}

// MFMA: Matrix multiply-accumulate with tied accumulator
// The result is tied to operand index 2 (the accumulator) when acc is a VGPR.
// MFMA instructions support inline 0 for the accumulator (no tying needed).
class MFMAOp<string mnemonic, int accSize, list<Trait> traits = []>
    : WAVEASMOp<mnemonic, !listconcat([Pure, DeclareOpInterfaceMethods<WaveASM_MFMAOp>], traits)> {
  let arguments = (ins WaveASM_VALUSrc:$a, WaveASM_VALUSrc:$b, WaveASM_MFMAAccSrc:$acc);
  let results = (outs WaveASM_AnyVecReg:$dst);
  let assemblyFormat = "$a `,` $b `,` $acc attr-dict `:` type($a) `,` type($b) `,` type($acc) `->` type($dst)";

  // Tied operand: result 0 is tied to operand 2 (accumulator) when acc is VGPR
  // Register allocator must use the same physical register
  int TiedOperandIndex = 2;

  // Store accumulator size for verification
  int AccumulatorSize = accSize;

  // Custom verifier for accumulator size matching
  let hasVerifier = 1;
}

// VMEM Load: results = load(saddr, voffset, soffset) with optional instruction offset.
// effective_address = SRD_base + voffset + soffset.
class VMEMLoadOp<string mnemonic, int numResults, list<Trait> traits = []>
    : WAVEASMOp<mnemonic, !listconcat([WaveASM_MemoryOp, DeclareOpInterfaceMethods<WaveASM_VMEMLoadOp>], traits)> {
  // voffset can be VGPR or immediate (for constant offsets).
  // soffset is an SGPR or immediate scalar offset (default 0).
  // instOffset is an optional immediate offset added to voffset (emitted as "offset:N").
  let arguments = (ins WaveASM_AnySGPR:$saddr, WaveASM_VRegOrImm:$voffset,
                       WaveASM_SRegOrImm:$soffset,
                       DefaultValuedAttr<I64Attr, "0">:$instOffset);
  let results = (outs Variadic<WaveASM_AnyVGPR>:$results);
  let assemblyFormat = "$saddr `,` $voffset `,` $soffset (`offset` `:` $instOffset^)? attr-dict `:` type($saddr) `,` type($voffset) `,` type($soffset) `->` type($results)";

  int NumResults = numResults;

  // Custom verifier for result count and SRD size
  let hasVerifier = 1;
}

// VMEM Store: store(data, saddr, voffset) with optional instruction offset
class VMEMStoreOp<string mnemonic, list<Trait> traits = []>
    : WAVEASMOp<mnemonic, !listconcat([WaveASM_MemoryOp, DeclareOpInterfaceMethods<WaveASM_VMEMStoreOp>], traits)> {
  // voffset can be VGPR or immediate (for constant offsets)
  // instOffset is an optional immediate offset added to voffset (emitted as "offset:N")
  let arguments = (ins WaveASM_AnyVGPR:$data, WaveASM_AnySGPR:$saddr, WaveASM_VRegOrImm:$voffset,
                       DefaultValuedAttr<I64Attr, "0">:$instOffset);
  let assemblyFormat = "$data `,` $saddr `,` $voffset (`offset` `:` $instOffset^)? attr-dict `:` type($data) `,` type($saddr) `,` type($voffset)";

  // Custom verifier for SRD size validation
  let hasVerifier = 1;
}

// SMEM Load: results = load(saddr, offset)
class SMEMLoadOp<string mnemonic, int numResults, list<Trait> traits = []>
    : WAVEASMOp<mnemonic, !listconcat([WaveASM_MemoryOp], traits)> {
  let arguments = (ins WaveASM_AnySGPR:$saddr, WaveASM_SRegOrImm:$offset);
  let results = (outs Variadic<WaveASM_AnySGPR>:$results);
  let assemblyFormat = "$saddr `,` $offset attr-dict `:` type($saddr) `,` type($offset) `->` type($results)";

  int NumResults = numResults;
}

// LDS Load: results = load(vaddr + offset)
// vaddr can be VGPR or immediate offset
// offset is an optional immediate byte offset (for ds_read offset:N syntax)
class LDSLoadOp<string mnemonic, int numResults, list<Trait> traits = []>
    : WAVEASMOp<mnemonic, !listconcat([WaveASM_MemoryOp, DeclareOpInterfaceMethods<WaveASM_LDSLoadOp>], traits)> {
  let arguments = (ins WaveASM_VRegOrImm:$vaddr,
                       OptionalAttr<I64Attr>:$offset);
  let results = (outs Variadic<WaveASM_AnyVGPR>:$results);
  let assemblyFormat = "$vaddr attr-dict `:` type($vaddr) `->` type($results)";

  int NumResults = numResults;
}

// LDS Store: store(data, vaddr)
// vaddr can be VGPR or immediate offset
class LDSStoreOp<string mnemonic, list<Trait> traits = []>
    : WAVEASMOp<mnemonic, !listconcat([WaveASM_MemoryOp, DeclareOpInterfaceMethods<WaveASM_LDSStoreOp>], traits)> {
  let arguments = (ins WaveASM_AnyVGPR:$data, WaveASM_VRegOrImm:$vaddr);
  let assemblyFormat = "$data `,` $vaddr attr-dict `:` type($data) `,` type($vaddr)";
}

//===----------------------------------------------------------------------===//
// Control Flow Operations
//===----------------------------------------------------------------------===//

include "waveasm/Dialect/WaveASMControlFlowOps.td"
include "waveasm/Dialect/WaveASMArithOps.td"

//===----------------------------------------------------------------------===//
// Program Container
//===----------------------------------------------------------------------===//

def WaveASM_ProgramOp : WAVEASMOp<"program", [
    IsolatedFromAbove,
    SingleBlock,
    NoTerminator,
    Symbol
]> {
  let summary = "Container for a kernel program";
  let description = [{
    A kernel program contains all instructions for a single GPU kernel.
    Uses pure SSA - all values flow through def-use chains.

    Example:
    ```mlir
    waveasm.program @my_kernel
      target = #waveasm.target<#waveasm.gfx942, 5>
      abi = #waveasm.abi<tid = 0, kernarg = 0>
      attributes {vgprs = 64 : i64, sgprs = 32 : i64} {
      %tid = waveasm.precolored.vreg 0 : !waveasm.vreg
      %sum = waveasm.v_add_u32 %tid, %tid : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg
      waveasm.s_endpgm
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    WaveASM_TargetAttr:$target,
    WaveASM_KernelABIAttr:$abi,
    DefaultValuedAttr<I64Attr, "256">:$vgprs,
    DefaultValuedAttr<I64Attr, "104">:$sgprs,
    OptionalAttr<I64ArrayAttr>:$workgroup_size,
    OptionalAttr<I64Attr>:$lds_size,
    // Kernel argument preloading (Code Object v5+, gfx95*)
    // Number of SGPRs to preload from kernarg segment.
    // For pointer args, use 2 SGPRs per pointer (8 bytes each).
    DefaultValuedAttr<I64Attr, "0">:$kernarg_preload_length
  );

  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{
    $sym_name
    `target` `=` $target
    `abi` `=` $abi
    attr-dict-with-keyword
    $body
  }];

  let extraClassDeclaration = [{
    ::mlir::Block &getBodyBlock() { return getBody().front(); }
  }];

  let builders = [
    OpBuilder<(ins
      "::llvm::StringRef":$sym_name,
      "TargetAttr":$target,
      "KernelABIAttr":$abi,
      CArg<"int64_t", "256">:$vgprs,
      CArg<"int64_t", "104">:$sgprs,
      CArg<"::mlir::ArrayAttr", "{}">:$workgroup_size,
      CArg<"::mlir::IntegerAttr", "{}">:$lds_size,
      CArg<"int64_t", "0">:$kernarg_preload_length
    ), [{
      build($_builder, $_state,
            $_builder.getStringAttr(sym_name),
            target, abi,
            $_builder.getI64IntegerAttr(vgprs),
            $_builder.getI64IntegerAttr(sgprs),
            workgroup_size,
            lds_size,
            $_builder.getI64IntegerAttr(kernarg_preload_length));
      // Ensure the body region has a block (region is created by generated builder).
      if ($_state.regions.back()->empty())
        $_state.regions.back()->emplaceBlock();
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// Precolored Register Definitions (ABI Registers)
//===----------------------------------------------------------------------===//

def WaveASM_PrecoloredVRegOp : WAVEASMOp<"precolored.vreg", [Pure, WaveASM_NonEmittingOp]> {
  let summary = "Define a precolored physical VGPR";
  let description = [{
    Defines a VGPR at a specific physical register index.
    Used for ABI registers like v0 (thread ID).

    Example:
    ```mlir
    %tid = waveasm.precolored.vreg 0 : !waveasm.vreg
    %wide = waveasm.precolored.vreg 4, 4 : !waveasm.vreg<4>
    ```
  }];

  let arguments = (ins I64Attr:$index, DefaultValuedAttr<I64Attr, "1">:$size);
  let results = (outs WaveASM_AnyVGPR:$result);
  let assemblyFormat = "$index (`,` $size^)? attr-dict `:` type($result)";
}

def WaveASM_PrecoloredSRegOp : WAVEASMOp<"precolored.sreg", [Pure, WaveASM_NonEmittingOp]> {
  let summary = "Define a precolored physical SGPR";
  let description = [{
    Defines an SGPR at a specific physical register index.
    Used for ABI registers like s[0:1] (kernarg pointer).

    Example:
    ```mlir
    %karg = waveasm.precolored.sreg 0, 2 : !waveasm.sreg<2>
    %wg_id = waveasm.precolored.sreg 2 : !waveasm.sreg
    ```
  }];

  let arguments = (ins I64Attr:$index, DefaultValuedAttr<I64Attr, "1">:$size);
  let results = (outs WaveASM_AnySGPR:$result);
  let assemblyFormat = "$index (`,` $size^)? attr-dict `:` type($result)";
}

def WaveASM_PrecoloredARegOp : WAVEASMOp<"precolored.areg", [Pure, WaveASM_NonEmittingOp]> {
  let summary = "Define a precolored physical AGPR";
  let description = [{
    Defines an AGPR at a specific physical register index.
  }];

  let arguments = (ins I64Attr:$index, DefaultValuedAttr<I64Attr, "1">:$size);
  let results = (outs WaveASM_AnyAGPR:$result);
  let assemblyFormat = "$index (`,` $size^)? attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Immediate Constant
//===----------------------------------------------------------------------===//

def WaveASM_ConstantOp : WAVEASMOp<"constant", [Pure, WaveASM_NonEmittingOp]> {
  let summary = "Create an immediate constant";
  let description = [{
    Creates an immediate constant value.

    Example:
    ```mlir
    %c0 = waveasm.constant 0 : !waveasm.imm<0>
    %c42 = waveasm.constant 42 : !waveasm.imm<42>
    ```
  }];

  let arguments = (ins I64Attr:$value);
  let results = (outs WaveASM_ImmType:$result);
  let assemblyFormat = "$value attr-dict `:` type($result)";
}

//===----------------------------------------------------------------------===//
// Vector Pack/Extract Operations
//===----------------------------------------------------------------------===//

def WaveASM_PackOp : WAVEASMOp<"pack", [Pure, WaveASM_NonEmittingOp]> {
  let summary = "Pack multiple registers into a wider register";
  let description = [{
    Packs multiple registers into a wider register of the same register
    file. Elements are packed lo-to-hi. Supports VGPRs, AGPRs, and SGPRs.

    Example:
    ```mlir
    %vec = waveasm.pack %v0, %v1, %v2, %v3 : (!waveasm.vreg, !waveasm.vreg, !waveasm.vreg, !waveasm.vreg) -> !waveasm.vreg<4>
    %pair = waveasm.pack %s0, %s1 : (!waveasm.sreg, !waveasm.sreg) -> !waveasm.sreg<2, 2>
    ```
  }];

  let arguments = (ins Variadic<WaveASM_AnyReg>:$elements);
  let results = (outs WaveASM_AnyReg:$result);
  let assemblyFormat = "$elements attr-dict `:` functional-type($elements, $result)";
}

def WaveASM_ExtractOp : WAVEASMOp<"extract", [Pure, WaveASM_NonEmittingOp]> {
  let summary = "Extract sub-register(s) from a wider register";
  let description = [{
    Extracts element(s) from a wider register at a given offset.
    Supports VGPRs, AGPRs, and SGPRs. The result register file must
    match the source (VGPR->VGPR, AGPR->AGPR, SGPR->SGPR).

    Example:
    ```mlir
    %v0 = waveasm.extract %vec[0] : !waveasm.vreg<4> -> !waveasm.vreg
    %a0 = waveasm.extract %avec[0] : !waveasm.areg<4> -> !waveasm.areg
    %lo = waveasm.extract %pair[0] : !waveasm.sreg<2, 2> -> !waveasm.sreg
    ```
  }];

  let arguments = (ins WaveASM_AnyReg:$vector, I64Attr:$index);
  let results = (outs WaveASM_AnyReg:$result);
  let assemblyFormat = "$vector `[` $index `]` attr-dict `:` type($vector) `->` type($result)";
  let hasVerifier = 1;
}

def WaveASM_InsertOp : WAVEASMOp<"insert", [Pure, WaveASM_NonEmittingOp]> {
  let summary = "Replace a sub-register in a wider register";
  let description = [{
    Returns a copy of the source register with a single slot replaced.
    This is the dual of ExtractOp. Non-emitting: the register allocator
    assigns the result to the same contiguous block as the source and
    emits a single move for the replaced slot if needed.

    Example:
    ```mlir
    %new_srd = waveasm.insert %word into %srd[2] : !waveasm.sreg, !waveasm.sreg<4, 4> -> !waveasm.sreg<4, 4>
    ```
  }];

  let arguments = (ins WaveASM_AnyReg:$value, WaveASM_AnyReg:$vector, I64Attr:$index);
  let results = (outs WaveASM_AnyReg:$result);
  let assemblyFormat = "$value `into` $vector `[` $index `]` attr-dict `:` type($value) `,` type($vector) `->` type($result)";
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// VALU Unary Instructions
//===----------------------------------------------------------------------===//

// v_mov_b32 supports both VGPR and AGPR destinations. When the destination
// is an AGPR, the assembly emitter produces v_accvgpr_write_b32 instead.
// This enables AGPR zero-init for MFMA accumulator loop init args on gfx950,
// eliminating 128 VGPRs of accumulator zero-init pressure.
def WaveASM_V_MOV_B32 : WAVEASMOp<"v_mov_b32", [Pure, WaveASM_ArithmeticOp]> {
  let arguments = (ins WaveASM_VALUSrc:$src);
  let results = (outs WaveASM_AnyVecReg:$dst);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($dst)";
}
def WaveASM_V_MOV_B64 : VALUUnaryOp<"v_mov_b64">;

// Conversions
def WaveASM_V_CVT_F32_I32 : VALUUnaryOp<"v_cvt_f32_i32">;
def WaveASM_V_CVT_F32_U32 : VALUUnaryOp<"v_cvt_f32_u32">;
def WaveASM_V_CVT_I32_F32 : VALUUnaryOp<"v_cvt_i32_f32">;
def WaveASM_V_CVT_U32_F32 : VALUUnaryOp<"v_cvt_u32_f32">;
def WaveASM_V_CVT_F16_F32 : VALUUnaryOp<"v_cvt_f16_f32">;
def WaveASM_V_CVT_F32_F16 : VALUUnaryOp<"v_cvt_f32_f16">;
def WaveASM_V_CVT_F32_F64 : VALUUnaryOp<"v_cvt_f32_f64">;
def WaveASM_V_CVT_F64_F32 : VALUUnaryOp<"v_cvt_f64_f32">;
def WaveASM_V_CVT_F32_BF16 : VALUUnaryOp<"v_cvt_f32_bf16">;
def WaveASM_V_CVT_BF16_F32 : VALUUnaryOp<"v_cvt_bf16_f32">;
def WaveASM_V_CVT_PK_BF16_F32 : VALUBinaryOp<"v_cvt_pk_bf16_f32">;

// Transcendentals
def WaveASM_V_RCP_F32 : VALUUnaryOp<"v_rcp_f32">;
def WaveASM_V_RCP_F64 : VALUUnaryOp<"v_rcp_f64">;
def WaveASM_V_RSQ_F32 : VALUUnaryOp<"v_rsq_f32">;
def WaveASM_V_RSQ_F64 : VALUUnaryOp<"v_rsq_f64">;
def WaveASM_V_SQRT_F32 : VALUUnaryOp<"v_sqrt_f32">;
def WaveASM_V_SQRT_F64 : VALUUnaryOp<"v_sqrt_f64">;
def WaveASM_V_EXP_F32 : VALUUnaryOp<"v_exp_f32">;
def WaveASM_V_LOG_F32 : VALUUnaryOp<"v_log_f32">;
def WaveASM_V_SIN_F32 : VALUUnaryOp<"v_sin_f32">;
def WaveASM_V_COS_F32 : VALUUnaryOp<"v_cos_f32">;

// Lane operations (VGPR -> SGPR)
def WaveASM_V_READFIRSTLANE_B32 : WAVEASMOp<"v_readfirstlane_b32", [Pure]> {
  let summary = "Read first lane of VGPR to SGPR";
  let description = [{
    Reads the value from the first active lane of a VGPR and places it in an SGPR.
    This is useful for extracting scalar values from vector registers.

    Note: On gfx940+ architectures, there is a VALU → v_readfirstlane hazard.
    If a VALU instruction writes a VGPR that is immediately read by v_readfirstlane,
    an s_nop must be inserted between them.
  }];
  let arguments = (ins WaveASM_AnyVGPR:$src);
  let results = (outs WaveASM_AnySGPR:$dst);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($dst)";
}

// Bit operations
def WaveASM_V_NOT_B32 : VALUUnaryOp<"v_not_b32">;
def WaveASM_V_NOT_B64 : VALUUnaryOp<"v_not_b64">;
def WaveASM_V_BFREV_B32 : VALUUnaryOp<"v_bfrev_b32">;
def WaveASM_V_FFBH_U32 : VALUUnaryOp<"v_ffbh_u32">;
def WaveASM_V_FFBL_B32 : VALUUnaryOp<"v_ffbl_b32">;
def WaveASM_V_FFBH_I32 : VALUUnaryOp<"v_ffbh_i32">;
def WaveASM_V_FREXP_MANT_F32 : VALUUnaryOp<"v_frexp_mant_f32">;
def WaveASM_V_FREXP_EXP_I32_F32 : VALUUnaryOp<"v_frexp_exp_i32_f32">;

// Floating point
def WaveASM_V_FRACT_F32 : VALUUnaryOp<"v_fract_f32">;
def WaveASM_V_TRUNC_F32 : VALUUnaryOp<"v_trunc_f32">;
def WaveASM_V_CEIL_F32 : VALUUnaryOp<"v_ceil_f32">;
def WaveASM_V_FLOOR_F32 : VALUUnaryOp<"v_floor_f32">;
def WaveASM_V_RNDNE_F32 : VALUUnaryOp<"v_rndne_f32">;

// Absolute value / negate
def WaveASM_V_ABS_F32 : VALUUnaryOp<"v_abs_f32">;
def WaveASM_V_ABS_I32 : VALUUnaryOp<"v_abs_i32">;

//===----------------------------------------------------------------------===//
// VALU Binary Instructions
//===----------------------------------------------------------------------===//

// Integer arithmetic (commutative: add, mul)
def WaveASM_V_ADD_U32 : VALUBinaryOp<"v_add_u32", [Commutative]>;
def WaveASM_V_ADD_I32 : VALUBinaryOp<"v_add_i32", [Commutative]>;
def WaveASM_V_SUB_U32 : VALUBinaryOp<"v_sub_u32">;
def WaveASM_V_SUB_I32 : VALUBinaryOp<"v_sub_i32">;
def WaveASM_V_SUBREV_U32 : VALUBinaryOp<"v_subrev_u32">;
def WaveASM_V_MUL_LO_U32 : VALUBinaryOp<"v_mul_lo_u32", [Commutative]>;
def WaveASM_V_MUL_HI_U32 : VALUBinaryOp<"v_mul_hi_u32", [Commutative]>;
def WaveASM_V_MUL_LO_I32 : VALUBinaryOp<"v_mul_lo_i32", [Commutative]>;
def WaveASM_V_MUL_HI_I32 : VALUBinaryOp<"v_mul_hi_i32", [Commutative]>;

// Floating point arithmetic
def WaveASM_V_ADD_F32 : VALUBinaryOp<"v_add_f32">;
def WaveASM_V_ADD_F64 : VALUBinaryOp<"v_add_f64">;
def WaveASM_V_ADD_F16 : VALUBinaryOp<"v_add_f16">;
def WaveASM_V_SUB_F32 : VALUBinaryOp<"v_sub_f32">;
def WaveASM_V_SUB_F64 : VALUBinaryOp<"v_sub_f64">;
def WaveASM_V_SUB_F16 : VALUBinaryOp<"v_sub_f16">;
def WaveASM_V_SUBREV_F32 : VALUBinaryOp<"v_subrev_f32">;
def WaveASM_V_MUL_F32 : VALUBinaryOp<"v_mul_f32">;
def WaveASM_V_MUL_F64 : VALUBinaryOp<"v_mul_f64">;
def WaveASM_V_MUL_F16 : VALUBinaryOp<"v_mul_f16">;
def WaveASM_V_MUL_LEGACY_F32 : VALUBinaryOp<"v_mul_legacy_f32">;

// Min/Max
def WaveASM_V_MIN_F32 : VALUBinaryOp<"v_min_f32">;
def WaveASM_V_MIN_F64 : VALUBinaryOp<"v_min_f64">;
def WaveASM_V_MIN_I32 : VALUBinaryOp<"v_min_i32">;
def WaveASM_V_MIN_U32 : VALUBinaryOp<"v_min_u32">;
def WaveASM_V_MAX_F32 : VALUBinaryOp<"v_max_f32">;
def WaveASM_V_MAX_F64 : VALUBinaryOp<"v_max_f64">;
def WaveASM_V_MAX_I32 : VALUBinaryOp<"v_max_i32">;
def WaveASM_V_MAX_U32 : VALUBinaryOp<"v_max_u32">;

// Bitwise (commutative: and, or, xor)
def WaveASM_V_AND_B32 : VALUBinaryOp<"v_and_b32", [Commutative]>;
def WaveASM_V_AND_B64 : VALUBinaryOp<"v_and_b64", [Commutative]>;
def WaveASM_V_OR_B32 : VALUBinaryOp<"v_or_b32", [Commutative]>;
def WaveASM_V_OR_B64 : VALUBinaryOp<"v_or_b64", [Commutative]>;
def WaveASM_V_XOR_B32 : VALUBinaryOp<"v_xor_b32", [Commutative]>;
def WaveASM_V_XOR_B64 : VALUBinaryOp<"v_xor_b64", [Commutative]>;

// Shifts
def WaveASM_V_LSHLREV_B32 : VALUBinaryOp<"v_lshlrev_b32">;
def WaveASM_V_LSHLREV_B64 : VALUBinaryOp<"v_lshlrev_b64">;
def WaveASM_V_LSHRREV_B32 : VALUBinaryOp<"v_lshrrev_b32">;
def WaveASM_V_LSHRREV_B64 : VALUBinaryOp<"v_lshrrev_b64">;
def WaveASM_V_ASHRREV_I32 : VALUBinaryOp<"v_ashrrev_i32">;
def WaveASM_V_ASHRREV_I64 : VALUBinaryOp<"v_ashrrev_i64">;

// Misc
def WaveASM_V_LDEXP_F32 : VALUBinaryOp<"v_ldexp_f32">;
def WaveASM_V_LDEXP_F64 : VALUBinaryOp<"v_ldexp_f64">;
def WaveASM_V_CVT_PKRTZ_F16_F32 : VALUBinaryOp<"v_cvt_pkrtz_f16_f32">;

// Thread ID computation (use FlexOp since src0 can be an inline constant)
def WaveASM_V_MBCNT_LO_U32_B32 : VALUBinaryFlexOp<"v_mbcnt_lo_u32_b32">;
def WaveASM_V_MBCNT_HI_U32_B32 : VALUBinaryFlexOp<"v_mbcnt_hi_u32_b32">;

//===----------------------------------------------------------------------===//
// VALU Binary with Carry
//===----------------------------------------------------------------------===//

def WaveASM_V_ADD_CO_U32 : VALUBinaryCarryOp<"v_add_co_u32">;
def WaveASM_V_SUB_CO_U32 : VALUBinaryCarryOp<"v_sub_co_u32">;
def WaveASM_V_ADDC_CO_U32 : VALUBinaryCarryOp<"v_addc_co_u32">;
def WaveASM_V_SUBB_CO_U32 : VALUBinaryCarryOp<"v_subb_co_u32">;

//===----------------------------------------------------------------------===//
// VALU Ternary Instructions
//===----------------------------------------------------------------------===//

def WaveASM_V_FMA_F32 : VALUTernaryOp<"v_fma_f32">;
def WaveASM_V_FMA_F64 : VALUTernaryOp<"v_fma_f64">;
def WaveASM_V_FMA_F16 : VALUTernaryOp<"v_fma_f16">;
def WaveASM_V_MAD_F32 : VALUTernaryOp<"v_mad_f32">;
def WaveASM_V_MAD_I32_I24 : VALUTernaryOp<"v_mad_i32_i24">;
def WaveASM_V_MAD_U32_U24 : VALUTernaryOp<"v_mad_u32_u24">;
def WaveASM_V_ADD3_U32 : VALUTernaryOp<"v_add3_u32">;
def WaveASM_V_MED3_F32 : VALUTernaryOp<"v_med3_f32">;
def WaveASM_V_MED3_I32 : VALUTernaryOp<"v_med3_i32">;
def WaveASM_V_MED3_U32 : VALUTernaryOp<"v_med3_u32">;
def WaveASM_V_SAD_U32 : VALUTernaryOp<"v_sad_u32">;
def WaveASM_V_LERP_U8 : VALUTernaryOp<"v_lerp_u8">;
def WaveASM_V_BFE_U32 : VALUTernaryOp<"v_bfe_u32">;
def WaveASM_V_BFE_I32 : VALUTernaryOp<"v_bfe_i32">;
def WaveASM_V_BFI_B32 : VALUTernaryOp<"v_bfi_b32">;
def WaveASM_V_ALIGNBIT_B32 : VALUTernaryOp<"v_alignbit_b32">;
def WaveASM_V_ALIGNBYTE_B32 : VALUTernaryOp<"v_alignbyte_b32">;

// Fused shift-or operation: dst = (src << shift) | orend
def WaveASM_V_LSHL_OR_B32 : VALUTernaryOp<"v_lshl_or_b32">;

// Fused shift-add operation: dst = (src << shift) + addend
def WaveASM_V_LSHL_ADD_U32 : VALUTernaryOp<"v_lshl_add_u32">;

// Conditional mask and lane operations.
//
// V_CNDMASK_B32 selects per-lane between src0 and src1 based on VCC.
// The hardware instruction reads VCC implicitly (VOP2 encoding), so
// the assembly emitter drops the vcc_dep operand and always emits
// "vcc" as the condition source.
//
// vcc_dep exists solely to create a def-use edge to the preceding
// v_cmp that wrote VCC.  Without it, no SSA dependency prevents
// reordering v_cndmask before the comparison.
//
// Must NOT carry Pure or ArithmeticOp -- CSE would merge two
// v_cndmask ops that share explicit operands but read different VCC
// values from different comparisons.
def WaveASM_V_CNDMASK_B32 : WAVEASMOp<"v_cndmask_b32", []> {
  let summary = "Conditional select per lane via implicit VCC read.";
  let arguments = (ins
    WaveASM_VALUSrc:$src0,
    WaveASM_VALUSrc:$src1,
    WaveASM_VALUSrc:$vcc_dep
  );
  let results = (outs WaveASM_AnyVGPR:$dst);
  let assemblyFormat = [{
    $src0 `,` $src1 `,` $vcc_dep attr-dict `:`
    type($src0) `,` type($src1) `,` type($vcc_dep) `->` type($dst)
  }];
}

// Lane read operations (VGPR -> SGPR)
def WaveASM_V_READLANE_B32 : WAVEASMOp<"v_readlane_b32", [Pure]> {
  let summary = "Read value from VGPR lane to SGPR";
  let arguments = (ins WaveASM_AnyVGPR:$src, WaveASM_SRegOrImm:$lane);
  let results = (outs WaveASM_AnySGPR:$dst);
  let assemblyFormat = "$src `,` $lane attr-dict `:` type($src) `,` type($lane) `->` type($dst)";
}

// Write lane operation (SGPR -> VGPR lane)
def WaveASM_V_WRITELANE_B32 : WAVEASMOp<"v_writelane_b32", [Pure]> {
  let summary = "Write SGPR value to VGPR lane";
  let arguments = (ins WaveASM_SRegOrImm:$src, WaveASM_SRegOrImm:$lane, WaveASM_AnyVGPR:$vdst_in);
  let results = (outs WaveASM_AnyVGPR:$dst);
  let assemblyFormat = "$src `,` $lane `,` $vdst_in attr-dict `:` type($src) `,` type($lane) `,` type($vdst_in) `->` type($dst)";
}

// VGPR/AGPR cross-bank moves
def WaveASM_V_ACCVGPR_READ_B32 : WAVEASMOp<"v_accvgpr_read_b32", [Pure]> {
  let summary = "Read AGPR lane into VGPR";
  let arguments = (ins WaveASM_AnyAGPR:$src);
  let results = (outs WaveASM_AnyVGPR:$dst);
  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($dst)";
}

def WaveASM_V_ACCVGPR_WRITE_B32 : WAVEASMOp<"v_accvgpr_write_b32", [Pure]> {
  let summary = "Write VGPR lane into AGPR";
  let arguments = (ins WaveASM_AnyVGPR:$src, WaveASM_AnyAGPR:$adst);
  let results = (outs);
  let assemblyFormat = "$src `,` $adst attr-dict `:` type($src) `,` type($adst)";
}

//===----------------------------------------------------------------------===//
// VALU Compare Instructions
//===----------------------------------------------------------------------===//

// Float compares
def WaveASM_V_CMP_EQ_F32 : VALUCmpOp<"v_cmp_eq_f32">;
def WaveASM_V_CMP_NE_F32 : VALUCmpOp<"v_cmp_ne_f32">;
def WaveASM_V_CMP_LT_F32 : VALUCmpOp<"v_cmp_lt_f32">;
def WaveASM_V_CMP_LE_F32 : VALUCmpOp<"v_cmp_le_f32">;
def WaveASM_V_CMP_GT_F32 : VALUCmpOp<"v_cmp_gt_f32">;
def WaveASM_V_CMP_GE_F32 : VALUCmpOp<"v_cmp_ge_f32">;
def WaveASM_V_CMP_O_F32 : VALUCmpOp<"v_cmp_o_f32">;
def WaveASM_V_CMP_U_F32 : VALUCmpOp<"v_cmp_u_f32">;

// Integer compares
def WaveASM_V_CMP_EQ_I32 : VALUCmpOp<"v_cmp_eq_i32">;
def WaveASM_V_CMP_NE_I32 : VALUCmpOp<"v_cmp_ne_i32">;
def WaveASM_V_CMP_LT_I32 : VALUCmpOp<"v_cmp_lt_i32">;
def WaveASM_V_CMP_LE_I32 : VALUCmpOp<"v_cmp_le_i32">;
def WaveASM_V_CMP_GT_I32 : VALUCmpOp<"v_cmp_gt_i32">;
def WaveASM_V_CMP_GE_I32 : VALUCmpOp<"v_cmp_ge_i32">;
def WaveASM_V_CMP_EQ_U32 : VALUCmpOp<"v_cmp_eq_u32">;
def WaveASM_V_CMP_NE_U32 : VALUCmpOp<"v_cmp_ne_u32">;
def WaveASM_V_CMP_LT_U32 : VALUCmpOp<"v_cmp_lt_u32">;
def WaveASM_V_CMP_LE_U32 : VALUCmpOp<"v_cmp_le_u32">;
def WaveASM_V_CMP_GT_U32 : VALUCmpOp<"v_cmp_gt_u32">;
def WaveASM_V_CMP_GE_U32 : VALUCmpOp<"v_cmp_ge_u32">;

// 64-bit compares
def WaveASM_V_CMP_EQ_I64 : VALUCmpOp<"v_cmp_eq_i64">;
def WaveASM_V_CMP_NE_I64 : VALUCmpOp<"v_cmp_ne_i64">;
def WaveASM_V_CMP_LT_I64 : VALUCmpOp<"v_cmp_lt_i64">;
def WaveASM_V_CMP_LE_I64 : VALUCmpOp<"v_cmp_le_i64">;
def WaveASM_V_CMP_GT_I64 : VALUCmpOp<"v_cmp_gt_i64">;
def WaveASM_V_CMP_GE_I64 : VALUCmpOp<"v_cmp_ge_i64">;
def WaveASM_V_CMP_EQ_U64 : VALUCmpOp<"v_cmp_eq_u64">;
def WaveASM_V_CMP_NE_U64 : VALUCmpOp<"v_cmp_ne_u64">;
def WaveASM_V_CMP_LT_U64 : VALUCmpOp<"v_cmp_lt_u64">;
def WaveASM_V_CMP_LE_U64 : VALUCmpOp<"v_cmp_le_u64">;
def WaveASM_V_CMP_GT_U64 : VALUCmpOp<"v_cmp_gt_u64">;
def WaveASM_V_CMP_GE_U64 : VALUCmpOp<"v_cmp_ge_u64">;

//===----------------------------------------------------------------------===//
// SALU Unary Instructions
//===----------------------------------------------------------------------===//

// No SCC clobber
def WaveASM_S_MOV_B32 : SALUUnaryOp<"s_mov_b32">;
def WaveASM_S_MOV_B64 : SALUUnaryOp<"s_mov_b64">;
def WaveASM_S_SEXT_I32_I8 : SALUUnaryOp<"s_sext_i32_i8">;
def WaveASM_S_SEXT_I32_I16 : SALUUnaryOp<"s_sext_i32_i16">;

// Clobber SCC (SCC = f(result))
def WaveASM_S_NOT_B32 : SALUUnaryWithSCCOp<"s_not_b32">;
def WaveASM_S_NOT_B64 : SALUUnaryWithSCCOp<"s_not_b64">;
def WaveASM_S_BREV_B32 : SALUUnaryWithSCCOp<"s_brev_b32">;
def WaveASM_S_BREV_B64 : SALUUnaryWithSCCOp<"s_brev_b64">;
def WaveASM_S_BCNT0_I32_B32 : SALUUnaryWithSCCOp<"s_bcnt0_i32_b32">;
def WaveASM_S_BCNT0_I32_B64 : SALUUnaryWithSCCOp<"s_bcnt0_i32_b64">;
def WaveASM_S_BCNT1_I32_B32 : SALUUnaryWithSCCOp<"s_bcnt1_i32_b32">;
def WaveASM_S_BCNT1_I32_B64 : SALUUnaryWithSCCOp<"s_bcnt1_i32_b64">;
def WaveASM_S_FF0_I32_B32 : SALUUnaryWithSCCOp<"s_ff0_i32_b32">;
def WaveASM_S_FF0_I32_B64 : SALUUnaryWithSCCOp<"s_ff0_i32_b64">;
def WaveASM_S_FF1_I32_B32 : SALUUnaryWithSCCOp<"s_ff1_i32_b32">;
def WaveASM_S_FF1_I32_B64 : SALUUnaryWithSCCOp<"s_ff1_i32_b64">;
def WaveASM_S_FLBIT_I32_B32 : SALUUnaryWithSCCOp<"s_flbit_i32_b32">;
def WaveASM_S_FLBIT_I32_B64 : SALUUnaryWithSCCOp<"s_flbit_i32_b64">;
def WaveASM_S_ABS_I32 : SALUUnaryWithSCCOp<"s_abs_i32">;

//===----------------------------------------------------------------------===//
// SALU Binary Instructions
//===----------------------------------------------------------------------===//

// Arithmetic with carry (sets SCC)
def WaveASM_S_ADD_U32 : SALUBinaryWithCarryOp<"s_add_u32">;
def WaveASM_S_ADDC_U32 : SALUBinaryWithCarryInOp<"s_addc_u32">;
def WaveASM_S_ADD_I32 : SALUBinaryWithCarryOp<"s_add_i32">;
def WaveASM_S_SUB_U32 : SALUBinaryWithCarryOp<"s_sub_u32">;
def WaveASM_S_SUB_I32 : SALUBinaryWithCarryOp<"s_sub_i32">;
// Conditional select: dst = SCC ? src0 : src1 (reads SCC, no SCC write).
// Takes an explicit SCC operand for SSA tracking of the SCC dependency.
// NOT CSE-eligible: result depends on SCC state.  NoMemoryEffect allows DCE.
def WaveASM_S_CSELECT_B32 : WAVEASMOp<"s_cselect_b32",
    [WaveASM_SCCUse, NoMemoryEffect]> {
  let arguments = (ins WaveASM_SCC:$scc_in, WaveASM_SRegOrImm:$src0, WaveASM_SRegOrImm:$src1);
  let results = (outs WaveASM_AnySGPR:$dst);
  let assemblyFormat = "$scc_in `,` $src0 `,` $src1 attr-dict `:` type($scc_in) `,` type($src0) `,` type($src1) `->` type($dst)";
}
// Multiplication does not set SCC
def WaveASM_S_MUL_I32 : SALUBinaryOp<"s_mul_i32">;
def WaveASM_S_MUL_HI_U32 : SALUBinaryOp<"s_mul_hi_u32">;
def WaveASM_S_MUL_HI_I32 : SALUBinaryOp<"s_mul_hi_i32">;

// Bitwise -- all clobber SCC (SCC = result != 0).
def WaveASM_S_AND_B32 : SALUBinaryWithSCCOp<"s_and_b32">;
def WaveASM_S_AND_B64 : SALUBinaryWithSCCOp<"s_and_b64">;
def WaveASM_S_OR_B32 : SALUBinaryWithSCCOp<"s_or_b32">;
def WaveASM_S_OR_B64 : SALUBinaryWithSCCOp<"s_or_b64">;
def WaveASM_S_XOR_B32 : SALUBinaryWithSCCOp<"s_xor_b32">;
def WaveASM_S_XOR_B64 : SALUBinaryWithSCCOp<"s_xor_b64">;
def WaveASM_S_ANDN2_B32 : SALUBinaryWithSCCOp<"s_andn2_b32">;
def WaveASM_S_ANDN2_B64 : SALUBinaryWithSCCOp<"s_andn2_b64">;
def WaveASM_S_ORN2_B32 : SALUBinaryWithSCCOp<"s_orn2_b32">;
def WaveASM_S_ORN2_B64 : SALUBinaryWithSCCOp<"s_orn2_b64">;
def WaveASM_S_NAND_B32 : SALUBinaryWithSCCOp<"s_nand_b32">;
def WaveASM_S_NAND_B64 : SALUBinaryWithSCCOp<"s_nand_b64">;
def WaveASM_S_NOR_B32 : SALUBinaryWithSCCOp<"s_nor_b32">;
def WaveASM_S_NOR_B64 : SALUBinaryWithSCCOp<"s_nor_b64">;
def WaveASM_S_XNOR_B32 : SALUBinaryWithSCCOp<"s_xnor_b32">;
def WaveASM_S_XNOR_B64 : SALUBinaryWithSCCOp<"s_xnor_b64">;

// Shifts
def WaveASM_S_LSHL_B32 : SALUBinaryWithSCCOp<"s_lshl_b32">;
def WaveASM_S_LSHL_B64 : SALUBinaryWithSCCOp<"s_lshl_b64">;
def WaveASM_S_LSHR_B32 : SALUBinaryWithSCCOp<"s_lshr_b32">;
def WaveASM_S_LSHR_B64 : SALUBinaryWithSCCOp<"s_lshr_b64">;
def WaveASM_S_ASHR_I32 : SALUBinaryWithSCCOp<"s_ashr_i32">;
def WaveASM_S_ASHR_I64 : SALUBinaryWithSCCOp<"s_ashr_i64">;

// Min/Max
def WaveASM_S_MIN_I32 : SALUBinaryWithSCCOp<"s_min_i32">;
def WaveASM_S_MIN_U32 : SALUBinaryWithSCCOp<"s_min_u32">;
def WaveASM_S_MAX_I32 : SALUBinaryWithSCCOp<"s_max_i32">;
def WaveASM_S_MAX_U32 : SALUBinaryWithSCCOp<"s_max_u32">;

// Misc
def WaveASM_S_BFE_U32 : SALUBinaryWithSCCOp<"s_bfe_u32">;
def WaveASM_S_BFE_I32 : SALUBinaryWithSCCOp<"s_bfe_i32">;
def WaveASM_S_BFE_U64 : SALUBinaryWithSCCOp<"s_bfe_u64">;
def WaveASM_S_BFE_I64 : SALUBinaryWithSCCOp<"s_bfe_i64">;
def WaveASM_S_BFM_B32 : SALUBinaryOp<"s_bfm_b32">;
def WaveASM_S_BFM_B64 : SALUBinaryOp<"s_bfm_b64">;
def WaveASM_S_PACK_LL_B32_B16 : SALUBinaryOp<"s_pack_ll_b32_b16">;
def WaveASM_S_PACK_LH_B32_B16 : SALUBinaryOp<"s_pack_lh_b32_b16">;
def WaveASM_S_PACK_HH_B32_B16 : SALUBinaryOp<"s_pack_hh_b32_b16">;

//===----------------------------------------------------------------------===//
// SALU Compare Instructions
//===----------------------------------------------------------------------===//

def WaveASM_S_CMP_EQ_I32 : SALUCmpOp<"s_cmp_eq_i32">;
def WaveASM_S_CMP_NE_I32 : SALUCmpOp<"s_cmp_ne_i32">;
def WaveASM_S_CMP_LT_I32 : SALUCmpOp<"s_cmp_lt_i32">;
def WaveASM_S_CMP_LE_I32 : SALUCmpOp<"s_cmp_le_i32">;
def WaveASM_S_CMP_GT_I32 : SALUCmpOp<"s_cmp_gt_i32">;
def WaveASM_S_CMP_GE_I32 : SALUCmpOp<"s_cmp_ge_i32">;
def WaveASM_S_CMP_EQ_U32 : SALUCmpOp<"s_cmp_eq_u32">;
def WaveASM_S_CMP_NE_U32 : SALUCmpOp<"s_cmp_ne_u32">;
def WaveASM_S_CMP_LT_U32 : SALUCmpOp<"s_cmp_lt_u32">;
def WaveASM_S_CMP_LE_U32 : SALUCmpOp<"s_cmp_le_u32">;
def WaveASM_S_CMP_GT_U32 : SALUCmpOp<"s_cmp_gt_u32">;
def WaveASM_S_CMP_GE_U32 : SALUCmpOp<"s_cmp_ge_u32">;
def WaveASM_S_CMP_EQ_U64 : SALUCmpOp<"s_cmp_eq_u64">;
def WaveASM_S_CMP_NE_U64 : SALUCmpOp<"s_cmp_ne_u64">;

//===----------------------------------------------------------------------===//
// MFMA Instructions (Matrix Fused Multiply-Add)
// All have TiedResult<0, 2> - result tied to accumulator input
//===----------------------------------------------------------------------===//

// F32 output, F16 input
def WaveASM_V_MFMA_F32_16X16X16_F16 : MFMAOp<"v_mfma_f32_16x16x16_f16", 4>;
def WaveASM_V_MFMA_F32_16X16X32_F16 : MFMAOp<"v_mfma_f32_16x16x32_f16", 4>;  // gfx950+
def WaveASM_V_MFMA_F32_32X32X8_F16 : MFMAOp<"v_mfma_f32_32x32x8_f16", 16>;
def WaveASM_V_MFMA_F32_16X16X4_F16 : MFMAOp<"v_mfma_f32_16x16x4_f16", 16>;
def WaveASM_V_MFMA_F32_32X32X4_F16 : MFMAOp<"v_mfma_f32_32x32x4_f16", 32>;
def WaveASM_V_MFMA_F32_4X4X4_F16 : MFMAOp<"v_mfma_f32_4x4x4_f16", 4>;

// F32 output, BF16 input
def WaveASM_V_MFMA_F32_16X16X16_BF16 : MFMAOp<"v_mfma_f32_16x16x16_bf16", 4>;
def WaveASM_V_MFMA_F32_16X16X32_BF16 : MFMAOp<"v_mfma_f32_16x16x32_bf16", 4>;  // gfx950+
def WaveASM_V_MFMA_F32_32X32X8_BF16 : MFMAOp<"v_mfma_f32_32x32x8_bf16", 16>;
def WaveASM_V_MFMA_F32_16X16X4_BF16 : MFMAOp<"v_mfma_f32_16x16x4_bf16", 16>;
def WaveASM_V_MFMA_F32_32X32X4_BF16 : MFMAOp<"v_mfma_f32_32x32x4_bf16", 32>;
def WaveASM_V_MFMA_F32_4X4X4_BF16 : MFMAOp<"v_mfma_f32_4x4x4_bf16", 4>;

// I32 output, I8 input
def WaveASM_V_MFMA_I32_16X16X16_I8 : MFMAOp<"v_mfma_i32_16x16x16_i8", 4>;
def WaveASM_V_MFMA_I32_32X32X8_I8 : MFMAOp<"v_mfma_i32_32x32x8_i8", 16>;
def WaveASM_V_MFMA_I32_16X16X4_I8 : MFMAOp<"v_mfma_i32_16x16x4_i8", 16>;
def WaveASM_V_MFMA_I32_32X32X4_I8 : MFMAOp<"v_mfma_i32_32x32x4_i8", 32>;
def WaveASM_V_MFMA_I32_4X4X4_I8 : MFMAOp<"v_mfma_i32_4x4x4_i8", 4>;

// F32 output, F32 input
def WaveASM_V_MFMA_F32_16X16X4_F32 : MFMAOp<"v_mfma_f32_16x16x4_f32", 4>;
def WaveASM_V_MFMA_F32_32X32X2_F32 : MFMAOp<"v_mfma_f32_32x32x2_f32", 16>;
def WaveASM_V_MFMA_F32_4X4X1_F32 : MFMAOp<"v_mfma_f32_4x4x1_f32", 4>;

// F64 output, F64 input
def WaveASM_V_MFMA_F64_16X16X4_F64 : MFMAOp<"v_mfma_f64_16x16x4_f64", 8>;
def WaveASM_V_MFMA_F64_4X4X4_F64 : MFMAOp<"v_mfma_f64_4x4x4_f64", 2>;

// FP8 variants (gfx942+)
def WaveASM_V_MFMA_F32_16X16X32_FP8_FP8 : MFMAOp<"v_mfma_f32_16x16x32_fp8_fp8", 4>;
def WaveASM_V_MFMA_F32_32X32X16_FP8_FP8 : MFMAOp<"v_mfma_f32_32x32x16_fp8_fp8", 16>;
def WaveASM_V_MFMA_F32_16X16X32_FP8_BF8 : MFMAOp<"v_mfma_f32_16x16x32_fp8_bf8", 4>;
def WaveASM_V_MFMA_F32_32X32X16_FP8_BF8 : MFMAOp<"v_mfma_f32_32x32x16_fp8_bf8", 16>;
def WaveASM_V_MFMA_F32_16X16X32_BF8_FP8 : MFMAOp<"v_mfma_f32_16x16x32_bf8_fp8", 4>;
def WaveASM_V_MFMA_F32_32X32X16_BF8_FP8 : MFMAOp<"v_mfma_f32_32x32x16_bf8_fp8", 16>;
def WaveASM_V_MFMA_F32_16X16X32_BF8_BF8 : MFMAOp<"v_mfma_f32_16x16x32_bf8_bf8", 4>;
def WaveASM_V_MFMA_F32_32X32X16_BF8_BF8 : MFMAOp<"v_mfma_f32_32x32x16_bf8_bf8", 16>;

// Scaled MFMA: Matrix multiply with per-group scale factors (gfx950+)
// These instructions take additional scale operands for MXFP4/FP6/FP8 support
class ScaledMFMAOp<string mnemonic, int accSize, list<Trait> traits = []>
    : WAVEASMOp<mnemonic, !listconcat([Pure, DeclareOpInterfaceMethods<WaveASM_MFMAOp>], traits)> {
  // Operands: a_data, b_data, acc, a_scale, b_scale
  let arguments = (ins WaveASM_VALUSrc:$a, WaveASM_VALUSrc:$b, WaveASM_MFMAAccSrc:$acc,
                       WaveASM_VALUSrc:$a_scale, WaveASM_VALUSrc:$b_scale);
  let results = (outs WaveASM_AnyVecReg:$dst);
  let assemblyFormat = "$a `,` $b `,` $acc `,` $a_scale `,` $b_scale attr-dict `:` type($a) `,` type($b) `,` type($acc) `,` type($a_scale) `,` type($b_scale) `->` type($dst)";

  // Tied operand: result 0 is tied to operand 2 (accumulator) when acc is VGPR
  int TiedOperandIndex = 2;

  // Store accumulator size for verification
  int AccumulatorSize = accSize;

  // Custom verifier for accumulator size matching
  let hasVerifier = 1;
}

// MXFP4/FP6/FP8 variants (gfx950+)
def WaveASM_V_MFMA_SCALE_F32_16X16X128_F8F6F4 : ScaledMFMAOp<"v_mfma_scale_f32_16x16x128_f8f6f4", 4>;
def WaveASM_V_MFMA_SCALE_F32_32X32X64_F8F6F4 : ScaledMFMAOp<"v_mfma_scale_f32_32x32x64_f8f6f4", 16>;

//===----------------------------------------------------------------------===//
// VMEM Load Instructions
//===----------------------------------------------------------------------===//

// Buffer loads
def WaveASM_BUFFER_LOAD_DWORD : VMEMLoadOp<"buffer_load_dword", 1>;
def WaveASM_BUFFER_LOAD_DWORDX2 : VMEMLoadOp<"buffer_load_dwordx2", 2>;
def WaveASM_BUFFER_LOAD_DWORDX3 : VMEMLoadOp<"buffer_load_dwordx3", 3>;
def WaveASM_BUFFER_LOAD_DWORDX4 : VMEMLoadOp<"buffer_load_dwordx4", 4>;
def WaveASM_BUFFER_LOAD_UBYTE : VMEMLoadOp<"buffer_load_ubyte", 1>;
def WaveASM_BUFFER_LOAD_SBYTE : VMEMLoadOp<"buffer_load_sbyte", 1>;
def WaveASM_BUFFER_LOAD_USHORT : VMEMLoadOp<"buffer_load_ushort", 1>;
def WaveASM_BUFFER_LOAD_SSHORT : VMEMLoadOp<"buffer_load_sshort", 1>;

// Global loads
def WaveASM_GLOBAL_LOAD_DWORD : VMEMLoadOp<"global_load_dword", 1>;
def WaveASM_GLOBAL_LOAD_DWORDX2 : VMEMLoadOp<"global_load_dwordx2", 2>;
def WaveASM_GLOBAL_LOAD_DWORDX3 : VMEMLoadOp<"global_load_dwordx3", 3>;
def WaveASM_GLOBAL_LOAD_DWORDX4 : VMEMLoadOp<"global_load_dwordx4", 4>;
def WaveASM_GLOBAL_LOAD_UBYTE : VMEMLoadOp<"global_load_ubyte", 1>;
def WaveASM_GLOBAL_LOAD_SBYTE : VMEMLoadOp<"global_load_sbyte", 1>;
def WaveASM_GLOBAL_LOAD_USHORT : VMEMLoadOp<"global_load_ushort", 1>;
def WaveASM_GLOBAL_LOAD_SSHORT : VMEMLoadOp<"global_load_sshort", 1>;

// Flat loads
def WaveASM_FLAT_LOAD_DWORD : VMEMLoadOp<"flat_load_dword", 1>;
def WaveASM_FLAT_LOAD_DWORDX2 : VMEMLoadOp<"flat_load_dwordx2", 2>;
def WaveASM_FLAT_LOAD_DWORDX3 : VMEMLoadOp<"flat_load_dwordx3", 3>;
def WaveASM_FLAT_LOAD_DWORDX4 : VMEMLoadOp<"flat_load_dwordx4", 4>;

//===----------------------------------------------------------------------===//
// VMEM-to-LDS Load Instructions (Gather-to-LDS)
//
// These instructions load from global memory directly to LDS without going
// through VGPRs. M0 register must be set to the LDS destination offset.
//===----------------------------------------------------------------------===//

// VMEM-to-LDS Load: load from global to LDS (M0 contains LDS offset)
// No VGPR result - data goes directly to LDS
class VMEMToLDSLoadOp<string mnemonic, list<Trait> traits = []>
    : WAVEASMOp<mnemonic, !listconcat([DeclareOpInterfaceMethods<WaveASM_VMEMToLDSLoadOp>], traits)> {
  let summary = "Load from global memory directly to LDS";
  let arguments = (ins
    WaveASM_VRegOrImm:$voffset,   // Global memory offset per lane
    WaveASM_AnySGPR:$srd,         // Buffer descriptor (4 SGPRs)
    WaveASM_SRegOrImm:$soffset    // Scalar offset (usually 0)
  );
  let results = (outs);  // No VGPR result - writes to LDS via M0
  let assemblyFormat = "$voffset `,` $srd `,` $soffset attr-dict `:` type($voffset) `,` type($srd) `,` type($soffset)";
}

// Gather-to-LDS: buffer_load_dword with LDS destination
def WaveASM_BUFFER_LOAD_DWORD_LDS : VMEMToLDSLoadOp<"buffer_load_dword_lds">;
def WaveASM_BUFFER_LOAD_DWORDX4_LDS : VMEMToLDSLoadOp<"buffer_load_dwordx4_lds">;

//===----------------------------------------------------------------------===//
// M0 Register Operations
//===----------------------------------------------------------------------===//

def WaveASM_S_MOV_B32_M0 : WAVEASMOp<"s_mov_b32_m0", [WaveASM_SpecialRegOp]> {
  let summary = "Move value to M0 special register";
  let description = [{
    Sets the M0 register which is used for:
    - LDS destination offset in gather-to-LDS operations
    - LDS/GDS indirect addressing
  }];
  let arguments = (ins WaveASM_SRegOrImm:$src);
  let assemblyFormat = "$src attr-dict `:` type($src)";
}

def WaveASM_S_AND_SAVEEXEC_B64 : WAVEASMOp<"s_and_saveexec_b64", [WaveASM_SpecialRegOp]> {
  let summary = "Save exec to dst, then AND exec with VCC (implicit)";
  let description = [{
    dst = exec; exec &= vcc.
    VCC is read implicitly (set by a preceding V_CMP).
    Used for conditional execution: lanes where VCC is 0 become inactive.
    The saved exec is restored later via s_mov_b64_exec.
  }];
  let results = (outs WaveASM_AnySGPR:$dst);
  let assemblyFormat = "attr-dict `->` type($dst)";
}

def WaveASM_S_MOV_B64_EXEC : WAVEASMOp<"s_mov_b64_exec", [WaveASM_SpecialRegOp]> {
  let summary = "Restore exec from saved SGPR pair";
  let description = [{
    exec = src.
    Used to restore exec after a conditional execution region.
  }];
  let arguments = (ins WaveASM_AnySGPR:$src);
  let assemblyFormat = "$src attr-dict `:` type($src)";
}

//===----------------------------------------------------------------------===//
// VMEM Atomic Instructions
//===----------------------------------------------------------------------===//

// Buffer atomic: atomically add packed bf16 values to memory.
// vdata contains 2 packed bf16 values in a 32-bit VGPR.
// The instruction atomically adds both bf16 values to the corresponding
// bf16 halves at the 32-bit aligned memory address.
class VMEMAtomicOp<string mnemonic, list<Trait> traits = []>
    : WAVEASMOp<mnemonic, !listconcat([WaveASM_MemoryOp, DeclareOpInterfaceMethods<WaveASM_VMEMStoreOp>], traits)> {
  let arguments = (ins WaveASM_AnyVGPR:$data, WaveASM_AnySGPR:$saddr, WaveASM_VRegOrImm:$voffset,
                       DefaultValuedAttr<I64Attr, "0">:$instOffset);
  let assemblyFormat = "$data `,` $saddr `,` $voffset (`offset` `:` $instOffset^)? attr-dict `:` type($data) `,` type($saddr) `,` type($voffset)";
  let hasVerifier = 1;
}

def WaveASM_BUFFER_ATOMIC_PK_ADD_BF16 : VMEMAtomicOp<"buffer_atomic_pk_add_bf16">;
def WaveASM_BUFFER_ATOMIC_ADD_F32 : VMEMAtomicOp<"buffer_atomic_add_f32">;

//===----------------------------------------------------------------------===//
// VMEM Store Instructions
//===----------------------------------------------------------------------===//

// Buffer stores
def WaveASM_BUFFER_STORE_DWORD : VMEMStoreOp<"buffer_store_dword">;
def WaveASM_BUFFER_STORE_DWORDX2 : VMEMStoreOp<"buffer_store_dwordx2">;
def WaveASM_BUFFER_STORE_DWORDX3 : VMEMStoreOp<"buffer_store_dwordx3">;
def WaveASM_BUFFER_STORE_DWORDX4 : VMEMStoreOp<"buffer_store_dwordx4">;
def WaveASM_BUFFER_STORE_BYTE : VMEMStoreOp<"buffer_store_byte">;
def WaveASM_BUFFER_STORE_SHORT : VMEMStoreOp<"buffer_store_short">;

// Global stores
def WaveASM_GLOBAL_STORE_DWORD : VMEMStoreOp<"global_store_dword">;
def WaveASM_GLOBAL_STORE_DWORDX2 : VMEMStoreOp<"global_store_dwordx2">;
def WaveASM_GLOBAL_STORE_DWORDX3 : VMEMStoreOp<"global_store_dwordx3">;
def WaveASM_GLOBAL_STORE_DWORDX4 : VMEMStoreOp<"global_store_dwordx4">;
def WaveASM_GLOBAL_STORE_BYTE : VMEMStoreOp<"global_store_byte">;
def WaveASM_GLOBAL_STORE_SHORT : VMEMStoreOp<"global_store_short">;

// Flat stores
def WaveASM_FLAT_STORE_DWORD : VMEMStoreOp<"flat_store_dword">;
def WaveASM_FLAT_STORE_DWORDX2 : VMEMStoreOp<"flat_store_dwordx2">;
def WaveASM_FLAT_STORE_DWORDX3 : VMEMStoreOp<"flat_store_dwordx3">;
def WaveASM_FLAT_STORE_DWORDX4 : VMEMStoreOp<"flat_store_dwordx4">;

//===----------------------------------------------------------------------===//
// SMEM Load Instructions
//===----------------------------------------------------------------------===//

def WaveASM_S_LOAD_DWORD : SMEMLoadOp<"s_load_dword", 1>;
def WaveASM_S_LOAD_DWORDX2 : SMEMLoadOp<"s_load_dwordx2", 2>;
def WaveASM_S_LOAD_DWORDX4 : SMEMLoadOp<"s_load_dwordx4", 4>;
def WaveASM_S_LOAD_DWORDX8 : SMEMLoadOp<"s_load_dwordx8", 8>;
def WaveASM_S_LOAD_DWORDX16 : SMEMLoadOp<"s_load_dwordx16", 16>;

//===----------------------------------------------------------------------===//
// LDS (Local Data Share) Instructions
//===----------------------------------------------------------------------===//

// LDS Loads
def WaveASM_DS_READ_B32 : LDSLoadOp<"ds_read_b32", 1>;
def WaveASM_DS_READ_B64 : LDSLoadOp<"ds_read_b64", 2>;
def WaveASM_DS_READ_B128 : LDSLoadOp<"ds_read_b128", 4>;
def WaveASM_DS_READ2_B32 : LDSLoadOp<"ds_read2_b32", 2>;
def WaveASM_DS_READ2_B64 : LDSLoadOp<"ds_read2_b64", 4>;
def WaveASM_DS_READ_U8 : LDSLoadOp<"ds_read_u8", 1>;
def WaveASM_DS_READ_I8 : LDSLoadOp<"ds_read_i8", 1>;
def WaveASM_DS_READ_U16 : LDSLoadOp<"ds_read_u16", 1>;
def WaveASM_DS_READ_I16 : LDSLoadOp<"ds_read_i16", 1>;

// LDS Stores
def WaveASM_DS_WRITE_B32 : LDSStoreOp<"ds_write_b32">;
def WaveASM_DS_WRITE_B64 : LDSStoreOp<"ds_write_b64">;
def WaveASM_DS_WRITE_B128 : LDSStoreOp<"ds_write_b128">;
def WaveASM_DS_WRITE2_B32 : LDSStoreOp<"ds_write2_b32">;
def WaveASM_DS_WRITE2_B64 : LDSStoreOp<"ds_write2_b64">;
def WaveASM_DS_WRITE_B8 : LDSStoreOp<"ds_write_b8">;
def WaveASM_DS_WRITE_B16 : LDSStoreOp<"ds_write_b16">;

//===----------------------------------------------------------------------===//
// Control Flow Instructions
//===----------------------------------------------------------------------===//

def WaveASM_S_ENDPGM : WAVEASMOp<"s_endpgm", [WaveASM_ControlFlowOp]> {
  let summary = "End program execution";
  let assemblyFormat = "attr-dict";
}

// Note: Branch ops do not use Terminator trait because they exist
// inside a SingleBlock/NoTerminator container (ProgramOp).
// Control flow is assembly-like, not SSA block-based.

class WaveASM_BranchLikeOp<string mnemonic> :
    WAVEASMOp<mnemonic, [WaveASM_ControlFlowOp]> {
  let arguments = (ins SymbolRefAttr:$target);
  let assemblyFormat = "$target attr-dict";
}

def WaveASM_S_BRANCH : WaveASM_BranchLikeOp<"s_branch"> {
  let summary = "Unconditional branch";
}

def WaveASM_S_CBRANCH_SCC0 : WaveASM_BranchLikeOp<"s_cbranch_scc0"> {
  let summary = "Branch if SCC == 0";
}

def WaveASM_S_CBRANCH_SCC1 : WaveASM_BranchLikeOp<"s_cbranch_scc1"> {
  let summary = "Branch if SCC == 1";
}

def WaveASM_S_CBRANCH_VCCZ : WaveASM_BranchLikeOp<"s_cbranch_vccz"> {
  let summary = "Branch if VCC == 0";
}

def WaveASM_S_CBRANCH_VCCNZ : WaveASM_BranchLikeOp<"s_cbranch_vccnz"> {
  let summary = "Branch if VCC != 0";
}

def WaveASM_S_CBRANCH_EXECZ : WaveASM_BranchLikeOp<"s_cbranch_execz"> {
  let summary = "Branch if EXEC == 0";
}

def WaveASM_S_CBRANCH_EXECNZ : WaveASM_BranchLikeOp<"s_cbranch_execnz"> {
  let summary = "Branch if EXEC != 0";
}

def WaveASM_LabelOp : WAVEASMOp<"label"> {
  let summary = "Define a branch target label";
  let arguments = (ins SymbolNameAttr:$name);
  let assemblyFormat = "$name attr-dict";
}

//===----------------------------------------------------------------------===//
// Synchronization Instructions
//===----------------------------------------------------------------------===//

class WaveASM_CountedWaitOp<string mnemonic> :
  WAVEASMOp<mnemonic, [WaveASM_ControlFlowOp]> {
  let arguments = (ins I32Attr:$count);
  let assemblyFormat = "$count attr-dict";
}

def WaveASM_S_WAITCNT : WAVEASMOp<"s_waitcnt", [WaveASM_ControlFlowOp]> {
  let summary = "Wait for memory operations to complete";
  let arguments = (ins
    OptionalAttr<I32Attr>:$vmcnt,
    OptionalAttr<I32Attr>:$lgkmcnt,
    OptionalAttr<I32Attr>:$expcnt
  );
  let assemblyFormat = "(`vmcnt` `(` $vmcnt^ `)`)? (`lgkmcnt` `(` $lgkmcnt^ `)`)? (`expcnt` `(` $expcnt^ `)`)? attr-dict";
}

def WaveASM_S_WAITCNT_VMCNT : WaveASM_CountedWaitOp<"s_waitcnt_vmcnt"> {
  let summary = "Wait for vector memory operations";
}

def WaveASM_S_WAITCNT_LGKMCNT : WaveASM_CountedWaitOp<"s_waitcnt_lgkmcnt"> {
  let summary = "Wait for LDS/GDS/scalar memory operations";
}

def WaveASM_S_WAITCNT_EXPCNT : WaveASM_CountedWaitOp<"s_waitcnt_expcnt"> {
  let summary = "Wait for export operations";
}

def WaveASM_S_BARRIER : WAVEASMOp<"s_barrier", [WaveASM_ControlFlowOp]> {
  let summary = "Workgroup barrier synchronization";
  let assemblyFormat = "attr-dict";
}

// Ops that take a single integer operand but are not memory waits.
// Structurally identical to CountedWaitOp (same arguments and format)
// but semantically distinct: these are scalar control ops, not waitcnts.
class WaveASM_CountedScalarOp<string mnemonic> :
  WAVEASMOp<mnemonic, [WaveASM_ControlFlowOp]> {
  let arguments = (ins I32Attr:$count);
  let assemblyFormat = "$count attr-dict";
}

def WaveASM_S_NOP : WaveASM_CountedScalarOp<"s_nop"> {
  let summary = "No operation (wait cycles)";
}

def WaveASM_S_SETPRIO : WaveASM_CountedScalarOp<"s_setprio"> {
  let summary = "Set wave priority level (0=default, 1-3=elevated)";
}

def WaveASM_S_SCHED_BARRIER : WaveASM_CountedScalarOp<"s_sched_barrier"> {
  let summary = "Scheduling barrier (bitmask controls which instructions cross)";
}

//===----------------------------------------------------------------------===//
// Misc/Utility Operations
//===----------------------------------------------------------------------===//

def WaveASM_DCEProtectOp : WAVEASMOp<"dce_protect", [WaveASM_NonEmittingOp]> {
  let summary = "Prevent DCE of a value written to a precolored register";
  let description = [{
    Dummy side-effecting op that keeps its operand alive through DCE.

    When an op writes to a precolored physical register (e.g. an SRD word)
    but no SSA consumer uses the result, DCE would eliminate the producer.
    Placing dce_protect on the result prevents this.  The op emits nothing
    in the final assembly.
  }];
  let arguments = (ins WaveASM_AnyReg:$val);
  let assemblyFormat = "$val attr-dict `:` type($val)";
}

def WaveASM_CommentOp : WAVEASMOp<"comment"> {
  let summary = "Emit a comment in generated assembly";
  let arguments = (ins StrAttr:$text);
  let assemblyFormat = "$text attr-dict";
}

def WaveASM_RawOp : WAVEASMOp<"raw"> {
  let summary = "Emit raw assembly text";
  let arguments = (ins StrAttr:$text);
  let assemblyFormat = "$text attr-dict";
}

#endif // WaveASM_DIALECT_WAVEASMOPS
