// Copyright 2026 The Wave Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef WaveASM_TRANSFORMS_PASSES
#define WaveASM_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Linear Scan Register Allocation Pass
//===----------------------------------------------------------------------===//

def WAVEASMLinearScan : Pass<"waveasm-linear-scan"> {
  let summary = "Linear scan register allocation";
  let description = [{
    Allocates physical registers to virtual registers using the
    linear scan algorithm:

    1. Sort live ranges by start point
    2. For each range (in order):
       a. Expire ranges that ended before this point
       b. Allocate physical register respecting constraints
       c. Add to active set

    Features:
    - Respects ABI precoloring (v0, s[0:1], etc.)
    - Handles alignment constraints (for register pairs/quads)
    - Handles size constraints (for MFMA accumulators)
    - Fails compilation if allocation exceeds limits (no spilling)

    After allocation, virtual register types are replaced with
    physical register types in-place.
  }];

  let options = [
    Option<"maxVGPRs", "max-vgprs", "int64_t", "256",
           "Maximum VGPRs available">,
    Option<"maxSGPRs", "max-sgprs", "int64_t", "102",
           "Maximum user-addressable SGPRs (s0-s101 on GFX9+)">,
    Option<"maxAGPRs", "max-agprs", "int64_t", "256",
           "Maximum AGPRs available">,
    Option<"bidirectionalThreshold", "bidirectional-threshold", "int64_t", "75",
           "Lifetime percentage threshold for top-down VGPR allocation (0 to disable)">
  ];

  let statistics = [
    Statistic<"peakVGPRs", "Peak VGPRs used", "registers">,
    Statistic<"peakSGPRs", "Peak SGPRs used", "registers">,
    Statistic<"rangesAllocated", "Total ranges allocated", "count">,
    Statistic<"rangesExpired", "Total ranges expired", "count">
  ];

  let dependentDialects = ["::waveasm::WaveASMDialect"];
}

//===----------------------------------------------------------------------===//
// Hazard Mitigation Pass
//===----------------------------------------------------------------------===//

def WAVEASMHazardMitigation : Pass<"waveasm-hazard-mitigation"> {
  let summary = "Insert s_nop instructions to mitigate hardware hazards";
  let description = [{
    Handles hardware-specific hazards that require NOP insertion.

    Currently handles:
    - VALU → v_readfirstlane hazard (gfx940+): When a VALU instruction
      writes a VGPR and the next instruction is v_readfirstlane reading
      that VGPR, an s_nop 0 must be inserted between them.

    The pass scans the instruction stream and inserts s_nop instructions
    where needed to ensure correct execution on the target hardware.
  }];

  let options = [
    Option<"targetArch", "target", "std::string", "\"gfx942\"",
           "Target GPU architecture for hazard rules">
  ];

  let statistics = [
    Statistic<"numNopsInserted", "s_nop instructions inserted", "count">
  ];

  let dependentDialects = ["::waveasm::WaveASMDialect"];
}

//===----------------------------------------------------------------------===//
// Waitcnt Insertion Pass
//===----------------------------------------------------------------------===//

def WAVEASMInsertWaitcnt : Pass<"waveasm-insert-waitcnt"> {
  let summary = "Insert s_waitcnt instructions for memory operation synchronization";
  let description = [{
    Implements a ticket-based tracking system for VMEM and LGKM memory
    operations to insert optimal s_waitcnt instructions.

    The pass tracks two types of memory operations:
    - VMEM: buffer_load, buffer_store, global_load, global_store
    - LGKM: ds_read, ds_write, s_load

    For each memory operation, a "ticket" is assigned. When a value
    produced by a memory operation is used, the pass computes the
    minimum wait threshold needed and inserts s_waitcnt if required.

    Wait coalescing: The pass tracks previously emitted waits and only
    emits a new wait if it's stricter than the previous one, avoiding
    redundant waits and maximizing latency hiding.

    Example:
    - If last_vmem_ticket=5 and we need ticket 3 to complete:
    - threshold = max(0, 5-3) = 2
    - Emit: s_waitcnt vmcnt(2)
    - This waits for ticket 3 while allowing tickets 4,5 to remain in-flight
  }];

  let options = [
    Option<"insertAfterLoads", "insert-after-loads", "bool", "false",
           "Insert waitcnt immediately after loads (conservative)">,
    Option<"ticketedWaitcnt", "ticketed-waitcnt", "bool", "true",
           "Ticket-based waitcnt/barrier insertion (set to false to disable)">
  ];

  let statistics = [
    Statistic<"numVmemOps", "VMEM operations tracked", "count">,
    Statistic<"numLgkmOps", "LGKM operations tracked", "count">,
    Statistic<"numWaitcntInserted", "Waitcnt instructions inserted", "count">
  ];

  let dependentDialects = ["::waveasm::WaveASMDialect"];
}

//===----------------------------------------------------------------------===//
// Peephole Optimization Pass
//===----------------------------------------------------------------------===//

def WAVEASMPeephole : Pass<"waveasm-peephole"> {
  let summary = "Local pattern-based peephole optimizations for WAVEASM IR";
  let description = [{
    Instruction fusion, strength reduction, constant folding, and redundant
    move elimination.  All transformations are local rewrite patterns applied
    via the greedy pattern rewrite driver.
  }];

  let dependentDialects = ["::waveasm::WaveASMDialect"];
}

//===----------------------------------------------------------------------===//
// M0 Redundancy Elimination Pass
//===----------------------------------------------------------------------===//

def WAVEASMM0RedundancyElim : Pass<"waveasm-m0-redundancy-elim"> {
  let summary = "Eliminate redundant M0 register writes";
  let description = [{
    Tracks the last s_mov_b32_m0 source value within each basic block and
    erases writes that set M0 to the value it already holds.  Conservatively
    resets tracking across region-bearing ops (loops, ifs) whose bodies may
    clobber M0.
  }];

  let dependentDialects = ["::waveasm::WaveASMDialect"];
}

//===----------------------------------------------------------------------===//
// Scoped CSE Pass
//===----------------------------------------------------------------------===//

def WAVEASMScopedCSE : Pass<"waveasm-scoped-cse"> {
  let summary = "Scoped Common Subexpression Elimination for WAVEASM IR";
  let description = [{
    Implements scoped CSE respecting structured control flow by maintaining
    a scope stack.  Only operations within the same or dominating scope can
    be deduplicated.
  }];

  let dependentDialects = ["::waveasm::WaveASMDialect"];
}

//===----------------------------------------------------------------------===//
// Buffer Load Strength Reduction Pass
//===----------------------------------------------------------------------===//

def WAVEASMBufferLoadStrengthReduction
    : Pass<"waveasm-buffer-load-strength-reduction"> {
  let summary = "Replace per-iteration buffer_load address computation with "
                "soffset-based incremental addressing";
  let description = [{
    Replaces per-iteration buffer_load voffset recomputation with precomputed
    voffsets and SGPR soffset bumping.  Symbolically differentiates the address
    chain w.r.t. the induction variable to compute a constant stride, then
    groups candidates by (SRD, stride) to share soffsets.
  }];

  let dependentDialects = ["::waveasm::WaveASMDialect"];
}

//===----------------------------------------------------------------------===//
// Loop Address Promotion Pass
//===----------------------------------------------------------------------===//

def WAVEASMLoopAddressPromotion : Pass<"waveasm-loop-address-promotion"> {
  let summary = "Promote loop-carried LDS read address computation";
  let description = [{
    Promotes loop-carried LDS read address computation from per-iteration
    V_ADD_U32 operations to precomputed rotating VGPR iter_args.  Eliminates
    VALU address computation from the loop body.
  }];

  let dependentDialects = ["::waveasm::WaveASMDialect"];
}

//===----------------------------------------------------------------------===//
// Scale Pack Elimination Pass
//===----------------------------------------------------------------------===//

def WAVEASMScalePackElimination : Pass<"waveasm-scale-pack-elimination"> {
  let summary = "Eliminate BFE/LSHL_OR round-trips for B-scale iter_args";
  let description = [{
    Eliminates redundant BFE->iter_arg->LSHL_OR round-trips for B-scale values
    in scaled MFMA loops.  Carries the dword directly as a loop iter_arg instead
    of four separate byte extractions.
  }];

  let dependentDialects = ["::waveasm::WaveASMDialect"];
}

//===----------------------------------------------------------------------===//
// Extract Scalarization Pass
//===----------------------------------------------------------------------===//

def WAVEASMExtractScalarization
    : Pass<"waveasm-extract-scalarization"> {
  let summary = "Scalarize vector.extract from broadcast+dense-const patterns";
  let description = [{
    Pre-translation pass that rewrites
      vector.extract[k]( index_cast?( select?( addi(broadcast(x), dense<[...]>) )))
    into scalar operations: arith.addi %x, dense[k], with an optional scalar
    arith.select if the original chain included one.

    This eliminates non-splat dense vector constants before the WaveASM
    translator runs, so translation handlers only see ordinary scalar IR.
  }];

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
    "::mlir::vector::VectorDialect"
  ];
}

//===----------------------------------------------------------------------===//
// SCC Spill/Reload Pass
//===----------------------------------------------------------------------===//

def WAVEASMSCCSpillReload : Pass<"waveasm-scc-spill-reload"> {
  let summary = "Insert SCC spill/reload around SCC-clobbering ops";
  let description = [{
    When an SCC live range extends across SCC-clobbering ops (e.g. after
    CSE merges identical SCC producers), this pass inserts s_cselect_b32
    (spill: SCC -> SGPR) right after the producer and s_cmp_ne_u32
    (reload: SGPR -> SCC) right before each affected consumer.
  }];
  let dependentDialects = ["::waveasm::WaveASMDialect"];
}

//===----------------------------------------------------------------------===//
// SCC Verifier Pass
//===----------------------------------------------------------------------===//

def WAVEASMSCCVerifier : Pass<"waveasm-scc-verifier"> {
  let summary = "Verify SCC (Scalar Condition Code) liveness invariants";
  let description = [{
    Checks that no SCC-clobbering SALU instruction sits between an
    SCC-producing op (SCCDef trait) and its consumer.
  }];
  let dependentDialects = ["::waveasm::WaveASMDialect"];
}

//===----------------------------------------------------------------------===//
// VGPR Compaction Pass
//===----------------------------------------------------------------------===//

def WAVEASMVGPRCompaction : Pass<"waveasm-vgpr-compaction"> {
  let summary = "Compact physical VGPR assignments to reduce peak register count";
  let description = [{
    Re-assigns physical VGPRs after linear scan allocation to eliminate
    fragmentation caused by interleaved buffer_load and ds_read destinations.
    Uses a shortest-first greedy strategy: short-lived values get low
    registers, long-lived values get high registers, eliminating gaps
    from interleaving.

    Scope: VGPRs only (PVRegType).  SGPRs and AGPRs are not compacted
    because SGPR pressure is rarely the bottleneck and AGPRs have fixed
    hardware mapping for MFMA accumulators.
  }];
  let dependentDialects = ["::waveasm::WaveASMDialect"];
}

//===----------------------------------------------------------------------===//
// Memory Offset Optimization Pass
//===----------------------------------------------------------------------===//

def WAVEASMMemoryOffsetOpt : Pass<"waveasm-memory-offset-opt"> {
  let summary = "Fold constant address components into memory instruction "
                "offset fields";
  let description = [{
    Folds constant address components into memory instruction offset fields,
    reducing VALU instruction count.  Handles constant adds, constant through
    V_MOV_B32, and algebraic distribution through shifts.
  }];

  let dependentDialects = ["::waveasm::WaveASMDialect"];
}

//===----------------------------------------------------------------------===//
// Translate from LLVM Dialect Pass
//===----------------------------------------------------------------------===//

def WAVEASMGPUModuleToBinary : Pass<"waveasm-gpu-module-to-binary"> {
  let summary = "Emit assembly and link gpu.module { waveasm.program } to gpu.binary";
  let description = [{
    Final pass in the WaveASM pipeline.  Expects gpu.module ops containing
    register-allocated waveasm.program ops.  Emits AMDGCN assembly, assembles
    + links to HSACO via LLVM MC, and replaces each gpu.module with a
    gpu.binary holding the code object.
  }];

  let options = [
    Option<"targetArch", "target", "std::string", "\"gfx942\"",
           "Target GPU architecture">,
    Option<"lldPath", "lld-path", "std::string", "\"\"",
           "Path to ld.lld for linking HSACO">
  ];

  let dependentDialects = [
    "::waveasm::WaveASMDialect",
    "::mlir::gpu::GPUDialect"
  ];
}

def WAVEASMArithLegalization : Pass<"waveasm-arith-legalization"> {
  let summary = "Lower generic arithmetic pseudo-ops to concrete SALU/VALU ops";
  let description = [{
    Lowers register-file-agnostic arith pseudo-ops (arith.add, arith.mul,
    arith.cmp, arith.select, arith.trunc, arith.sext, arith.zext) to
    concrete SALU or VALU machine ops.

    Register file assignment (demand-driven divergence):
    - If any operand is a VGPR → use VALU, result is VGPR.
    - If all operands are SGPR/immediate → use SALU, result is SGPR.

    Constant bus enforcement:
    - VALU ops can read at most one SGPR per instruction.
    - If both operands are SGPRs, one is moved to VGPR via v_mov_b32.

    i64 legalization (split into i32 halves):
    - add: carry chain (s_add_u32 + s_addc_u32 / v_add_co_u32 + v_addc_co_u32).
    - mul: schoolbook decomposition (lo*lo, hi*lo, lo*hi).
    - cmp eq/ne: XOR each half, OR, compare to zero.
    - cmp ordered: hi/lo decomposition with signed hi / unsigned lo.
    - select: split both operands, v_cndmask_b32 each half, merge.
    - sext/zext: extend to {lo, sign/zero} pair.
    - trunc: extract lo half.

    Reports errors on unsupported widths (anything other than i32/i64).
  }];

  let statistics = [
    Statistic<"numOpsLegalized", "Pseudo-ops legalized", "count">,
    Statistic<"numSGPRToVGPRMoves", "SGPR-to-VGPR moves inserted", "count">
  ];

  let dependentDialects = ["::waveasm::WaveASMDialect"];
}

def WAVEASMTranslateFromLLVM : Pass<"waveasm-translate-from-llvm"> {
  let summary = "Translate LLVM dialect kernels to WaveASM IR";
  let description = [{
    Translates gpu.module { llvm.func } with rocdl intrinsics into
    waveasm.program ops.  Fails on any unhandled op (strict mode).
  }];

  let options = [
    Option<"targetArch", "target", "std::string", "\"gfx950\"",
           "Target GPU architecture">
  ];

  let dependentDialects = ["::waveasm::WaveASMDialect"];
}

#endif // WaveASM_TRANSFORMS_PASSES
