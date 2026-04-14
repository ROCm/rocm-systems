# WaveASM Legalization Design

This document describes the legalization strategy for the WaveASM backend:
how pseudo-ops are lowered to legal AMDGCN machine ops with correct
register placement and operand widths.

## Problem Statement

The WaveASM backend accepts MLIR input from two frontends:

1. **TranslateFromMLIR** — translates `gpu`, `arith`, `vector`, `scf`,
   `amdgpu` dialect ops.
2. **TranslateFromLLVM** — translates LLVM dialect ops (used when the Water
   frontend lowers through LLVM IR).

Both frontends face the same set of legalization problems:

- **Type width**: LLVM IR uses i64 for pointer arithmetic and index math.
   AMDGCN VALU instructions are 32-bit. i64 operations must be narrowed or
   expanded to i32 pairs.
- **Register placement**: AMDGCN has two register files (SGPR and VGPR) with
   strict placement rules. Kernel arguments live in SGPRs; lane-varying values
   must be in VGPRs.
- **Constant bus**: VALU instructions can read at most one SGPR operand per
   instruction (pre-GFX10). Violations require inserting SGPR-to-VGPR moves.
- **Instruction selection**: A pseudo-op `arith.add` can lower to `s_add_u32` (scalar),
   `v_add_u32` (vector), or a carry chain for i64.

## Architecture

The legalization strategy follows the real LLVM AMDGPU backend: emit
structurally correct but "unlegalized" IR during translation, then fix it up
in dedicated post-translation passes.

### Pipeline

```
  Translation (TranslateFromMLIR / TranslateFromLLVM)
       │
       │  Emits pseudo-ops for arithmetic.
       │  Does NOT make register placement or width decisions.
       │
       ▼
  ┌──────────────────────────────────────┐
  │       Type Legalization Pass         │
  │                                      │
  │  - i64 → i32 narrowing or expansion  │
  │  - i64 add → i32 carry chain         │
  │  - i64 mul → i32 partial products    │
  │  - i64 icmp → hi/lo comparison       │
  │  - trunc/sext/zext lowering          │
  └──────────────┬───────────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────────┐
  │    Register Legalization Pass        │
  │                                      │
  │  - Assigns SGPR vs VGPR per value    │
  │  - Inserts v_mov_b32 for SGPR→VGPR   │
  │  - Enforces constant bus limits      │
  └──────────────┬───────────────────────┘
                 │
                 ▼
  ┌──────────────────────────────────────┐
  │     Instruction Selection Pass       │
  │                                      │
  │  - Lowers pseudo-ops to concrete     │
  │    machine ops (s_add/v_add/etc.)    │
  │  - Selects SALU vs VALU encoding     │
  └──────────────┬───────────────────────┘
                 │
                 ▼
       Existing pipeline (CSE, Peephole, Liveness, RegAlloc, Emit)
```

Note: these three passes may be combined into fewer passes if the separation
proves unnecessary. The logical ordering matters more than the physical pass
count.

## Pseudo-Operations

Translation emits a targeted set of **pseudo-ops** for cases where the
register file and/or width decision is non-trivial. These ops carry the
original type information and are lowered by legalization.

### Scope

Pseudo-ops are introduced only for arithmetic and comparison -- operations
where the SGPR/VGPR and width choice depends on context. Operations with
unambiguous lowering (SRD setup, buffer loads, MFMA, barriers, branch, endpgm)
continue to emit concrete machine ops directly during translation.

### Implemented pseudo-ops

| Pseudo-op | Lowers to (SALU) | Lowers to (VALU) |
|-----------|-----------------|-------------------|
| `waveasm.arith.add : iN` | `s_add_u32` (+ `s_addc_u32` for i64) | `v_add_u32` (+ `v_addc_co_u32` for i64) |
| `waveasm.arith.mul : iN` | `s_mul_i32` (+ `s_mul_hi_u32` for i64) | `v_mul_lo_u32` (+ partial products for i64) |
| `waveasm.arith.cmp_XX : iN` | `s_cmp_XX_u32/i32` | `v_cmp_XX_u32/i32` (+ hi/lo for i64) |
| `waveasm.arith.sext : i32 -> i64` | `s_ashr_i32` + pack | `v_ashrrev_i32` + pack |
| `waveasm.arith.zext : i32 -> i64` | `s_mov_b32(0)` + pack | `v_mov_b32(0)` + pack |
| `waveasm.arith.trunc : i64 -> i32` | extract sub-register | extract sub-register |
| `waveasm.arith.select` | `s_cselect_b32` | `v_cndmask_b32` |

The pseudo-ops use a register-file-agnostic type (e.g., a plain integer type
or a new `!waveasm.greg` virtual type) to defer the SGPR/VGPR decision.

### What stays concrete

These operations emit concrete machine ops directly during translation:

- **SRD/buffer ops**: `s_load_dwordx2`, `s_mov_b32` for descriptor fields,
  `buffer_load_*`, `buffer_store_*`.
- **LDS ops**: `ds_read_*`, `ds_write_*`.
- **MFMA**: `v_mfma_*` — always VALU, always concrete widths.
- **Lane ID**: `v_mbcnt_lo/hi` — always VGPR.
- **Control flow**: `s_branch`, `s_cbranch_*`, `s_endpgm`, `s_barrier`.
- **Waitcnts**: `s_waitcnt` — always scalar.
- **Comments, labels, raw asm**: pass-through, no legalization needed.

## Type Legalization Details

### i64 narrowing (short-term)

For dynamic dimension arguments that are known to fit in 32 bits, narrowing
is sufficient:

```
waveasm.arith.add %a, %b : i64
  →  %a32 = waveasm.arith.trunc %a : i64 → i32
     %b32 = waveasm.arith.trunc %b : i64 → i32
     %r32 = waveasm.arith.add %a32, %b32 : i32
     %r   = waveasm.arith.sext %r32 : i32 → i64   // if result used as i64
```

### i64 expansion (long-term)

For true 64-bit arithmetic, expand to carry chains:

```
waveasm.arith.add %a, %b : i64
  →  %a_lo, %a_hi = split %a
     %b_lo, %b_hi = split %b
     %r_lo, %carry = v_add_co_u32 %a_lo, %b_lo
     %r_hi = v_addc_co_u32 %a_hi, %b_hi, %carry
     %r = merge %r_lo, %r_hi
```

The decision between narrowing and expansion can be driven by value range
analysis or simply by op type (e.g., always narrow kernel args, always expand
address arithmetic).

## Register Legalization Details

### Divergence model

Values are classified as **uniform** (same across all lanes) or **divergent**
(varies per lane):

| Source | Classification |
|--------|---------------|
| Kernel arguments (preloaded SGPRs) | Uniform |
| `workitem.id.x` / `v_mbcnt` | Divergent |
| Constants / immediates | Uniform |
| Derived from any divergent value | Divergent |
| Loop induction variable (scalar) | Uniform |
| `v_readfirstlane_b32` result | Uniform |

Uniform values prefer SGPRs. Divergent values must be in VGPRs.

For the initial implementation, divergence is inferred demand-driven: when an
instruction requires a VGPR operand and the source is an SGPR, insert a move.
No explicit divergence analysis pass is needed yet.

### Constant bus enforcement

After register assignment, scan each VALU instruction. If it reads more than
one distinct SGPR:
1. Pick one SGPR operand (prefer the one with fewer uses, to minimize moves).
2. Insert `v_mov_b32` to copy it to a VGPR.
3. Rely on CSE to deduplicate identical moves.

### VGPR → SGPR (future)

`v_readfirstlane_b32` extracts lane 0 from a VGPR into an SGPR. Needed when a
divergent value must feed a scalar-only instruction (e.g., `s_load`, SRD
field). Not needed for current use cases but the legalization framework should
accommodate it.

## Comparison With LLVM AMDGPU Backend

| Aspect | LLVM AMDGPU | WaveASM |
|--------|------------|---------|
| Pseudo-ops | G_ADD, G_MUL (GlobalISel) | waveasm.arith.add, .mul |
| Type legalization | LegalizerInfo rules | Dedicated pass |
| Register placement | RegBankSelect + UniformityAnalysis | Demand-driven + constant bus pass |
| Constant bus | SIInstrInfo::legalizeOperands (post-ISel) | Register legalization pass |
| i64 lowering | RegBankLegalizeHelper | Type legalization pass |
| VGPR↔SGPR copies | SIFixSGPRCopies (cost heuristic) | Simple demand-driven insertion |

WaveASM intentionally simplifies the LLVM model. The full LLVM backend has
~15 post-ISel fixup passes; WaveASM targets 2-3 legalization passes covering
the same ground for the subset of IR patterns we generate.

## Implementation Plan

### Phase 1: Pseudo-ops + type legalization

1. Define arithmetic pseudo-ops in `WaveASMOps.td`.
2. Update `TranslateFromLLVM` to emit pseudo-ops instead of concrete ops with
   inline `ensure32Bit`/`ensureVGPR` calls.
3. Implement type legalization pass (i64 narrowing only).

### Phase 2: Register legalization

1. Implement register legalization pass: SGPR/VGPR assignment + constant bus.
2. Update `TranslateFromMLIR` to also emit pseudo-ops where appropriate.
3. Remove all inline register fixup code from both translators.

### Phase 3: Full i64 + instruction selection

1. Add i64 carry-chain expansion to type legalization.
2. Add instruction selection pass (pseudo-op → concrete, SALU vs VALU).
3. Add VGPR→SGPR support (`v_readfirstlane_b32`) if needed.

## Resolved Questions

- **Pseudo-op type representation**: pseudo-ops use polymorphic result types
  (register-file-agnostic `!waveasm.sreg`/`!waveasm.vreg`/`!waveasm.imm`).
  Width is inferred from operand register sizes.
- **Pass count**: type legalization, register legalization, and instruction
  selection are currently combined in a single `ArithLegalization` pass.
  Splitting into separate passes is tracked as future work.
- **Value range analysis**: not needed. The LLVM frontend already emits
  explicit `trunc` ops where narrowing is safe; legalization lowers those
  and uses i64 carry-chain expansion for true 64-bit arithmetic.
