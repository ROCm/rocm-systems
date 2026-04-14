# WaveASM Architecture

This document describes the architecture of the WaveASM C++ backend, which translates MLIR to optimized AMDGCN assembly for AMD GPUs.

## High-Level Overview

```
                    ┌─────────────────────────────────────────────────────────────┐
                    │                        Wave DSL                             │
                    │              (Python frontend for GPU kernels)              │
                    └─────────────────────────┬───────────────────────────────────┘
                                              │
                                              ▼
                    ┌─────────────────────────────────────────────────────────────┐
                    │                      MLIR Module                            │
                    │    (gpu, arith, vector, memref, scf, amdgpu dialects)       │
                    └─────────────────────────┬───────────────────────────────────┘
                                              │
                    ╔═════════════════════════▼═══════════════════════════════════╗
                    ║                  WaveASM C++ Backend                        ║
                    ╠═════════════════════════════════════════════════════════════╣
                    ║  ┌─────────────────────────────────────────────────────┐    ║
                    ║  │              TranslateFromMLIR Pass                 │    ║
                    ║  │         (MLIR → WaveASM Dialect IR)                 │    ║
                    ║  └─────────────────────────┬───────────────────────────┘    ║
                    ║                            ▼                                ║
                    ║  ┌─────────────────────────────────────────────────────┐    ║
                    ║  │              WaveASM Dialect IR                     │    ║
                    ║  │    (Generic pseudo-ops + concrete machine ops)      │    ║
                    ║  └─────────────────────────┬───────────────────────────┘    ║
                    ║                            ▼                                ║
                    ║  ┌─────────────────────────────────────────────────────┐    ║
                    ║  │              Legalization Passes                    │    ║
                    ║  │    (Type legalization, register placement, ISel)    │    ║
                    ║  └─────────────────────────┬───────────────────────────┘    ║
                    ║                            ▼                                ║
                    ║  ┌─────────────────────────────────────────────────────┐    ║
                    ║  │               Optimization Passes                   │    ║
                    ║  │    (CSE, Peephole, Liveness, Ticketing, Hazard)     │    ║
                    ║  └─────────────────────────┬───────────────────────────┘    ║
                    ║                            ▼                                ║
                    ║  ┌─────────────────────────────────────────────────────┐    ║
                    ║  │            Linear Scan Register Allocation          │    ║
                    ║  │         (Virtual → Physical registers)              │    ║
                    ║  └─────────────────────────┬───────────────────────────┘    ║
                    ║                            ▼                                ║
                    ║  ┌─────────────────────────────────────────────────────┐    ║
                    ║  │               Assembly Emitter                      │    ║
                    ║  │         (WaveASM IR → AMDGCN assembly)              │    ║
                    ║  └─────────────────────────┬───────────────────────────┘    ║
                    ╚═════════════════════════════╪═══════════════════════════════╝
                                              │
                                              ▼
                    ┌─────────────────────────────────────────────────────────────┐
                    │                 AMDGCN Assembly (.s)                        │
                    │           (with .amdhsa_kernel metadata)                    │
                    └─────────────────────────┬───────────────────────────────────┘
                                              │
                                              ▼
                    ┌─────────────────────────────────────────────────────────────┐
                    │              ROCm Toolchain (clang + lld)                   │
                    │                 HSACO Binary (.hsaco)                       │
                    └─────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. Translation Layer (TranslateFromMLIR)

The translation layer converts standard MLIR dialects to WaveASM dialect operations.

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                          TranslateFromMLIR Pass                                │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│   ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐        │
│   │  Handler Registry │    │   Value Mapping  │    │   SRD Manager    │        │
│   │                  │    │                  │    │                  │        │
│   │  arith.addi  ───►│    │ MLIR Value ────► │    │  Buffer desc     │        │
│   │  arith.muli     │    │ WaveASM Value    │    │  tracking        │        │
│   │  gpu.thread_id  │    │                  │    │                  │        │
│   │  amdgpu.mfma    │    │                  │    │                  │        │
│   │  vector.load    │    │                  │    │                  │        │
│   │  scf.for        │    │                  │    │                  │        │
│   │  ...            │    │                  │    │                  │        │
│   └──────────────────┘    └──────────────────┘    └──────────────────┘        │
│                                                                                │
│   ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐        │
│   │  Loop Context    │    │  Expression      │    │   Binding        │        │
│   │                  │    │  Cache (CSE)     │    │   Tracker        │        │
│   │  Iter args       │    │                  │    │                  │        │
│   │  Phi resolution  │    │  Avoid recompute │    │  Kernel args     │        │
│   └──────────────────┘    └──────────────────┘    └──────────────────┘        │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

**Key Features:**
- **Handler Registry**: Maps MLIR operations to WaveASM generation functions
- **Value Mapping**: Tracks correspondence between MLIR SSA values and WaveASM values
- **SRD Manager**: Manages Shader Resource Descriptors for buffer operations
- **Loop Context**: Handles `scf.for` iteration arguments and loop-carried values
- **Expression Cache**: Avoids redundant computation through CSE

### 2. WaveASM Dialect

The WaveASM dialect is a pure SSA representation close to AMDGCN assembly.

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                             WaveASM Dialect                                    │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  Types:                                                                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐           │
│  │   vreg      │  │   sreg      │  │   pvreg<N>  │  │   psreg<N>  │           │
│  │  (virtual   │  │  (virtual   │  │  (physical  │  │  (physical  │           │
│  │   VGPR)     │  │   SGPR)     │  │   VGPR)     │  │   SGPR)     │           │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘           │
│                                                                                │
│  Operation Categories:                                                         │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                   │
│  │  Vector ALU    │  │  Scalar ALU    │  │    Memory      │                   │
│  │  v_add_u32     │  │  s_add_u32     │  │  global_load   │                   │
│  │  v_mul_lo_u32  │  │  s_mul_i32     │  │  global_store  │                   │
│  │  v_fma_f32     │  │  s_lshl_b32    │  │  ds_read_b32   │                   │
│  │  v_mov_b32     │  │  s_and_b32     │  │  ds_write_b32  │                   │
│  └────────────────┘  └────────────────┘  └────────────────┘                   │
│                                                                                │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                   │
│  │     MFMA       │  │  Control Flow  │  │    Utility     │                   │
│  │  v_mfma_f32_   │  │  s_branch      │  │  comment       │                   │
│  │  16x16x16_f16  │  │  s_cbranch_*   │  │  label         │                   │
│  │  v_mfma_f32_   │  │  s_endpgm      │  │  precolored.*  │                   │
│  │  32x32x8_f16   │  │  s_barrier     │  │  raw           │                   │
│  └────────────────┘  └────────────────┘  └────────────────┘                   │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

### 3. Legalization (see [legalization.md](legalization.md))

Translation emits generic pseudo-ops (`waveasm.arith.add`, `.mul`, `.cmp_*`)
for arithmetic, deferring type width and register file decisions. Dedicated
legalization passes then:

1. **Type legalization** — narrows i64 to i32 (or expands to carry chains).
2. **Register legalization** — assigns SGPR vs VGPR, enforces constant bus
   limits, inserts `v_mov_b32` copies as needed.
3. **Instruction selection** — lowers generic ops to concrete machine ops
   (`s_add_u32` / `v_add_u32` / etc.).

Operations with unambiguous lowering (buffer ops, MFMA, control flow, LDS)
emit concrete machine ops directly during translation and skip legalization.

### 4. Optimization Passes

The pass pipeline optimizes the legalized WaveASM IR before register allocation.

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                           Optimization Pipeline                                │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  ┌─────────────────────┐                                                       │
│  │     ScopedCSE       │  Common Subexpression Elimination                     │
│  │                     │  - Uses op traits (ArithmeticOp, MemoryOp)            │
│  │  a = v_add x, y     │  - Eliminates redundant computations                  │
│  │  b = v_add x, y ──► │  - Scope-aware for control flow                       │
│  │  b = a              │                                                       │
│  └──────────┬──────────┘                                                       │
│             ▼                                                                  │
│  ┌─────────────────────┐                                                       │
│  │     Peephole        │  Local optimizations                                  │
│  │                     │  - v_lshlrev + v_add → v_lshl_add                     │
│  │  Strength reduction │  - mul pow2 → shift                                   │
│  │  Identity elimination│  - Identity op elimination                            │
│  │  Instruction fusion │  - MFMA zero-accumulator optimization                 │
│  └──────────┬──────────┘                                                       │
│             ▼                                                                  │
│  ┌─────────────────────┐                                                       │
│  │     Liveness        │  Live range analysis                                  │
│  │                     │  - CFG construction                                   │
│  │  Live intervals     │  - Dataflow analysis                                  │
│  │  for regalloc       │  - Use-def chains                                     │
│  └──────────┬──────────┘                                                       │
│             ▼                                                                  │
│  ┌─────────────────────┐                                                       │
│  │     Ticketing       │  Wait count insertion                                 │
│  │                     │  - s_waitcnt vmcnt/lgkmcnt                            │
│  │  Memory ordering    │  - Coalesces adjacent waits                           │
│  │                     │  - Tracks pending memory ops                          │
│  └──────────┬──────────┘                                                       │
│             ▼                                                                  │
│  ┌─────────────────────┐                                                       │
│  │  HazardMitigation   │  Hardware hazard handling                             │
│  │                     │  - MFMA pipeline hazards                              │
│  │  s_nop insertion    │  - Memory-compute hazards                             │
│  │                     │  - Target-specific rules                              │
│  └─────────────────────┘                                                       │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

### 5. Register Allocation

Linear scan register allocation converts virtual registers to physical registers.

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                        Linear Scan Register Allocation                         │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  Input: WaveASM IR with virtual registers (!waveasm.vreg, !waveasm.sreg)       │
│                                                                                │
│  ┌────────────────────────────────────────────────────────────────────┐        │
│  │                    Live Range Construction                         │        │
│  │                                                                    │        │
│  │    vreg1: ──[═══════════]────────────────                         │        │
│  │    vreg2: ────────[════════════════]─────                         │        │
│  │    vreg3: ──────────────────[═══════════]                         │        │
│  │                                                                    │        │
│  │    Instruction positions (program order)                           │        │
│  └────────────────────────────────────────────────────────────────────┘        │
│                                                                                │
│  ┌────────────────────────────────────────────────────────────────────┐        │
│  │                    Allocation Algorithm                            │        │
│  │                                                                    │        │
│  │  1. Sort intervals by start position                               │        │
│  │  2. For each interval:                                             │        │
│  │     - Expire old intervals (free registers)                        │        │
│  │     - Handle precolored (fixed) registers                          │        │
│  │     - Handle tied operands (MFMA accumulator)                      │        │
│  │     - Allocate from free pool with alignment                       │        │
│  │  3. Rewrite IR with physical registers                             │        │
│  └────────────────────────────────────────────────────────────────────┘        │
│                                                                                │
│  Special Handling:                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  Precolored  │  │    Tied      │  │  Alignment   │  │   Spilling   │       │
│  │              │  │   Operands   │  │              │  │   (future)   │       │
│  │  thread_id   │  │  MFMA acc    │  │  SGPR pairs  │  │              │       │
│  │  block_id    │  │  must match  │  │  for 64-bit  │  │              │       │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘       │
│                                                                                │
│  Output: WaveASM IR with physical registers (!waveasm.pvreg, !waveasm.psreg)   │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

### 6. Assembly Emission

The assembly emitter generates AMDGCN assembly with full HSA metadata.

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                            Assembly Emitter                                    │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ .amdgcn_target "amdgcn-amd-amdhsa--gfx950"                              │   │
│  │                                                                         │   │
│  │ .text                                                                   │   │
│  │ .globl kernel_name                                                      │   │
│  │ .p2align 8                                                              │   │
│  │ kernel_name:                                                            │   │
│  │   ; Prologue - SRD setup                                                │   │
│  │   s_load_dwordx2 s[0:1], s[4:5], 0x0                                    │   │
│  │   ...                                                                   │   │
│  │                                                                         │   │
│  │   ; Main kernel body                                                    │   │
│  │   v_add_u32 v0, v1, v2                                                  │   │
│  │   v_mfma_f32_16x16x16_f16 v[4:7], v[0:1], v[2:3], v[4:7]                │   │
│  │   ...                                                                   │   │
│  │                                                                         │   │
│  │   s_endpgm                                                              │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ .section .rodata                                                        │   │
│  │ .amdhsa_kernel kernel_name                                              │   │
│  │   .amdhsa_group_segment_fixed_size 1024                                 │   │
│  │   .amdhsa_user_sgpr_kernarg_segment_ptr 1                               │   │
│  │   .amdhsa_next_free_vgpr 24                                             │   │
│  │   .amdhsa_next_free_sgpr 16                                             │   │
│  │   .amdhsa_accum_offset 24                                               │   │
│  │   ...                                                                   │   │
│  │ .end_amdhsa_kernel                                                      │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ .amdgpu_metadata                                                        │   │
│  │ ---                                                                     │   │
│  │ amdhsa.kernels:                                                         │   │
│  │   - .name: kernel_name                                                  │   │
│  │     .args:                                                              │   │
│  │       - .name: arg0_ptr                                                 │   │
│  │         .size: 8                                                        │   │
│  │         .value_kind: global_buffer                                      │   │
│  │     .kernarg_segment_size: 24                                           │   │
│  │     ...                                                                 │   │
│  │ .end_amdgpu_metadata                                                    │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                │
└────────────────────────────────────────────────────────────────────────────────┘
```

## Data Flow Through the Compiler

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Example: Vector Add                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  MLIR Input:                                                                    │
│  ┌────────────────────────────────────────────────────────────────────────┐     │
│  │  %tid = gpu.thread_id x                                                │     │
│  │  %a = vector.load %A[%tid] : memref<64xf32>, vector<1xf32>             │     │
│  │  %b = vector.load %B[%tid] : memref<64xf32>, vector<1xf32>             │     │
│  │  %c = arith.addf %a, %b : vector<1xf32>                                │     │
│  │  vector.store %c, %C[%tid] : memref<64xf32>, vector<1xf32>             │     │
│  └────────────────────────────────────────────────────────────────────────┘     │
│                            │                                                    │
│                            ▼                                                    │
│  WaveASM IR (virtual regs):                                                     │
│  ┌────────────────────────────────────────────────────────────────────────┐     │
│  │  %v0 = waveasm.v_mbcnt_lo_u32_b32 -1, 0 : !waveasm.vreg                 │     │
│  │  %v1 = waveasm.v_mbcnt_hi_u32_b32 -1, %v0 : !waveasm.vreg               │     │
│  │  %v2 = waveasm.buffer_load_b32 %srd_a, %v1, 0 : !waveasm.vreg           │     │
│  │  %v3 = waveasm.buffer_load_b32 %srd_b, %v1, 0 : !waveasm.vreg           │     │
│  │  %v4 = waveasm.v_add_f32 %v2, %v3 : !waveasm.vreg                       │     │
│  │  waveasm.buffer_store_b32 %v4, %srd_c, %v1, 0                           │     │
│  └────────────────────────────────────────────────────────────────────────┘     │
│                            │                                                    │
│                            ▼                                                    │
│  WaveASM IR (physical regs):                                                    │
│  ┌────────────────────────────────────────────────────────────────────────┐     │
│  │  %p0 = waveasm.v_mbcnt_lo_u32_b32 -1, 0 : !waveasm.pvreg<0>             │     │
│  │  %p1 = waveasm.v_mbcnt_hi_u32_b32 -1, %p0 : !waveasm.pvreg<1>           │     │
│  │  %p2 = waveasm.buffer_load_b32 %srd_a, %p1, 0 : !waveasm.pvreg<2>       │     │
│  │  %p3 = waveasm.buffer_load_b32 %srd_b, %p1, 0 : !waveasm.pvreg<3>       │     │
│  │  %p4 = waveasm.v_add_f32 %p2, %p3 : !waveasm.pvreg<2>                   │     │
│  │  waveasm.buffer_store_b32 %p4, %srd_c, %p1, 0                           │     │
│  └────────────────────────────────────────────────────────────────────────┘     │
│                            │                                                    │
│                            ▼                                                    │
│  AMDGCN Assembly:                                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐     │
│  │  v_mbcnt_lo_u32_b32 v0, -1, 0                                          │     │
│  │  v_mbcnt_hi_u32_b32 v1, -1, v0                                         │     │
│  │  buffer_load_dword v2, v1, s[0:3], 0 offen                             │     │
│  │  buffer_load_dword v3, v1, s[4:7], 0 offen                             │     │
│  │  s_waitcnt vmcnt(0)                                                    │     │
│  │  v_add_f32 v2, v2, v3                                                  │     │
│  │  buffer_store_dword v2, v1, s[8:11], 0 offen                           │     │
│  └────────────────────────────────────────────────────────────────────────┘     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
wave-asm/
├── include/waveasm/
│   ├── Dialect/                    # TableGen definitions and C++ headers
│   │   ├── WaveASMOps.td           # Operation definitions
│   │   ├── WaveASMTypes.td         # Type definitions (vreg, sreg, pvreg, psreg)
│   │   └── WaveASMAttrs.td         # Attribute definitions
│   ├── Transforms/                 # Pass declarations
│   │   ├── Passes.td               # Pass registration
│   │   └── TranslateFromMLIR.h     # Translation context
│   └── Target/AMDGCN/              # Target-specific headers
│       └── InstructionInfo.h       # Instruction metadata
│
├── lib/
│   ├── Dialect/                    # Dialect implementation
│   │   ├── WaveASMDialect.cpp      # Dialect registration
│   │   ├── WaveASMOps.cpp          # Operation implementations
│   │   └── WaveASMTypes.cpp        # Type implementations
│   ├── Transforms/                 # Pass implementations
│   │   ├── TranslateFromMLIR.cpp   # MLIR → WaveASM translation (2500+ lines)
│   │   ├── Liveness.cpp            # Live range analysis
│   │   ├── LinearScanPass.cpp      # Register allocation driver
│   │   ├── LinearScanRegAlloc.cpp  # Allocation algorithm
│   │   ├── Ticketing.cpp           # Wait count insertion
│   │   ├── HazardMitigation.cpp    # Hardware hazard handling
│   │   ├── ScopedCSE.cpp           # Common subexpression elimination
│   │   ├── Peephole.cpp            # Peephole optimizations
│   │   └── AssemblyEmitter.cpp     # AMDGCN assembly output
│   └── Target/AMDGCN/              # Target implementations
│       ├── GFX942Target.cpp        # MI300 support
│       ├── GFX950Target.cpp        # MI350 support
│       ├── GFX1250Target.cpp       # RDNA4 support
│       └── InstructionInfo.cpp     # Instruction metadata
│
├── tools/
│   └── waveasm-translate/          # CLI tool
│       └── waveasm-translate.cpp
│
└── test/
    ├── lit/                        # FileCheck-based unit tests
    └── e2e/                        # End-to-end GPU tests
        └── test_asm_backend_e2e.py
```

## Target Support

| Target | Architecture | GPU | Status |
|--------|-------------|-----|--------|
| gfx942 | CDNA3 | MI300 | Supported |
| gfx950 | CDNA4 | MI350 | Supported |
| gfx1250 | RDNA4 | RX 9000 | Experimental |

## Key Design Decisions

1. **Pure SSA IR**: Each GPU instruction is an MLIR operation with SSA values, enabling standard MLIR analyses and transformations.

2. **Two-Phase Register Allocation**: Virtual registers (vreg/sreg) are used during optimization, then converted to physical registers (pvreg/psreg) during allocation.

3. **Op Traits for CSE**: Operations use MLIR traits (ArithmeticOp, MemoryOp) to determine CSE eligibility, rather than hardcoded lists.

4. **Target-Specific Passes**: Hazard mitigation and instruction selection use target information for architecture-specific behavior.

5. **Full HSA Metadata**: Assembly output includes complete `.amdhsa_kernel` and `.amdgpu_metadata` sections for direct HSACO generation.
