# The Wave Assembly Backend

A backend compiler for the [Wave DSL](https://github.com/iree-org/wave), translating high-level MLIR representations into optimized AMDGCN assembly code.

## Overview

The Wave Assembly Backend takes MLIR input (from the Wave DSL compiler frontend) and produces:
- **AMDGCN assembly files** (`.s`) ready for assembly with `clang`
- **Optimized register allocation** using linear scan allocation
- **Target-specific code generation** for AMD GPUs (gfx942, gfx950, gfx1250)

### Input

The backend accepts MLIR modules containing:
- `gpu.func` kernels with `gpu`, `arith`, `vector`, `memref`, `scf`, and `amdgpu` dialect operations
- Pre-lowered WaveASM IR (`waveasm.program` with virtual/physical register operations)

### Output

- **WaveASM IR** - Intermediate representation with virtual registers
- **AMDGCN assembly** (`.s` files) compatible with ROCm toolchain
- **HSACO GPU binaries** when linked with `clang` and `lld`

## Quick Start

```bash
# Build
mkdir build && cd build
cmake -G Ninja .. -DMLIR_DIR=/path/to/llvm/build/lib/cmake/mlir
ninja

# Translate MLIR to WaveASM IR
./tools/waveasm-translate/waveasm-translate input.mlir -o output.mlir

# Run tests
ninja check-waveasm
```

## Key Features

- **Pure SSA IR** with individual ops for each GPU instruction
- **Virtual and physical register types** for VGPRs, SGPRs, and special registers
- **Linear scan register allocation** with support for tied operands (MFMA)
- **MLIR translation** from `gpu`, `arith`, `vector`, `memref`, `scf`, `amdgpu` dialects
- **Direct assembly emission** with proper metadata for HSACO generation

## Directory Structure

```
waveasm/
├── include/waveasm/
│   ├── Dialect/           # TableGen definitions and headers
│   ├── Transforms/        # Pass definitions
│   └── Target/AMDGCN/     # Target-specific headers
├── lib/
│   ├── Dialect/           # Dialect implementation
│   ├── Transforms/        # Pass implementations
│   └── Target/AMDGCN/     # Target implementations
├── tools/                 # CLI tools
└── test/                  # Tests
```

## Building

### Prerequisites

- CMake 3.20+
- Ninja
- C++17 compiler (clang++ recommended)

### LLVM/MLIR Dependency

WaveASM shares the LLVM version pinned in `water/llvm-sha.txt`. See the
top-level README for instructions on building the shared LLVM/MLIR install.

### Standalone Build

Point CMake at the shared LLVM install:

```bash
mkdir build && cd build
cmake -G Ninja ../waveasm \
  -DLLVM_DIR=/path/to/llvm-install/lib/cmake/llvm \
  -DMLIR_DIR=/path/to/llvm-install/lib/cmake/mlir
ninja
```

### pip Build

```bash
WAVE_BUILD_WAVEASM=1 WAVE_LLVM_DIR=/path/to/llvm-install uv pip install -e .
```

### Verify Build

```bash
./build/bin/waveasm-translate --help
```

## Dialect Design

### Pure SSA IR

The dialect uses pure SSA form where each GPU instruction is its own MLIR op.
Values flow through def-use chains, enabling standard MLIR analyses and transformations.

**Virtual Registers (Pre-Allocation)**
```mlir
%v0 = waveasm.precolored.vreg 0 : !waveasm.vreg
%v1 = waveasm.precolored.vreg 1 : !waveasm.vreg
%sum = waveasm.v_add_u32 %v0, %v1 : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg
```

**Physical Registers (Post-Allocation)**
```mlir
%p0 = waveasm.precolored.vreg 5 : !waveasm.pvreg<5>
%p1 = waveasm.precolored.vreg 6 : !waveasm.pvreg<6>
%result = waveasm.v_add_u32 %p0, %p1 : !waveasm.pvreg<5>, !waveasm.pvreg<6> -> !waveasm.vreg
```

### Types

| Type | Description |
|------|-------------|
| `!waveasm.vreg` | Virtual VGPR (default size 1) |
| `!waveasm.vreg<size>` | Virtual VGPR with explicit size |
| `!waveasm.sreg` | Virtual SGPR (default size 1) |
| `!waveasm.sreg<size, align>` | Virtual SGPR with size and alignment |
| `!waveasm.pvreg<index>` | Physical VGPR at specific index |
| `!waveasm.pvreg<index, size>` | Physical VGPR range |
| `!waveasm.psreg<index>` | Physical SGPR at specific index |
| `!waveasm.psreg<index, size>` | Physical SGPR range |
| `!waveasm.imm<value>` | Immediate constant |

### Operations

The dialect provides ~300 individual instruction ops organized by category:

**VALU (Vector ALU)**
| Operation | Description |
|-----------|-------------|
| `waveasm.v_add_u32` | Vector add unsigned 32-bit |
| `waveasm.v_sub_u32` | Vector subtract unsigned 32-bit |
| `waveasm.v_mul_lo_u32` | Vector multiply low 32-bit |
| `waveasm.v_mov_b32` | Vector move 32-bit |
| `waveasm.v_fma_f32` | Vector fused multiply-add f32 |
| ... | And many more |

**SALU (Scalar ALU)**
| Operation | Description |
|-----------|-------------|
| `waveasm.s_add_u32` | Scalar add unsigned 32-bit |
| `waveasm.s_mul_i32` | Scalar multiply signed 32-bit |
| `waveasm.s_mov_b32` | Scalar move 32-bit |
| ... | And many more |

**MFMA (Matrix Multiply-Accumulate)**
| Operation | Description |
|-----------|-------------|
| `waveasm.v_mfma_f32_16x16x16_f16` | 16x16x16 FP16 matrix multiply |
| `waveasm.v_mfma_f32_32x32x8_f16` | 32x32x8 FP16 matrix multiply |
| ... | And many more |

**Memory Operations**
| Operation | Description |
|-----------|-------------|
| `waveasm.global_load_b32` | Global memory load 32-bit |
| `waveasm.global_store_b32` | Global memory store 32-bit |
| `waveasm.ds_read_b32` | LDS read 32-bit |
| `waveasm.ds_write_b32` | LDS write 32-bit |
| `waveasm.s_load_b32` | Scalar memory load 32-bit |
| ... | And many more |

**Control Flow**
| Operation | Description |
|-----------|-------------|
| `waveasm.s_branch` | Unconditional branch |
| `waveasm.s_cbranch_scc0` | Branch if SCC == 0 |
| `waveasm.s_cbranch_vccz` | Branch if VCC == 0 |
| `waveasm.s_endpgm` | End program |
| `waveasm.s_barrier` | Barrier synchronization |
| `waveasm.s_waitcnt` | Wait for memory operations |
| `waveasm.label` | Branch target label |

**Utility**
| Operation | Description |
|-----------|-------------|
| `waveasm.comment` | Assembly comment |
| `waveasm.raw` | Raw assembly passthrough |
| `waveasm.constant` | Immediate constant definition |
| `waveasm.precolored.vreg` | Define precolored VGPR |
| `waveasm.precolored.sreg` | Define precolored SGPR |

### Attributes

| Attribute | Description |
|-----------|-------------|
| `#waveasm.abi<tid = 0, kernarg = 0>` | ABI bindings (precolored regs) |
| `#waveasm.target<#waveasm.gfx942, 5>` | Target architecture and wave size |

## Example

```mlir
waveasm.program @my_kernel
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {vgprs = 256 : i64, sgprs = 104 : i64} {

  // Define precolored ABI registers
  %tid = waveasm.precolored.vreg 0 : !waveasm.vreg

  // Arithmetic operations - pure SSA
  %doubled = waveasm.v_add_u32 %tid, %tid : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg

  // Labels for control flow
  waveasm.label @loop_start

  // Comments
  waveasm.comment "Loop body"

  // Conditional branch
  waveasm.s_cbranch_scc1 @loop_start

  // End program
  waveasm.s_endpgm
}
```

## GEMM Example

Here is a complete GEMM kernel (16x16x16 tile, f16 inputs, f32 accumulator) in the WaveASM dialect:

```mlir
waveasm.program @gemm_16x16x16
  target = #waveasm.target<#waveasm.gfx942, 5>
  abi = #waveasm.abi<tid = 0, kernarg = 0>
  attributes {lds_size = 1024 : i64, vgprs = 24 : i64, sgprs = 32 : i64} {

  // === Precolored ABI registers ===
  %kernarg_ptr = waveasm.precolored.sreg 0, 2 : !waveasm.psreg<0, 2>  // s[0:1]
  %wgid_x = waveasm.precolored.sreg 8 : !waveasm.psreg<8>             // s8
  %wgid_y = waveasm.precolored.sreg 9 : !waveasm.psreg<9>             // s9

  // === Load kernel arguments (A, B, C pointers) ===
  %arg0 = waveasm.s_load_dwordx2 %kernarg_ptr, 0 : !waveasm.psreg<0, 2> -> !waveasm.sreg<2>
  %arg1 = waveasm.s_load_dwordx2 %kernarg_ptr, 8 : !waveasm.psreg<0, 2> -> !waveasm.sreg<2>
  %arg2 = waveasm.s_load_dwordx2 %kernarg_ptr, 16 : !waveasm.psreg<0, 2> -> !waveasm.sreg<2>
  waveasm.s_waitcnt 0, 0  // lgkmcnt(0)

  // === Build buffer resource descriptors (SRDs) ===
  // SRD for matrix A: s[12:15]
  %srd_a = waveasm.build_srd %arg0, 0x7FFFFFFC, 0x20000 : !waveasm.sreg<2> -> !waveasm.psreg<12, 4>
  // SRD for matrix B: s[16:19]
  %srd_b = waveasm.build_srd %arg1, 0x7FFFFFFC, 0x20000 : !waveasm.sreg<2> -> !waveasm.psreg<16, 4>
  // SRD for matrix C: s[20:23]
  %srd_c = waveasm.build_srd %arg2, 0x4000, 0x20000 : !waveasm.sreg<2> -> !waveasm.psreg<20, 4>

  // === Compute thread/lane IDs ===
  %c_neg1 = waveasm.constant -1 : !waveasm.imm<-1>
  %c0 = waveasm.constant 0 : !waveasm.imm<0>
  %lane_lo = waveasm.v_mbcnt_lo_u32_b32 %c_neg1, %c0 : !waveasm.imm<-1>, !waveasm.imm<0> -> !waveasm.vreg
  %lane_id = waveasm.v_mbcnt_hi_u32_b32 %c_neg1, %lane_lo : !waveasm.imm<-1>, !waveasm.vreg -> !waveasm.vreg

  // === Initialize accumulators to zero (4 VGPRs for 16x16 MFMA output) ===
  %acc0 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg
  %acc1 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg
  %acc2 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg
  %acc3 = waveasm.v_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.vreg

  // === Initialize loop counter ===
  %loop_cnt = waveasm.s_mov_b32 %c0 : !waveasm.imm<0> -> !waveasm.psreg<24>
  %c4 = waveasm.constant 4 : !waveasm.imm<4>
  %c1 = waveasm.constant 1 : !waveasm.imm<1>

  // === Main loop: iterate over K dimension ===
  waveasm.label @L_loop_k

  // Barrier before loading new tile
  waveasm.s_waitcnt 0, 0
  waveasm.s_barrier

  // --- Gather A tile from global memory to LDS ---
  // Compute global address for A tile
  %c5 = waveasm.constant 5 : !waveasm.imm<5>
  %k_offset = waveasm.v_lshlrev_b32 %c5, %loop_cnt : !waveasm.imm<5>, !waveasm.psreg<24> -> !waveasm.vreg
  %a_addr = waveasm.v_add_u32 %k_offset, %lane_id : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg

  // Set LDS destination offset in M0 and gather
  %c512 = waveasm.constant 512 : !waveasm.imm<512>
  waveasm.s_mov_b32_m0 %c512 : !waveasm.imm<512>
  waveasm.buffer_load_dword_lds %a_addr, %srd_a : !waveasm.vreg, !waveasm.psreg<12, 4>

  // --- Gather B tile from global memory to LDS ---
  %b_addr = waveasm.v_add_u32 %k_offset, %lane_id : !waveasm.vreg, !waveasm.vreg -> !waveasm.vreg
  waveasm.s_mov_b32_m0 %c0 : !waveasm.imm<0>
  waveasm.buffer_load_dword_lds %b_addr, %srd_b : !waveasm.vreg, !waveasm.psreg<16, 4>

  // Wait for LDS writes and synchronize
  waveasm.s_waitcnt 0, 0
  waveasm.s_barrier

  // --- Read tiles from LDS ---
  // Compute LDS read addresses
  %c15 = waveasm.constant 15 : !waveasm.imm<15>
  %lane_mod16 = waveasm.v_and_b32 %lane_id, %c15 : !waveasm.vreg, !waveasm.imm<15> -> !waveasm.vreg
  %lds_addr = waveasm.v_lshlrev_b32 %c5, %lane_mod16 : !waveasm.imm<5>, !waveasm.vreg -> !waveasm.vreg

  // Load A tile (8 bytes = 4 f16 elements)
  %a_tile = waveasm.ds_read_b64 %lds_addr, 512 : !waveasm.vreg -> !waveasm.vreg<2>

  // Load B tile (8 bytes = 4 f16 elements)
  %b_tile = waveasm.ds_read_b64 %lds_addr, 0 : !waveasm.vreg -> !waveasm.vreg<2>

  waveasm.s_waitcnt 0, 0  // Wait for LDS reads

  // --- Execute MFMA instruction ---
  // v_mfma_f32_16x16x16_f16: A[16x16] * B[16x16] -> C[16x16]
  // Accumulates into 4 VGPRs (16 floats across 64 lanes)
  %mfma_out:4 = waveasm.v_mfma_f32_16x16x16_f16 %a_tile, %b_tile, %acc0, %acc1, %acc2, %acc3
    : !waveasm.vreg<2>, !waveasm.vreg<2>, !waveasm.vreg, !waveasm.vreg, !waveasm.vreg, !waveasm.vreg
    -> !waveasm.vreg, !waveasm.vreg, !waveasm.vreg, !waveasm.vreg

  // --- Loop latch ---
  %loop_cnt_next = waveasm.s_add_u32 %loop_cnt, %c1 : !waveasm.psreg<24>, !waveasm.imm<1> -> !waveasm.psreg<24>
  waveasm.s_cmp_lt_u32 %loop_cnt_next, %c4 : !waveasm.psreg<24>, !waveasm.imm<4>
  waveasm.s_cbranch_scc1 @L_loop_k

  waveasm.label @L_loop_k_end

  // === Store results to global memory ===
  // Compute output address
  %c8 = waveasm.constant 8 : !waveasm.imm<8>
  %out_addr = waveasm.v_lshlrev_b32 %c8, %lane_id : !waveasm.imm<8>, !waveasm.vreg -> !waveasm.vreg

  // Store 4 result VGPRs with offsets (256 bytes apart for coalescing)
  %c256 = waveasm.constant 256 : !waveasm.imm<256>
  waveasm.buffer_store_dword %mfma_out#0, %out_addr, %srd_c, 0 : !waveasm.vreg, !waveasm.vreg, !waveasm.psreg<20, 4>
  waveasm.buffer_store_dword %mfma_out#1, %out_addr, %srd_c, 256 : !waveasm.vreg, !waveasm.vreg, !waveasm.psreg<20, 4>
  waveasm.buffer_store_dword %mfma_out#2, %out_addr, %srd_c, 512 : !waveasm.vreg, !waveasm.vreg, !waveasm.psreg<20, 4>
  waveasm.buffer_store_dword %mfma_out#3, %out_addr, %srd_c, 768 : !waveasm.vreg, !waveasm.vreg, !waveasm.psreg<20, 4>

  waveasm.s_endpgm
}
```

### Key GEMM Components

| Component | WaveASM Operations |
|-----------|-------------------|
| **Kernel arguments** | `s_load_dwordx2` to load A, B, C pointers |
| **Buffer descriptors** | `build_srd` to create 4-SGPR resource descriptors |
| **Lane ID** | `v_mbcnt_lo/hi_u32_b32` for thread identification |
| **Global→LDS gather** | `buffer_load_dword_lds` with M0 for LDS offset |
| **LDS reads** | `ds_read_b64` with offset modifiers |
| **Matrix multiply** | `v_mfma_f32_16x16x16_f16` (4 VGPR accumulators) |
| **Synchronization** | `s_barrier`, `s_waitcnt` |
| **Output stores** | `buffer_store_dword` with byte offsets |

### Generated Assembly

The above WaveASM IR compiles to AMDGCN assembly like:

```asm
gemm_16x16x16:
  s_load_dwordx2 s[2:3], s[0:1], 0x0
  s_load_dwordx2 s[4:5], s[0:1], 0x8
  s_load_dwordx2 s[6:7], s[0:1], 0x10
  s_waitcnt lgkmcnt(0)
  ; ... SRD setup ...

  v_mov_b32 v4, 0
  v_mov_b32 v5, 0
  v_mov_b32 v6, 0
  v_mov_b32 v7, 0        ; Initialize accumulators

L_loop_k:
  s_barrier
  s_mov_b32 m0, 512
  buffer_load_dword v8, s[12:15], 0 offen lds   ; A -> LDS
  s_mov_b32 m0, 0
  buffer_load_dword v9, s[16:19], 0 offen lds   ; B -> LDS
  s_waitcnt vmcnt(0)
  s_barrier

  ds_read_b64 v[10:11], v12                     ; Load A tile
  ds_read_b64 v[12:13], v12 offset:512          ; Load B tile
  s_waitcnt lgkmcnt(0)

  v_mfma_f32_16x16x16_f16 v[4:7], v[12:13], v[10:11], v[4:7]

  s_add_u32 s24, s24, 1
  s_cmp_lt_u32 s24, 4
  s_cbranch_scc1 L_loop_k

  buffer_store_dword v4, v14, s[20:23], 0 offen
  buffer_store_dword v5, v14, s[20:23], 0 offen offset:256
  buffer_store_dword v6, v14, s[20:23], 0 offen offset:512
  buffer_store_dword v7, v14, s[20:23], 0 offen offset:768
  s_endpgm
```

## Pass Pipeline

```
waveasm-translate-from-mlir  # Convert upstream MLIR to WaveASM IR
waveasm-liveness             # Compute live ranges
waveasm-linear-scan          # Register allocation
waveasm-emit-assembly        # Generate .s output
```

## Supported Operations

The translator currently supports:
- **GPU dialect**: `gpu.func`, `gpu.thread_id`, `gpu.block_id`, `gpu.barrier`, `gpu.return`
- **Arith dialect**: `arith.constant`, `arith.addi`, `arith.muli`, `arith.index_cast`
- **Vector dialect**: `vector.load`, `vector.store`
- **AMDGPU dialect**: `amdgpu.mfma`, `amdgpu.lds_barrier`
- **SCF dialect**: `scf.for`, `scf.yield`
- **Affine dialect**: `affine.apply`

## License

Apache 2.0 with LLVM Exceptions
