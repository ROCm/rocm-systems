# AMD GPU Mega Kernel Unit Test

A comprehensive unit test kernel that exercises most of the instructions and features available on AMD GPUs including Instinct (MI250/MI300/MI350), Radeon (RX 9060 / RX 9070 families, gfx120x), and Strix/Strix Halo/Krackan (RDNA 3.5, gfx115x) APU iGPUs.

## Supported GPU Architectures

*GPU/architecture tables below are partly **informational** (what silicon markets); **what this sample runs** is only what the **Feature Compatibility Matrix** and following “Features Tested” sections describe.*

### CDNA (Compute/Datacenter)

| GPU Series | Architecture | GFXIP | Key Features |
|------------|--------------|-------|--------------|
| **MI250/MI250X** | CDNA2 | gfx90a | FP32/FP64/FP16/BF16, HW FP64 atomics, MFMA, DOT4/DOT8 |
| **MI300/MI300X/MI325X** | CDNA3 | gfx942 | All MI250 + FP8/BF8 (FNUZ-style cvt), TF32, Sparse MFMA |
| **MI350X/MI355X** | CDNA4 | gfx950 | All MI300 + FP6/FP4, OCP FP8/BF8 where applicable, Async LDS, Enhanced WMMA |

### RDNA4 (Consumer)

| GPU Series | Architecture | GFXIP | Key Features |
|------------|--------------|-------|--------------|
| **RX 9070 XT/9070** | RDNA4 | gfx1201 | FP8/BF8 OCP, Async LDS, Wave32, WMMA (gfx12 path) |
| **RX 9060 series** | RDNA4 | gfx1200 | Same as gfx1201 with different CU count |

### RDNA 3.5+ APU iGPU (gfx115x-class)

| GPU Series | Architecture | GFXIP | Key Features |
|------------|--------------|-------|--------------|
| **Strix Point** | RDNA 3.5 | gfx1150 | 16 CUs, WMMA, Wave32, Async LDS |
| **Strix Halo** | RDNA 3.5 | gfx1151 | Up to 40 CUs, WMMA, Wave32, Async LDS |
| **Krackan** | RDNA 3.5 | gfx1152 | 32MB Infinity Cache, same as gfx1150 |


## Feature Compatibility Matrix

This table lists **what this sample tries to run** on each GFXIP, plus a few hardware facts that are **not** individually exercised here (no standalone fixed-function graphics or branded accelerator microbenchmarks in this kernel). MFMA/WMMA builtins use **non-zero operands** and pass only if the accumulator fragment has **at least one non-zero lane** (FP paths also require **finite** lanes)—smoke validation, not golden-matrix or full ISA conformance. Dual-issue is **BYPASSED** in the report (patterns only). Async LDS follows `__has_builtin(__builtin_amdgcn_global_load_async_to_lds_b32)` in code (gfx942 is treated as unsupported in this sample, matching `mega_kernel_device_arch.h`).

| Feature | MI250 (gfx90a) | MI300 (gfx942) | MI350 (gfx950) | Strix/Krackan (gfx115x) | RX 9070 XT (gfx1201) |
|---------|:--------------:|:--------------:|:--------------:|:-----------------------:|:--------------------:|
| FP32 Arithmetic | ✅ | ✅ | ✅ | ✅ | ✅ |
| FP64 Arithmetic | ✅ | ✅ | ✅ | ⚠️ (limited) | ⚠️ (limited) |
| FP16/BF16 Arithmetic | ✅ | ✅ | ✅ | ✅ | ✅ |
| INT32/INT64 Operations | ✅ | ✅ | ✅ | ✅ | ✅ |
| HW FP64 Atomics | ✅ | ✅ | ✅ | ❌ | ❌ |
| FP8/BF8 Conversions | ❌ | ✅ (FNUZ) | ✅ (OCP) | ❌ | ✅ (OCP) |
| FP6/FP4 Conversions | ❌ | ❌ | ✅ | ❌ | ❌ |
| DOT4/DOT8 Products | ✅ | ✅ | ✅ | ✅ | ✅ (software) |
| Warp/Wave Ops | ✅ (Wave64) | ✅ (Wave64) | ✅ (Wave64) | ✅ (Wave32) | ✅ (Wave32) |
| LDS Operations | ✅ | ✅ | ✅ | ✅ | ✅ |
| Async LDS (this sample) | ❌ | ❌ | ✅ | ✅ | ✅ |
| Dual-Issue VALU (this sample) | — | BYPASS | BYPASS | BYPASS | — |
| MFMA Matrix Ops | ✅ | ✅ | ✅ | ❌ | ❌ |
| WMMA Matrix Ops | ❌ | ❌ | ❌ | ✅ (gfx12) | ✅ (gfx12) |
| Packed FP16 Atomics | ✅ | ✅ | ✅ | ✅ | ✅ |
| Packed BF16 Atomics | ❌ | ✅ | ✅ | ✅ | ✅ |
| VMEM Inline ASM | ✅ | ✅ | ✅ | ✅ | ✅ |

⚠️ = Limited support / may vary by ROCm version. **BYPASS** = kernel emits instruction patterns but the host report marks the category bypassed (not a correctness proof). Fixed-function blocks not driven by this VALU/WMMA/TEX-focused kernel are **not exercised here**.

## Features Tested

### Test Category Summary

| # | Category | Tests | Description |
|---|----------|-------|-------------|
| 1 | Warp/Wave Operations | 4 | Shuffle, ballot, permute, reduce |
| 2 | Data Type Conversions | 4 | FP8, BF8, FP16, BF16 conversions |
| 3 | Arithmetic Operations | 4 | FP32, FP64, INT32, INT64 |
| 4 | Basic Atomic Operations | 6 | FP32/FP64 add, min/max, integer ops, LDS |
| 5 | Transcendental Functions | 3 | Trig, exp/log, rcp/rsq |
| 6 | Memory Operations | 2 | Global load/store |
| 7 | DOT Products | 2 | DOT4, DOT8 |
| 8 | MFMA Operations | 5 | FP32, FP64, FP16, BF16, INT8 matrix ops |
| 9 | Async LDS | 2 | Async load/store to LDS |
| 11 | Dual-Issue VALU | 2 | FP64+FP32, INT+FP32 patterns |
| 13 | WMMA Operations | 3 | FP16, BF16, INT8 matrix ops (RDNA4) |
| 14 | VMEM Operations | 5 | Flat, global, buffer, scratch, LDS (inline ASM) |
| 15 | Atomic Operations (ASM) | 10 | Global/flat/DS int/float, packed, CAS (inline ASM) |

### Warp/Wave Operations
- Warp shuffle (`__shfl`, `__shfl_xor`, `__shfl_up`, `__shfl_down`)
- Warp ballot (`__ballot`, `__any`, `__all`)
- DS permute operations (`ds_bpermute`, `ds_permute`)
- Warp-level reductions

### Floating Point Operations
- **FP8/BF8 Conversions**
  - `__builtin_amdgcn_cvt_scalef32_pk_*` (gfx950 specific)
- FP16/BF16 operations and packed arithmetic
- FP32/FP64 arithmetic with FMA

### Integer Operations
- INT32 arithmetic and bit operations
- INT64 arithmetic
- Population count and leading zeros

### Basic Atomic Operations (HIP API)
- Atomic Add FP32 (`atomicAdd`)
- Atomic Min/Max FP32 (via `atomicMin`/`atomicMax` with float-to-uint cast)
- Integer atomics: add, sub, and, or, xor, min, max
- LDS atomics

### Transcendental Functions
- sin, cos, tan, tanh
- exp, log, log2, log10
- sqrt, rsqrt, rcp
- pow, cbrt

### Memory Operations
- Global memory load/store (coalesced)
- LDS (shared memory) operations
- Memory barriers and fences

---

## Inline Assembly Tests (Architecture-Specific)

The following sections document the inline assembly tests that verify hardware instruction paths
directly, using architecture-specific syntax for CDNA (gfx9x) and RDNA4 (gfx12).

### VMEM (Vector Memory) Operations - Comprehensive Inline ASM Tests

This test exercises all memory instruction types using inline assembly to verify
the hardware memory paths are functioning correctly.

**Memory Instruction Categories:**

| Category | CDNA Syntax (gfx9x) | RDNA4 Syntax (gfx12) | Description |
|----------|---------------------|----------------------|-------------|
| **Flat** | `flat_load_dword` | `flat_load_b32` | Unified addressing (global/LDS/scratch) |
| **Global** | `global_load_dword` | `global_load_b32` | Direct global memory access |
| **Buffer** | `buffer_load_dword` | `buffer_load_b32` | Descriptor-based memory access |
| **Scratch** | `scratch_load_dword` | `scratch_load_b32` | Private per-thread stack |
| **LDS** | `ds_read_b32` | `ds_read_b32` | Shared memory (same on all archs) |

**Data Widths Supported:**

| Suffix | Size | CDNA Name | RDNA4 Name |
|--------|------|-----------|------------|
| b8/ubyte | 8-bit | `*_ubyte` | `*_b8` |
| b16/ushort | 16-bit | `*_ushort` | `*_b16` |
| b32/dword | 32-bit | `*_dword` | `*_b32` |
| b64/dwordx2 | 64-bit | `*_dwordx2` | `*_b64` |
| b128/dwordx4 | 128-bit | `*_dwordx4` | `*_b128` |

**Flat Memory Instructions:**
```asm
; CDNA (gfx90a/gfx942/gfx950)
flat_load_dword    vdst, vaddr [offset:N] [glc] [slc]
flat_store_dword   vaddr, vdata [offset:N] [glc] [slc]

; RDNA4 (gfx1200/gfx1201/)
flat_load_b32      vdst, vaddr [offset:N]
flat_store_b32     vaddr, vdata [offset:N]
```

**Global Memory Instructions:**
```asm
; CDNA (gfx90a/gfx942/gfx950)
global_load_dword    vdst, vaddr, saddr [offset:N]
global_store_dword   vaddr, vdata, saddr [offset:N]

; RDNA4 (gfx1200/gfx1201/)
global_load_b32      vdst, vaddr, saddr [offset:N]
global_store_b32     vaddr, vdata, saddr [offset:N]
```

**LDS (DS) Instructions:**
```asm
; Same syntax on all architectures
ds_read_b32     vdst, vaddr [offset:N]
ds_write_b32    vaddr, vdata [offset:N]
ds_read2_b32    vdst0:vdst1, vaddr offset0:N offset1:M
ds_write2_b32   vaddr, vdata0, vdata1 offset0:N offset1:M
```

**Architecture Support:**

| Instruction Type | gfx90a | gfx942 | gfx950 | gfx1150/1151 | gfx1200/1201 |  |
|------------------|:------:|:------:|:------:|:------------:|:-------:|
| flat_load/store | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| global_load/store | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| buffer_load/store | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| scratch_load/store | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| ds_read/write | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| global_load_async | ❌ | ⚠️ | ✅ | ✅ | ✅ | ✅ |
| tensor_load_to_lds | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ |

### Atomic Operations - Comprehensive Inline ASM Tests

This test exercises all atomic instruction types using inline assembly to verify
the hardware atomic paths are functioning correctly.

**Atomic Instruction Categories:**

| Category | Instructions | Description |
|----------|--------------|-------------|
| **Global INT** | `global_atomic_add/sub/and/or/xor/min/max/swap` | Integer atomics on global memory |
| **Global FP32** | `global_atomic_add_f32` | FP32 atomic add |
| **Global FP64** | `global_atomic_add_f64/min_f64/max_f64` | FP64 atomics (CDNA/ only) |
| **Flat INT** | `flat_atomic_add/sub/min/max` | Integer atomics (unified addressing) |
| **DS INT** | `ds_add_u32/sub_u32/min_i32/max_u32` | LDS integer atomics |
| **DS FP32** | `ds_add_f32/min_f32/max_f32` | LDS FP32 atomics |
| **Packed FP16** | `global_atomic_pk_add_f16` | Add two FP16 values atomically |
| **Packed BF16** | `global_atomic_pk_add_bf16` | Add two BF16 values atomically |
| **CAS** | `global_atomic_cmpswap` | Compare-and-swap |

**Naming Convention Changes (RDNA4/gfx12):**

| CDNA (gfx9x) | RDNA4 (gfx12) |
|--------------|---------------|
| `global_atomic_add` | `global_atomic_add_u32` |
| `global_atomic_smax` | `global_atomic_max_i32` |
| `flat_atomic_add` | `flat_atomic_add_u32` |
| `global_atomic_cmpswap` | `global_atomic_cmpswap_b32` |
| Return flag: `glc` | Return flag: `th:TH_ATOMIC_RETURN` |

**Architecture Support:**

| Atomic Type | gfx90a | gfx942 | gfx950 | gfx1150/51 | gfx1200/01 |  |
|-------------|:------:|:------:|:------:|:----------:|:-------:|
| INT32/INT64 atomics | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| FP32 atomic add | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| FP64 atomic add/min/max | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ |
| Packed FP16 atomic add | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Packed BF16 atomic add | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ |
| DS FP32 atomics | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| DS FP64 atomics | ✅ | ✅ | ✅ | ❌ | ❌ | ✅ |
| CAS operations | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

**Inline Assembly Examples:**

```asm
; CDNA - Global atomic add (integer with return)
global_atomic_add %0, %1, %2, off glc
s_waitcnt vmcnt(0)

; RDNA4 - Global atomic add (integer with return)
global_atomic_add_u32 %0, %1, %2, off th:TH_ATOMIC_RETURN
s_waitcnt vmcnt(0)

; DS atomic add (same on all architectures)
ds_add_rtn_u32 %0, %1, %2
s_waitcnt lgkmcnt(0)

; Packed FP16 atomic add
global_atomic_pk_add_f16 %0, %1, %2, off glc  ; CDNA
global_atomic_pk_add_f16 %0, %1, %2, off th:TH_ATOMIC_RETURN  ; RDNA4
```

### Architecture Detection and Conditional Compilation

The inline ASM tests use preprocessor macros to detect the target architecture and use
the correct instruction syntax:

```cpp
#if defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__)
    // CDNA syntax: dword suffix, glc for return
    asm volatile("global_atomic_add %0, %1, %2, off glc" : "=v"(old) : "v"(addr), "v"(val) : "memory");

#elif defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1200__) || defined(__gfx1201__)
    // RDNA 3.5 APU / RDNA4 (GFX12-style): explicit type suffix, TH_ATOMIC_RETURN for return
    asm volatile("global_atomic_add_u32 %0, %1, %2, off th:TH_ATOMIC_RETURN" : "=v"(old) : "v"(addr), "v"(val) : "memory");
#endif
```

**Architecture Macros:**
| Macro | Architecture | GPU |
|-------|--------------|-----|
| `__gfx90a__` | CDNA2 | MI250/MI250X |
| `__gfx942__` | CDNA3 | MI300/MI300X/MI325X |
| `__gfx950__` | CDNA4 | MI350X/MI355X |
| `__gfx1150__` | RDNA 3.5 | Strix Point |
| `__gfx1151__` | RDNA 3.5 | Strix Halo |
| `__gfx1152__` | RDNA 3.5 | Krackan |
| `__gfx1200__` | RDNA4 | RX 9060 series |
| `__gfx1201__` | RDNA4 | RX 9070 XT/9070 |

Note: Not test on MI308X and MI300A yet
---


### Dual-Issue VALU (gfx942+)

**Same-Wave Co-Issue (VOPD/VOPD3)** - Available on MI300, MI350 :
- Uses VOPD encoding to execute 2 operations from the same wave in parallel
- X instruction uses Core MACC, Y instruction uses Side MACC
- VOPD3 extends this with 3 source operands per instruction

**Cross-Wave Dual-Issue** - MI350 (gfx950):
MI350 can issue 2 VALU instructions from different waves per cycle under certain conditions:
- Both waves must be Wave32
- Both must be single-cycle instructions
- One uses DP-MACC (FP64), other uses Core/Side-MACC (FP32/FP16)
- Neither uses DPP, Transcendentals, or WMMA

**Dual-Issue Eligible Patterns:**
- FP64 operations + FP32 operations (from different waves)
- Integer operations + FP32 operations

**Dual-Issue Blockers:**
- VOPD/VOPD3 instructions
- DPP instructions
- WMMA/XDL/Matrix instructions
- Transcendentals (sin, cos, exp, log)
- Multi-cycle instructions

### MFMA (Matrix Fused Multiply-Add) Operations (gfx90a+/CDNA)

*This section lists representative MFMA intrinsics. The kernel feeds **non-zero** packed operands (scalars/vectors as required per opcode) and requires **finite** FP accumulators with **at least one non-zero** `float`/`double` lane, or **at least one non-zero** `int` lane for INT8 MFMA—not reference-matrix numerical validation.*

Matrix operations using dedicated Matrix Cores. For maintained MFMA / roofline work, see `src/roofline/benchmark/` in rocprofiler-compute (the older rocm-amdgpu-bench repo is deprecated).

**MI250 (gfx90a/CDNA2):**
- `__builtin_amdgcn_mfma_f32_32x32x2f32` - FP32 (4096 ops/inst)
- `__builtin_amdgcn_mfma_f64_16x16x4f64` - FP64 (2048 ops/inst)
- `__builtin_amdgcn_mfma_f32_32x32x8f16` - FP16 (16384 ops/inst)
- `__builtin_amdgcn_mfma_f32_32x32x4bf16` - BF16 (8192 ops/inst)
- `__builtin_amdgcn_mfma_i32_32x32x8i8` - INT8 (16384 ops/inst)

**MI300 (gfx942/CDNA3) - Enhanced:**
- All MI250 ops plus doubled K dimensions:
- `__builtin_amdgcn_mfma_f32_32x32x8bf16_1k` - BF16 (16384 ops/inst)
- `__builtin_amdgcn_mfma_i32_32x32x16_i8` - INT8 (32768 ops/inst)
- `__builtin_amdgcn_mfma_f32_32x32x16_fp8_fp8` - FP8 (32768 ops/inst) **NEW**

**MI350 (gfx950/CDNA4) - Scaled F8F6F4:**
- All MI300 ops plus:
- `__builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4` - FP8/BF8/FP6/FP4 (131072 ops/inst)

### WMMA (Wave Matrix Multiply Accumulate) Operations (RDNA4/gfx12)

*Same scope as MFMA: **non-zero** A/B fragments (indexed using **lane_id**), FP WMMA requires **finite** outputs with **at least one non-zero** lane in the per-lane accumulator, INT WMMA requires **at least one non-zero** lane—not exhaustive conformance. RDNA 3.5 (gfx115x) uses the 16-wide-input WMMA builtins; RDNA4 (gfx120x) uses the `_w32_gfx12` three-operand builtins—see `mega_kernel.hip`.*

WMMA instructions are the matrix acceleration approach for RDNA architecture (vs MFMA for CDNA).
Reference: [AMD GPUOpen - Using Matrix Cores on RDNA4](https://gpuopen.com/learn/using_matrix_core_amd_rdna4/)

**WMMA Performance (FLOPS/clock/CU):**
| Format | RDNA2 (RX 6950) | RDNA3 (RX 7900) | RDNA4 (RX 9070/) |
|--------|:---------------:|:---------------:|:---------------------:|
| FP16   | 256             | 512             | 1024                  |
| BF16   | N/A             | 512             | 1024                  |
| INT8   | 512             | 512             | 2048                  |

**WMMA Intrinsic Format:**
```
__builtin_amdgcn_wmma_<C,D format>_16x16x16_<A,B format>_w32_gfx12
```

**Available WMMA Intrinsics (gfx12/RDNA4):**
- `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` - FP16 → FP32 (16384 ops)
- `__builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12` - BF16 → FP32 (16384 ops)
- `__builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12` - INT8 → INT32 (16384 ops)
- `__builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12` - FP16 → FP16
- `__builtin_amdgcn_wmma_bf16_16x16x16_bf16_w32_gfx12` - BF16 → BF16

**WMMA Matrix Layout:**
- 16x16 matrix distributed across Wave32 (32 lanes)
- Each lane holds 8 elements (8 × 32 = 256 = 16 × 16)
- Matrix A: Column-major (transposed)
- Matrices B, C, D: Row-major
- `lane_id = threadIdx.x % warpSize` (matches `mega_kernel.hip`)
- `laneWrapped = lane_id % 16` (column index)
- `laneGroup = lane_id / 16` (upper/lower half)

**Key RDNA4 WMMA Differences vs RDNA3:**
- VGPR layout changed (NOT backward compatible)
- All intrinsics have `_gfx12` suffix
- Wave32 only (no Wave64 WMMA)
- 2x throughput improvement vs RDNA3

**Supported on:**
- Strix Point (gfx1150) / Strix Halo (gfx1151)
- RX 9070 XT/9070 (gfx1201)
- RX 9060 series (gfx1200)
- Bypassed on CDNA architectures (use MFMA instead)

### DOT Product Operations (CDNA)
- V_DOT4_U32_U8 (4-element unsigned INT8 dot product)
- V_DOT8_U32_U4 (8-element unsigned INT4 dot product)

### Lane/Thread Information
- Lane ID builtins
- Wavefront size detection

### Strix / Strix Halo / Krackan (RDNA 3.5 / gfx1150 / gfx1151 / gfx1152) Specific Features

*Background on RDNA 3.5 APUs; coverage here is limited to paths the matrix marks as tested (e.g. WMMA/TEX), not every fixed-function block.*

RDNA 3.5 is the integrated GPU in Strix Point (gfx1150, 16 CUs), Strix Halo (gfx1151, up to 40 CUs), and Krackan (gfx1152) client APUs:

1. **Wave32 Native**
   - Native Wave32 execution; optimized for mobile and APU workloads.

2. **WMMA (Wave Matrix Multiply Accumulate)**
   - 16x16x16 matrix operations (FP16/BF16/INT8), same ISA as RDNA4/gfx12.
   - Supported via `__builtin_amdgcn_wmma_*_w32_gfx12` intrinsics.

3. **VOPD Same-Wave Dual-Issue**
   - Same-wave dual-issue VALU via VOPD encoding (similar to CDNA3/CDNA4).

4. **Scalar FP and Vector Hints**
   - Scalar floating-point in the scalar unit; `s_singleuse_vdst` hint.
   - Note: FP8/BF8 conversion builtins (`__builtin_amdgcn_cvt_pk_fp8_f32`) are NOT available on RDNA 3.5 APUs.

5. **Async LDS and WGP**
   - Async LDS operations; WGP (Workgroup Processor) architecture.

   - No Tensor Data Mover; no hardware FP64 atomics (consumer iGPU).
   - Unified `s_waitcnt` (pre–gfx12 split waits).


## RDNA4 Consumer (RX 9070 XT/gfx1200/gfx1201) Specific Features

*Consumer RDNA4 background; this kernel focuses on VALU/WMMA/FP8/async-LDS-style paths in the matrix—not a graphics or fixed-function survey.*

The following features are available on RDNA4 consumer GPUs (RX 9000 series):

1. **Wave32 Native Architecture**
   - Native Wave32 execution (Wave64 emulated)
   - Optimized for gaming and inference workloads

2. **FP8/BF8 OCP Format Support**
   - Open Compute Platform (OCP) FP8/BF8 format
   - `__builtin_amdgcn_cvt_pk_fp8_f32`
   - `__builtin_amdgcn_cvt_pk_bf8_f32`

3. **AI Inference Accelerators**
   - Hardware acceleration for INT8/FP8 inference *(not separately benchmarked here—WMMA/FP8/dot coverage follows the matrix)*

4. **Async LDS Operations**
   - `global_load_async_to_lds_b*` instructions
   - Similar to CDNA4 but optimized for consumer workloads

5. **WGP (Workgroup Processor) Architecture**
   - 2 CUs per WGP
   - Improved SIMD scheduling

## Building

### Prerequisites
- ROCm 6.0+ with HIP
- hipcc compiler
- AMD GPU (e.g. Strix Halo / gfx1151, or other supported GFXIP)

### Using Make

```bash
# Build for gfx1151 (default; override with GPU_ARCH=gfx950, etc.)
make

# Build for specific architectures
make gfx950      # MI350/MI355X (CDNA4)
make gfx942      # MI300/MI300X/MI325X (CDNA3)
make gfx90a      # MI250/MI250X (CDNA2)
make gfx1150     # Strix Point (RDNA 3.5)
make gfx1151     # Strix Halo (RDNA 3.5)
make gfx1152     # Krackan (RDNA 3.5)
make gfx1201     # RX 9070 XT/9070 (RDNA4)
make gfx1200     # RX 9060 series (RDNA4)

# Build for all architectures (CDNA + RDNA4)
make all-arch

# Build and run
make run

# Show help
make help

# Clean build artifacts
make clean
```

### Using CMake

```bash
mkdir build && cd build

# For MI350 (CDNA4)
cmake .. -DGPU_TARGETS="gfx950"



# For multiple architectures
cmake .. -DGPU_TARGETS="gfx950;gfx942;gfx90a;gfx1150;gfx1151;gfx1201;"

make
./mega_kernel_test
```

### Manual Build

```bash
# For MI350 (CDNA4)
hipcc -O2 -std=c++17 --offload-arch=gfx950 -o mega_kernel_test main.cpp mega_kernel.hip


# For Strix Point/Strix Halo/Krackan (RDNA 3.5)
hipcc -O2 -std=c++17 --offload-arch=gfx1150 -o mega_kernel_test main.cpp mega_kernel.hip  # Strix Point
hipcc -O2 -std=c++17 --offload-arch=gfx1151 -o mega_kernel_test main.cpp mega_kernel.hip  # Strix Halo
hipcc -O2 -std=c++17 --offload-arch=gfx1152 -o mega_kernel_test main.cpp mega_kernel.hip  # Krackan
```

## Running

```bash
# Run with default settings (batch_size=1024)
./mega_kernel_test

# Run with custom batch size
./mega_kernel_test -b 4096

# Run with larger batch size for stress testing
./mega_kernel_test --batch-size 65536

# Run with custom batch size and block size
./mega_kernel_test -b 65536 -t 512

# Run kernel multiple times (for benchmarking/profiling)
./mega_kernel_test -n 100

# Run with larger batch and multiple iterations
./mega_kernel_test -b 65536 -n 50

# Run with only inline assembly MFMA tests
./mega_kernel_test --mfma-mode asm

# Run with only HIP builtin MFMA tests
./mega_kernel_test -m builtin

# Run both ASM and builtin MFMA tests (default)
./mega_kernel_test -m both

# Show help
./mega_kernel_test --help
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-b, --batch-size <N>` | Set batch/problem size (64 to 16M) | 1024 |
| `-t, --block-size <N>` | Set thread block size (power of 2, 64-256) | 256 |
| `-n, --num-iterations <N>` | Number of kernel iterations (1 to 10000) | 10 |
| `-m, --mfma-mode <MODE>` | MFMA test mode: `both`, `asm`, or `builtin` (see note) | both |
| `-v, --verbose` | Enable verbose output | off |
| `-h, --help` | Show help message | - |

**Note on MFMA mode:** MFMA instructions require complex accumulator register allocation that is difficult to express in inline assembly constraints. As a result, the MFMA tests primarily use HIP builtins which properly handle register allocation. The `--mfma-mode` option controls which test paths are exercised:
- `builtin` (recommended): Uses HIP builtins (e.g., `__builtin_amdgcn_mfma_f32_32x32x2f32`)
- `asm`: Attempts inline assembly where supported (falls back to builtins for complex MFMA ops)
- `both` (default): Runs both code paths where applicable


```
================================================================================
AMD GPU MEGA KERNEL UNIT TEST
================================================================================
Device Information:
  Name:                  AMD Instinct MI350
  GCN Architecture:      gfx950
  Compute Units:         304
  ...
================================================================================
Detected Platform:        MI350 (CDNA4/gfx950)
================================================================================

Running GPU Mega Kernel for  MI350 (CDNA4/gfx950)...

TEST RESULTS
================================================================================

[Category: Warp/Wave Operations]
  Warp Shuffle (shfl/shfl_xor/shfl_up/down)      : PASS (4/1)
  Warp Ballot (__ballot/__any/__all)             : PASS (4/1)
  ...

[Category: Atomic Operations]
  Atomic Add FP64 (HW atomic on CDNA)            : PASS (4/1)
  ...


SUMMARY
================================================================================
Test Categories Passed:   44 / 44
Test Categories Failed:   0
Test Categories Bypassed: 0 (N/A for this architecture)
Kernel Execution Time:    0.523 ms

*** OVERALL: PASSED (44/44, 0 bypassed) ***
```

### Sample Output (gfx1151 - Strix Halo)

```
================================================================================
AMD GPU MEGA KERNEL UNIT TEST
================================================================================
Device Information:
  Name:                  AMD Radeon(TM) Graphics
  GCN Architecture:      gfx1151
  Compute Units:         40
  ...
================================================================================
Detected Platform:       Strix/Strix Halo/Krackan (RDNA3.5/gfx1150,gfx1151,gfx1152)
================================================================================

Running GPU Mega Kernel for Strix/Strix Halo/Krackan (RDNA3.5/gfx1150,gfx1151,gfx1152)...

TEST RESULTS
================================================================================

[Category: Warp/Wave Operations]
  Warp Shuffle (shfl/shfl_xor/shfl_up/down)      : PASS (4/1)
  ...

[Category: Atomic Operations]
  Atomic Add FP64 (HW atomic on CDNA)            : BYPASSED (N/A)
  ...

[Category: Dual-Issue VALU (MI350/gfx950 specific)]
  Dual-Issue Patterns (FP64+FP32 co-exec)        : BYPASSED (N/A)


SUMMARY
================================================================================
Test Categories Passed:   35 / 35
Test Categories Failed:   0
Test Categories Bypassed: 9 (N/A for this architecture)
Kernel Execution Time:    0.312 ms

*** OVERALL: PASSED (35/35, 9 bypassed) ***
```

## Architecture Comparison

Informational hardware notes (not a full map of what this sample executes). For **async LDS in this binary**, see the main compatibility matrix above (gfx942 is off here).

| Feature | MI250 (gfx90a) | MI300 (gfx942) | MI350 (gfx950) | Strix/Krackan (gfx115x)  | RX 9070 XT (gfx1201) |
|---------|:-------------:|:--------------:|:--------------:|:-------------------:|:--------------------:|
| Architecture | CDNA2 | CDNA3 | CDNA4 | RDNA 3.5 | RDNA4 |
| FP8 Support | No | FNUZ | OCP | No | OCP |
| FP6/FP4 Support | No | No | Yes | No | No |
| WMMA Wave64 | Yes | Yes | No (Wave32) | No | No |
| VGPRs per Wave32 | 512 | 512 | 1024 | 512 | 512 |
| Packed FP16 Atomics | Yes | Yes | Yes | Yes | Yes |
| Packed BF16 Atomics | No | Yes | Yes | Yes | Yes |
| Dual VALU Issue | No | VOPD | Yes | VOPD | No |
| Async LDS (hardware) | No | Limited† | Yes | Yes | Yes |
| MFMA Matrix Ops | Yes | Yes | Yes | No | No |
| WMMA Matrix Ops | No | No | No | Yes | Yes |

† This sample still treats gfx942 as without the async-LDS builtins path; see `mega_kernel_device_arch.h`.

### Memory/Atomic Instruction Syntax

| Instruction Type | CDNA (gfx9x) | RDNA4 (gfx12) |
|------------------|--------------|---------------|
| Load 32-bit | `*_load_dword` | `*_load_b32` |
| Load 64-bit | `*_load_dwordx2` | `*_load_b64` |
| Store 32-bit | `*_store_dword` | `*_store_b32` |
| Atomic add | `global_atomic_add` | `global_atomic_add_u32` |
| Atomic max (signed) | `global_atomic_smax` | `global_atomic_max_i32` |
| Return old value | `glc` flag | `th:TH_ATOMIC_RETURN` |
| Wait for VMEM | `s_waitcnt vmcnt(0)` | `s_waitcnt vmcnt(0)` |
| Wait for LDS | `s_waitcnt lgkmcnt(0)` | `s_waitcnt lgkmcnt(0)` |

## License

Copyright (c) 2026 - For testing purposes

## References

- [AMD Instinct MI350/ Architecture](https://rocm.docs.amd.com/)
- [AMD Radeon RX 9000 Series](https://www.amd.com/radeon/)
- [HIP Programming Guide](https://rocm.docs.amd.com/projects/HIP/)
- [AMD Matrix Instruction Calculator](https://github.com/ROCm/amd_matrix_instruction_calculator)
