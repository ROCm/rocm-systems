# Salmon — AMDGPU Binary Transpiler

**Codename: Salmon**

## Overview

Salmon lifts native AMDGPU machine code to LLVM IR for cross-ISA binary
translation.  When active (`HSA_HOTSWAP_IR_RAISER=1`), Salmon handles
**all** ISA mismatches — both cross-family (e.g. gfx1250→gfx942) and
same-family (e.g. gfx950→gfx942) — by raising, optimizing, and
re-lowering through LLVM.  It is a standalone CMake project that links
against a pre-built LLVM with AMDGPU support.

## Prerequisites

| Dependency | Required? | Version | Notes |
|-----------|:---------:|---------|-------|
| **LLVM** (with AMDGPU backend) | **Yes** | 18+ (tested with 23.0.0git) | Must include `llc`, `llvm-mc`, `ld.lld` |
| **CMake** | **Yes** | 3.20+ | |
| **Ninja** | Recommended | any | `apt install ninja-build` |
| **C++17 compiler** | **Yes** | GCC 11+ or Clang 15+ | |
| **HIP + ROCm** | Optional | ROCm 6.x / 7.x | Only for GPU execution tests |
| **AMD GPU** | Optional | MI300X (gfx942) recommended | Only for GPU execution tests |

**HIP and a GPU are NOT required to build the transpiler or run the batch raise
test.**  They are only needed for the GPU execution tests that verify raised
kernels produce correct results on hardware.

### Building LLVM with AMDGPU support

If you don't already have a suitable LLVM install, build one:

```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
cmake -S llvm -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_TARGETS_TO_BUILD="AMDGPU" \
  -DLLVM_ENABLE_PROJECTS="lld" \
  -DCMAKE_INSTALL_PREFIX=$HOME/shared-llvm
ninja -C build install
```

This gives you the LLVM libraries the transpiler links against, plus the tools
(`llc`, `llvm-mc`, `ld.lld`) it invokes during the IR-to-HSACO pipeline.

## Building the transpiler

```bash
cd projects/rocr-runtime/runtime/hsa-runtime/hotswap/transpiler
mkdir build && cd build

cmake .. -G Ninja \
  -DLLVM_INSTALL_DIR=$HOME/shared-llvm \
  -DCMAKE_CXX_COMPILER=clang++

ninja transpiler_tests
```

This builds the transpiler library and the unified test binary.  No GPU or HIP
needed — GPU tests auto-skip at runtime when HIP is unavailable.

## Obtaining test code objects

The transpiler operates on pre-compiled AMDGPU code objects (`.co` / `.hsaco`
files).  None are checked into this directory — they are binary artifacts you
supply.  There are several ways to get them:

### Option A: Use any code object you already have

The `BatchRaise` and `Corpus` tests pick up `.co`/`.hsaco` files from paths
configured at CMake time.  If you have ROCm installed, rocBLAS ships
pre-compiled HSACOs that the corpus test finds automatically.

### Option B: Download AITER production kernels

The `hotswap/kernels/` directory contains a script to fetch pre-compiled AITER
CK kernels (the same corpus the transpiler was developed against):

```bash
cd ../../kernels
python3 fetch_aiter_kernels.py           # ~30 representative kernels
python3 fetch_aiter_kernels.py --full    # all kernels (~500+)
```

### Option C: Compile the vecadd and MFMA test kernels (for GPU tests)

The GPU execution tests (`ir_gpu_test`, `mfma_gpu_test`) need specific code
objects.  The MFMA kernel source is in the repo at `tests/mfma_gemm.hip`.  The
vecadd kernel is a standard HIP vecadd.

Compile them with `hipcc`:

```bash
# vecadd (write a trivial vecadd.hip or use any existing one):
hipcc --genco --offload-arch=gfx942 -o ../../build/mve_vecadd_gfx942.co vecadd.hip

# MFMA GEMM:
hipcc --genco --offload-arch=gfx942 -o build/mfma_gemm_gfx942_unbundled.co tests/mfma_gemm.hip
```

### Option D: gfx1250 Triton kernels (for `gfx1250_gpu_test`)

See `test_data/gfx1250/README.md` for how to generate these using Triton AOT
compilation.  They are compiled for gfx1250 (RDNA4) on a machine that does
not need a gfx1250 GPU.

---

## Running the tests

All tests live in a single GoogleTest binary (`transpiler_tests`), orchestrated
by CTest.  All paths below assume you are in `hotswap/transpiler/build`.

```bash
# Build (GPU tests auto-skip if HIP is unavailable):
cmake .. -G Ninja \
  -DCMAKE_PREFIX_PATH="/opt/rocm;$HOME/shared-llvm" \
  -DLLVM_INSTALL_DIR=$HOME/shared-llvm \
  -Dhip_DIR=/opt/rocm/lib/cmake/hip \
  -DCMAKE_CXX_COMPILER=clang++
ninja transpiler_tests

# Run via CTest (process isolation, timeouts, xfail handling).
# --output-on-failure prints GoogleTest output only for failing tests.
ctest --test-dir . --output-on-failure

# Or run the binary directly:
./transpiler_tests

# Filter to a single suite:
./transpiler_tests --gtest_filter='BatchRaise.*'

# Extended corpus (slow — thousands of kernels):
./transpiler_tests --test-all --gtest_filter='Corpus.*'

# Raise all kernels in an arbitrary directory (fork-isolated, ISA auto-detected):
./transpiler_tests --raise-dir=/path/to/kernels --gtest_filter='BatchRaise.CustomDir'
```

| GoogleTest suite | GPU? | What it tests |
|-----------------|:----:|---------------|
| `BatchRaise` | No | Raise rate on `.co`/`.hsaco` files or directories |
| `Corpus` | No | System HSACO corpus, fork-isolated per ISA |
| `IrGpu` | Yes | Same-ISA vecadd roundtrip (gfx942) |
| `MfmaGpu` | Yes | Same-ISA MFMA GEMM (gfx942) |
| `CrossArchGpu` | Yes | Cross-ISA raise + execute (rocBLAS HSACOs) |
| `Gfx1250Gpu` | Yes | gfx1250 Triton kernels → gfx942 |
| `Integration` | Yes | Multi-kernel raise + merge + load + execute |

GPU tests use `GTEST_SKIP()` when code objects or HIP are unavailable.
Known-failing tests are tracked in `tests/xfail.cmake` (CTest `WILL_FAIL`) —
they still run, but CTest expects them to fail and flags unexpected passes.

### Standalone tests (not part of the CMake project)

These require a Salmon-enabled ROCR build (`$ROCR_BUILD`).  See the source
files for build and run instructions:

- **`hotswap_load_test`** (`tests/hotswap_load_test.cpp`) — loads `.co` files
  via the HSA API, triggering Salmon raise + load.  No HIP needed.
- **`salmon_hip_test`** (`tests/salmon_hip_test.cpp`) + **`libsalmon_intercept.so`**
  (`tests/salmon_intercept.cpp`) — full end-to-end HIP integration: load a
  gfx950 kernel, dispatch on gfx942, verify correctness.

### Intermediates

By default, Salmon writes intermediate files to a temp directory that is
**cleaned up** when the pipeline finishes.  To persist them for debugging, set
`HSA_SALMON_DUMP_DIR`:

```bash
HSA_SALMON_DUMP_DIR=/tmp/salmon_dumps \
HSA_SALMON_DUMP_INPUT=1 \
HSA_HOTSWAP_ISA_OVERRIDE=gfx942 HSA_HOTSWAP_IR_RAISER=1 \
HSA_HOTSWAP_RULES=/dev/null \
LD_LIBRARY_PATH=$ROCR_BUILD/rocr/lib \
./my_program
```

Each invocation creates a unique subdirectory (e.g. `salmon_dumps/salmon-a1b2c3/`):

| File | Always? | Contents |
|------|:-------:|----------|
| `<kernel>.ll` | Yes | Raised LLVM IR |
| `<kernel>.s` | Yes | Target assembly (output of `llc`) |
| `k<N>.o` | Yes | Assembled object file |
| `merged.hsaco` | Yes | Linked HSACO (final code object) |
| `input.co` | `DUMP_INPUT=1` | Original input code object (binary) |
| `<kernel>.dis` | `DUMP_INPUT=1` | Disassembly of the input kernel |

---

## Kernel coverage

The transpiler targets production AMDGPU kernels compiled for the GFX9 family
(MI200/MI300 — gfx90a, gfx940, gfx942, gfx950) and GFX12/GFX1250 (RDNA4+).
Coverage depends on which instruction patterns a kernel uses.

### AITER kernels (gfx942/gfx950)

Tested against 27 representative AITER CK kernels (GEMM, FMHA, MoE, MLA,
paged-attention, topk-softmax, etc.).  The `BatchRaise.AiterGfx950` test
validates these numbers on every run.

| Metric | Value |
|--------|-------|
| Kernels raised | **12 / 27 (44.4%)** |
| Remaining failures | 15 (see breakdown below) |

Successfully raised kernel families:
- **FP8 GEMM block-scale** (2 of 2)
- **INT8 GEMM** (2 of 2)
- **MLA** (bf16 attention decode, 2 of 2)
- **Paged attention** (bf16 no-quant, 2 of 2)
- **TopK softmax** (f32 and bf16, 2 of 2)
- **MoE 2-stage** (fp8 block-scale, 2 of 2)

### Known failures

| Kernels | Count | Root cause |
|---------|:-----:|------------|
| `bf16gemm` | 2 | `v_add_i32` unsupported in VOP3 |
| `f4gemm` | 2 | `XNACK_MASK_HI` register not classified (crash) |
| `f8_block_scale` | 4 | CFG reconstruction edge case ("terminator in middle of BB") |
| `fmha_v3_bwd` | 2 | `LDS_DIRECT` operand (crash) |
| `fmha_v3_fwd` | 3 | `v_add_i32` unsupported in VOP3 |
| `fmoe` | 2 | `v_add_i32` unsupported in VOP3 |

### Known limitations

| Category | Description |
|----------|-------------|
| **`v_add_i32` (gfx950)** | GFX9 `v_add_i32` in VOP3 encoding is not yet mapped. Blocks bf16gemm, fmha_fwd, fmoe. |
| **`XNACK_MASK_HI`** | Register not classified by `parseReg`; triggers `report_fatal_error`. Blocks f4gemm. |
| **IR verification failures** | 4 f8_block_scale kernels fail IR verification due to a CFG reconstruction edge case. |
| **`LDS_DIRECT` operand** | The LDS_DIRECT pseudo-register triggers a fatal error in fmha_bwd kernels. |
| **WMMA (gfx1250)** | Wave Matrix Multiply support is being added by a separate effort. |

### Instruction coverage highlights

The transpiler handles the full GFX9 instruction set including:
- Scalar ALU (SOP1/SOP2/SOPK/SOPC), including carry-chain operations
- Vector ALU (VOP1/VOP2/VOP3), including GFX9 naming variants (v_add_u32 → v_add_co_u32)
- Packed f16 ops (v_pk_fmac_f16, v_pk_add_f32, v_pk_fma_f32)
- Dot product accumulate (v_dot2c_i32_i16, v_dot4c_i32_i8, v_dot8c_i32_i4)
- MFMA (matrix fused multiply-add, including scaled variants)
- MUBUF with buffer resource descriptors (loads, stores, atomics, LDS-direct)
- SMEM (scalar loads, s_atomic_swap)
- FLAT/GLOBAL memory (loads, stores, atomics including pk_add_bf16/f16)
- DS (LDS reads/writes, including strided and 128-bit)
- VOPC/CMPX (vector comparisons with VCC/EXEC writeback)
- Control flow (branches, saveexec, EXEC mask management)
- VOPD (dual-issue instructions, gfx12+)

---

## Troubleshooting

**`LLVM not found`** — Set `-DLLVM_INSTALL_DIR=` to the prefix where LLVM is
installed (the directory containing `lib/cmake/llvm/`).

**`llc` / `llvm-mc` / `ld.lld` not found at runtime** — The pipeline shells
out to these tools at `${LLVM_INSTALL_DIR}/bin/`.  Make sure they exist there.
The path is baked in at CMake configure time via the `LLVM_TOOLS_DIR` define.

**GPU tests not built** — CMake prints `GPU tests DISABLED (HIP not found)`.
Pass `-Dhip_DIR=/opt/rocm/lib/cmake/hip` or add ROCm to `CMAKE_PREFIX_PATH`.

**Link errors (missing LLVM symbols)** — The static LLVM library list in
`CMakeLists.txt` may need updating if your LLVM version adds or renames
component libraries.  Run `llvm-config --libs amdgpu codegen mc` to see
what your install provides.

---

## What Salmon reuses from the legacy hotswap

Salmon replaces the legacy transpiler's *instruction rewriting engine* but
reuses the surrounding integration infrastructure.  Specifically:

| Component | Source | Reused as-is? | Purpose |
|-----------|--------|:---:|---------|
| **HIP fat binary intercept** | `clr/hipamd/src/hip_fatbin.cpp` | **Yes** | Extracts cross-gen code objects from fat binaries, patches `e_flags` and `.note` ISA strings so HIP's ISA compatibility check passes |
| **ELF ISA patching** | `hotswap/hotswap.cpp: PatchElfIsa()` | **Yes** | Patches `e_flags` and `.note` sections to match the target ISA |
| **ISA override plumbing** | `loader/executable.cpp` | **Yes** | `HSA_HOTSWAP_ISA_OVERRIDE` env var, unknown-ISA early bypass, `.note` ISA mismatch detection |
| **`IsEnabled()` / `IsIsaOverrideEnabled()`** | `hotswap/hotswap.hpp` | **Yes** | Gate the hotswap hook based on env vars |
| **`rocr_hotswap_retarget` export** | `hotswap/hotswap.cpp` | **No** | C-linkage bridge for HIP→ROCR retarget calls; Salmon does not need this (it hooks at `LoadCodeObject` instead) |
| **`RetargetCodeObject`** | `hotswap/hotswap.cpp` | **No** | Legacy same-family byte-level retargeting; replaced by Salmon |
| **`TranspileCodeObject`** | `hotswap/transpiler.cpp` | **No** | Legacy cross-family byte-level transpiling; replaced by Salmon |

## Environment variables

| Variable | Required? | Description |
|----------|:---------:|-------------|
| `HSA_HOTSWAP_ISA_OVERRIDE` | **Yes** | Target ISA (e.g. `gfx942`).  Enables the hotswap hook. |
| `HSA_HOTSWAP_IR_RAISER` | **Yes** | Set to `1` to activate Salmon.  Without this, the legacy transpiler handles mismatches. |
| `HSA_HOTSWAP_RULES` | No | Path to JSON rules file.  Set to `/dev/null` if no rules are needed but the engine must be enabled. |
| `HSA_SALMON_DUMP_DIR` | No | Directory for intermediate files.  Each invocation creates a unique `salmon-XXXXXX/` subdirectory.  When unset, intermediates go to a temp dir that is cleaned up on exit. |
| `HSA_SALMON_DUMP_INPUT` | No | Set to `1` to also save the input code object (`input.co`) and per-kernel disassembly (`.dis`) alongside the raised IR. |

