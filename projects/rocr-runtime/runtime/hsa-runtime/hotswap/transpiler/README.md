# Salmon — AMDGPU Binary Transpiler

**Codename: Salmon**

## Overview

Salmon lifts native AMDGPU machine code to LLVM IR for cross-ISA binary
translation.  When active (`HSA_HOTSWAP_IR_RAISER=1`), Salmon handles
**all** ISA mismatches — both cross-family (e.g. gfx1250→gfx942) and
same-family (e.g. gfx950→gfx942) — by raising, and re-lowering through LLVM. 
It is a standalone CMake project that links
against a pre-built LLVM with AMDGPU support.

## Prerequisites

| Dependency | Required? | Version | Notes |
|-----------|:---------:|---------|-------|
| **LLVM** (with AMDGPU backend) | **Yes** | 18+ (tested with 23.0.0git) | **Build tree**, not install tree. Must include `llc`, `llvm-mc`, `ld.lld` |
| **CMake** | **Yes** | 3.20+ | |
| **Ninja** | Recommended | any | `apt install ninja-build` |
| **C++17 compiler** | **Yes** | GCC 11+ or Clang 15+ | |
| **GoogleTest** | **Yes** | 1.10+ | `apt install libgtest-dev` (Ubuntu 22.04+ ships CMake config files) |
| **HIP + ROCm** | Optional | ROCm 6.x / 7.x | Only for GPU execution tests |
| **AMD GPU** | Optional | MI300X (gfx942) recommended | Only for GPU execution tests |

**HIP and a GPU are NOT required to build the transpiler or run the batch raise
test.**  They are only needed for the GPU execution tests that verify raised
kernels produce correct results on hardware.

### Building LLVM with AMDGPU support

If you don't already have a suitable LLVM build, build one.  **Do not run
`ninja install`** — the transpiler links against the LLVM *build tree*, not
an install prefix, because it reaches into `Target/AMDGPU` for target-private
headers (`SIDefines.h`, `AMDGPUBaseInfo.h`) and the TableGen-generated
`AMDGPUGen*.inc` files, neither of which are copied by `ninja install`.

```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
cmake -S llvm -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_TARGETS_TO_BUILD="AMDGPU" \
  -DLLVM_ENABLE_PROJECTS="lld"
ninja -C build
```

This gives you the LLVM libraries the transpiler links against, plus the tools
(`llc`, `llvm-mc`, `ld.lld`) it invokes during the IR-to-HSACO pipeline.  The
build tree's `lib/cmake/llvm/` directory is what you point `LLVM_DIR` at below.

> **Note.**  `AMDGPU` alone is sufficient for the **standalone** transpiler
> build in the next section.  If you are going to build a full
> Salmon-enabled ROCR runtime (see
> [Building a Salmon-enabled ROCR runtime](#building-a-salmon-enabled-rocr-runtime-rocr_build)),
> add `X86` (or `native`) to `LLVM_TARGETS_TO_BUILD` as well —
> `find_package(Clang)` from ROCm's Clang CMake config transitively
> requires `LLVMX86*` targets to be visible in the LLVM package that
> ROCR consumes, and the standalone transpiler build tree is what
> gets reused there.

## Building the transpiler

Point `LLVM_DIR` at `<llvm-build>/lib/cmake/llvm` (the same convention every
other LLVM out-of-tree project uses).

```bash
cd projects/rocr-runtime/runtime/hsa-runtime/hotswap/transpiler
mkdir build && cd build

cmake .. -G Ninja \
  -DLLVM_DIR=$HOME/llvm-project/build/lib/cmake/llvm \
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
  -DCMAKE_PREFIX_PATH="/opt/rocm;$HOME/llvm-project/build" \
  -DLLVM_DIR=$HOME/llvm-project/build/lib/cmake/llvm \
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

## Building a Salmon-enabled ROCR runtime (`$ROCR_BUILD`)

Everything above only needs the transpiler's own build tree.  The
standalone tools in the next section (`smoke_test_compare_transpilers`,
`compare_correctness/`) instead load Salmon *through* a live
`libhsa-runtime64.so`, so they need a full ROCR build with the
hotswap + IR-raiser CMake options turned on.  This section walks
through producing that build on a typical gfx942 machine.

### Prerequisites (in addition to the transpiler's)

- Everything from the transpiler [Prerequisites](#prerequisites), plus
  `libelf-dev`, `libdrm-amdgpu-dev` (or `libdrm-dev`), `pkg-config`,
  `rocm-core`, `rocm-llvm-dev` — the standard ROCR build deps, see
  `projects/rocr-runtime/README.md`.
- A **local LLVM build tree** with both `AMDGPU` and `X86` (or
  `native`) targets.  See the note at the end of
  [Building LLVM with AMDGPU support](#building-llvm-with-amdgpu-support)
  above; if you already have an AMDGPU-only tree you can add X86
  with a one-line reconfigure:
  ```bash
  cmake -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86" ~/llvm-project/build && \
  ninja -C ~/llvm-project/build
  ```

### Step 1: Get the full `projects/rocr-runtime` tree

If you cloned via `rocm-systems` with a sparse checkout limited to
`hotswap/`, expand it to include the rocr-runtime source:

```bash
cd ~/rocm-systems
git sparse-checkout add projects/rocr-runtime
```

A plain full clone of `rocm-systems` or `rocr-runtime` already has
everything and can skip this step.

### Step 2: Apply the `trap_handler_gfx12.s` compatibility patch (if your clang predates upstream LLVM 78defc82, Nov 2025)

ROCR's gfx12 trap handler uses the symbolic hwreg name
`HW_REG_WAVE_SCHED_MODE`, which was added to LLVM's AMDGPU MC parser
by upstream commit
[`78defc8`](https://github.com/llvm/llvm-project/commit/78defc820eef5185d50e9aa396fb5a4b5ed07f92)
on 2025-11-28 (i.e. post-LLVM-22).  If your ROCm ships a clang whose
LLVM tree predates that commit (for example, **ROCm 7.2.1 ships AMD
clang 22.0.0git** at `/opt/rocm-7.2.1/lib/llvm/bin/clang`, branched
earlier), the ROCR build fails with:

```
trap_handler_gfx12.s:850:29: error: expected a register name or an absolute expression
    s_setreg_b32      hwreg(HW_REG_WAVE_SCHED_MODE, 0, 2), ttmp2
```

The patch rewrites that one line to use the numeric hwreg id (26,
`ID_SCHED_MODE` in `llvm/lib/Target/AMDGPU/SIDefines.h`).  The numeric
form has been the raw underlying `hwreg` syntax since long before the
symbolic name was exposed, so every reasonably recent AMDGPU clang
accepts it; the emitted encoding is byte-for-byte identical to what
the symbolic form produces on LLVM 23.

```bash
cd projects/rocr-runtime
patch -p1 < runtime/hsa-runtime/hotswap/transpiler/patches/trap_handler_gfx12_llvm22_compat.patch
```

Skip this step if your ROCm clang's LLVM already includes `78defc8`
(the patch is a no-op at runtime — it only changes the assembly
source encoding, not the resulting code object).

### Step 3: Configure and build

```bash
cd projects/rocr-runtime
mkdir build && cd build

cmake -G Ninja .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="/opt/rocm-7.2.1/lib/llvm;/opt/rocm-7.2.1" \
  -DROCR_ENABLE_HOTSWAP=ON \
  -DROCR_ENABLE_IR_RAISER=ON \
  -DLLVM_DIR=$HOME/llvm-project/build/lib/cmake/llvm

ninja
```

Key flags:

| flag | why it matters |
|------|----------------|
| `-DROCR_ENABLE_HOTSWAP=ON` | Compiles the hotswap hook (legacy byte-level transpiler) into `libhsa-runtime64.so`.  Without this, the runtime has no way to intercept code-object loads and both engines are unreachable. |
| `-DROCR_ENABLE_IR_RAISER=ON` | Also links the Salmon IR raiser (`hotswap-transpiler` static library from this directory) into the runtime.  Without this, only the legacy byte-level path is available. |
| `-DLLVM_DIR=<build>/lib/cmake/llvm` | Same LLVM **build tree** the standalone transpiler uses — needed for AMDGPU target-private headers not shipped by `ninja install`. |
| `-DCMAKE_PREFIX_PATH="…rocm-7.2.1/lib/llvm;…rocm-7.2.1"` | Points `find_package(Clang)` (used by the trap-handler subproject for `clang` + `llvm-objcopy`) at the ROCm LLVM install.  Without this, CMake may latch onto a system package such as `/usr/lib/llvm-20/lib/cmake/clang/` that references static libraries its distro package doesn't ship, and configure fails with `The imported target "clangBasic" references the file … but this file does not exist.` |

`HOTSWAP_TRANSPILER_BUILD_TOOLS` is **not** listed above on purpose: it
defaults to `OFF` whenever the transpiler CMakeLists is pulled in via
`add_subdirectory` (as ROCR does), so the optional `raise_cli` + test
binaries are already suppressed for you.  You only need to pass
`-DHOTSWAP_TRANSPILER_BUILD_TOOLS=ON` if you explicitly want those
built as part of the ROCR build (and you have GTest + HIP available).

Build output:

- `build/rocr/lib/libhsa-runtime64.so.1.21.0` — the Salmon-enabled
  runtime.  Verify by running
  `nm -D build/rocr/lib/libhsa-runtime64.so.1 | grep rocr_salmon_patch_elf`;
  the symbol must be present.
- **No `ninja install` required.**  The tools in the next section
  `rpath` into `$ROCR_BUILD/rocr/lib` directly, so nothing touches
  `/opt/rocm-7.2.1`.  Set:
  ```bash
  export ROCR_BUILD=$HOME/rocm-systems/projects/rocr-runtime/build
  ```
  and the tool Makefiles will pick it up.

### Known pitfalls

- **"The imported target `clangBasic` references the file … but this file does not exist."**
  CMake resolved `find_package(Clang)` against a system LLVM package
  (e.g. `/usr/lib/llvm-20/lib/cmake/clang/ClangTargets.cmake`) whose
  imported targets reference static libraries your distro package
  does not actually install.  Fix by forcing CMake to use ROCm's
  matching Clang package via
  `-DCMAKE_PREFIX_PATH="/opt/rocm-<ver>/lib/llvm;/opt/rocm-<ver>"`
  as shown above.
- **"The following imported targets are referenced, but are missing: `LLVMX86CodeGen` …"**
  Your local LLVM build tree was configured AMDGPU-only.  Add X86
  per the reconfigure in the prerequisites above.
- **Trap-handler assembly error on a *different* line than the
  `HW_REG_WAVE_SCHED_MODE` one.**  The Step-2 patch only covers that
  single symbol.  If a significantly older clang rejects other gfx12
  mnemonics in `trap_handler_gfx12.s`, upgrade ROCm.  As a last
  resort on a machine that has no gfx1200 agent (e.g. MI300-only),
  you can drop gfx1200 from
  `runtime/hsa-runtime/core/runtime/trap_handler/CMakeLists.txt`'s
  `TARGET_DEVS` *and* remove the matching `kCodeTrapHandlerV2_12`
  entries from `runtime/hsa-runtime/core/runtime/amd_gpu_agent.cpp`
  — `AssembleShader` selects the trap handler per agent ISA, so the
  gfx1200 blob is never bound on a gfx942 system and its build-time
  presence is purely to ship a multi-ISA runtime.

### Standalone tools and end-to-end tests (not part of the CMake project)

These require a Salmon-enabled ROCR build (`$ROCR_BUILD`) because they
link against (or load through) the live `libhsa-runtime64.so` rather
than the transpiler library. They are built separately from
`transpiler_tests`.

- **`tools/smoke_test_compare_transpilers`**
  (`tools/smoke_test_compare_transpilers.cpp`, `tools/Makefile`) —
  **load-level smoke test**: does the code object
  load + freeze under the legacy byte-level transpiler and under
  Salmon?  Accepts a single `.co`/`.hsaco` file or a directory and
  reports per-file agreement between the two engines.  The kernel is
  never dispatched, so a `PASS` does NOT imply numerical correctness —
  a miscompilation is invisible to this tool.  Fork-isolates every
  `(file, mode)` pair because ROCR's engine selection
  (`HSA_HOTSWAP_IR_RAISER`) is frozen at first use.  Use as the cheap
  first filter over a corpus; use `compare_correctness/` to actually
  dispatch and diff outputs.  See `tools/README.md` for build/run
  instructions, output format, and exit-code conventions.
- **`tools/compare_correctness/`** — head-to-head **numerical**
  comparison of the two engines.  Authored HIP kernels (one `.hip` per
  recipe) are compiled to both gfx942 and gfx1250, dispatched in
  isolated child processes under `native` / `legacy` / `salmon`, and
  their outputs are compared to a CPU reference.  Complements
  `smoke_test_compare_transpilers`: load success does not imply correct numerical
  output, and this tool measures the latter.  See
  `tools/compare_correctness/README.md`.
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

### AITER kernels (gfx950)

Tested against 27 representative AITER CK kernels (GEMM, FMHA, MoE, MLA,
paged-attention, topk-softmax, etc.).  The `BatchRaise.AiterGfx950` test
validates these numbers on every run.

| Metric | Value |
|--------|-------|
| Kernels raised | **27 / 27 (100%)** |
| Remaining failures | 0 |

All kernel families raise successfully:
- **BF16 GEMM** (2 of 2)
- **FP4 GEMM** (2 of 2)
- **FP8 GEMM block-scale** (4 of 4)
- **FP8 GEMM** (2 of 2)
- **INT8 GEMM** (2 of 2)
- **FMHA forward** (3 of 3)
- **FMHA backward** (2 of 2)
- **MoE** (2 of 2, plus 2-stage variants)
- **MLA** (bf16 attention decode, 2 of 2)
- **Paged attention** (bf16 no-quant, 2 of 2)
- **TopK softmax** (f32 and bf16, 2 of 2)

### Known limitations

| Category | Description |
|----------|-------------|
| **WMMA (gfx1250)** | Wave Matrix Multiply support is being added by a separate effort. |

### Instruction coverage highlights

The transpiler handles a large subset of the GFX9/gfx950 instruction set.
Coverage is validated against the AITER corpus above; kernels using
instructions not listed here will fail with "Unsupported instruction".
Currently handled:
- Scalar ALU (SOP1/SOP2/SOPK/SOPC), including carry-chain operations
- Vector ALU (VOP1/VOP2/VOP3), including GFX9 naming variants and `v_add_i32`/`v_sub_i32`
- Lane-swap operations (`v_permlane16_swap_b32`, `v_permlane32_swap_b32`)
- Packed f16 ops (v_pk_fmac_f16, v_pk_add_f32, v_pk_fma_f32)
- Dot product accumulate (v_dot2c_i32_i16, v_dot4c_i32_i8, v_dot8c_i32_i4)
- MFMA (matrix fused multiply-add, including scaled F8F6F4 variants)
- MUBUF with buffer resource descriptors (loads, stores, atomics, LDS-direct)
- SMEM (scalar loads, s_atomic_swap)
- FLAT/GLOBAL memory (loads, stores, atomics including pk_add_bf16/f16)
- DS (LDS reads/writes, including strided, 128-bit, and transpose reads)
- LDS_DIRECT operand support (direct LDS reads via M0 address)
- VOPC/CMPX (vector comparisons with VCC/EXEC writeback)
- Control flow (branches, saveexec, EXEC mask management)
- VOPD (dual-issue instructions, gfx12+)

---

## Troubleshooting

**`LLVM not found`** — Set `-DLLVM_DIR=` to `<llvm-build>/lib/cmake/llvm`
(the LLVM *build tree*, not an install prefix).  The configure step will
fail with a clear error if you point it at an install tree.

**`Could not find a package configuration file provided by "GTest"`** —
Install GoogleTest development files (`apt install libgtest-dev` on Ubuntu
22.04+, or build/install upstream GoogleTest and set `-DGTest_DIR=<prefix>/lib/cmake/GTest`).

**`llc` / `llvm-mc` / `ld.lld` not found at runtime** — The pipeline shells
out to these tools at `<llvm-build>/bin/`.  Make sure they exist there.  The
path is baked in at CMake configure time via the `LLVM_TOOLS_DIR` define,
which is derived from LLVM's own `LLVM_TOOLS_BINARY_DIR`.

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

