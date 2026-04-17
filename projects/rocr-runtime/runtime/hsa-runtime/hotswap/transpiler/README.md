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

ninja batch_raise_test
```

This builds the transpiler library and `batch_raise_test`.  No GPU or HIP needed.

## Build targets

| Target | Requires HIP? | What it does |
|--------|:------------:|------|
| `hotswap-transpiler` | No | Static library: raiser + pipeline + ELF utils |
| `batch_raise_test` | No | Batch-raises all kernels in a `.co`/`.hsaco` file or directory; reports raise rate |
| `ir_gpu_test` | Yes | Same-ISA raise + GPU execution of vecadd (gfx942) |
| `mfma_gpu_test` | Yes | Same-ISA raise + GPU execution of MFMA GEMM (gfx942) |
| `cross_arch_gpu_test` | Yes | Cross-ISA raise (gfx1200 → gfx942) + GPU execution |
| `gfx1250_gpu_test` | Yes | Cross-ISA raise of gfx1250 Triton kernels (gfx1250 → gfx942) |

**Standalone tests** (built manually, not part of the CMake project):

| Binary | Requires | What it does |
|--------|----------|------|
| `hotswap_load_test` | GPU + Salmon-enabled ROCR | Loads `.co` files via HSA API, triggers Salmon raise + load |
| `libsalmon_intercept.so` | — | LD_PRELOAD shim: patches `e_flags` so HIP accepts cross-ISA code objects |
| `salmon_hip_test` | GPU + HIP + shim + Salmon-enabled ROCR | Full HIP integration: load gfx950 kernel, dispatch on gfx942, verify correctness |

## Obtaining test code objects

The transpiler operates on pre-compiled AMDGPU code objects (`.co` / `.hsaco`
files).  None are checked into this directory — they are binary artifacts you
supply.  There are several ways to get them:

### Option A: Use any code object you already have

`batch_raise_test` accepts **any** AMDGPU `.co` or `.hsaco`.  If you have
ROCm installed, rocBLAS ships pre-compiled HSACOs you can use immediately:

```bash
./batch_raise_test /opt/rocm/lib/rocblas/library/Kernels.so-000-gfx942-xnack-.hsaco
```

### Option B: Download AITER production kernels

The `hotswap/kernels/` directory contains a script to fetch pre-compiled AITER
CK kernels (the same corpus the transpiler was developed against):

```bash
cd ../../kernels
python3 fetch_aiter_kernels.py           # ~30 representative kernels
python3 fetch_aiter_kernels.py --full    # all kernels (~500+)
```

Then point the batch test at the downloaded directory:

```bash
cd ../transpiler/build
./batch_raise_test ../../kernels/aiter_kernels/
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

All paths below assume you are in the standalone transpiler build directory
(`hotswap/transpiler/build`).  `$ROCR_BUILD` refers to the ROCR runtime build
directory (e.g. `projects/rocr-runtime/build`).

### 1. batch_raise_test — offline raise rate (no GPU needed)

The main smoke test.  Raises every kernel in a code object or directory and
reports the raise rate.  No GPU, HIP, or custom HSA runtime required.

```bash
# Single code object:
./batch_raise_test /path/to/kernel.co

# Directory (recursively finds all .co/.hsaco files):
./batch_raise_test /path/to/kernel_directory/

# Override target ISA (normally auto-detected from ELF metadata):
./batch_raise_test /path/to/kernel.co gfx942

# AITER kernels (85.2% raise rate):
./batch_raise_test ../../kernels/aiter_gfx950/
```

Example output:

```
=== BATCH RAISE SUMMARY ===
Kernels attempted:      27
  Succeeded:            23  (85.2%)
  Failed:               4  (14.8%)
```

### 2. hotswap_load_test — HSA-level integration (GPU required)

Loads `.co` files directly through the HSA runtime's `LoadCodeObject` path,
bypassing HIP entirely.  Exercises the same code path that PyTorch would hit
after HIP hands the code object to the HSA runtime.

**Build** (standalone, not part of the CMake project):

```bash
ROCR_BUILD=../../../../../build   # adjust to your ROCR build dir

g++ -std=c++17 -O2 tests/hotswap_load_test.cpp \
  -I/opt/rocm/include -I$ROCR_BUILD/rocr/include \
  -L$ROCR_BUILD/rocr/lib -lhsa-runtime64 \
  -Wl,-rpath,$ROCR_BUILD/rocr/lib \
  -o hotswap_load_test
```

**Run:**

```bash
HSA_HOTSWAP_ISA_OVERRIDE=gfx942 \
HSA_HOTSWAP_RULES=/dev/null \
HSA_HOTSWAP_IR_RAISER=1 \
LD_LIBRARY_PATH=$ROCR_BUILD/rocr/lib \
./hotswap_load_test ../../kernels/aiter_gfx950/ --recursive
```

Each code object is loaded via `hsa_executable_load_agent_code_object`.
Salmon intercepts, raises to LLVM IR, re-lowers to the target ISA, and the
HSA runtime loads the fresh HSACO.  The test reports pass/fail and timing
per file.

### 3. salmon_hip_test — end-to-end HIP integration (GPU required)

The full integration test.  Loads a gfx950 code object via `hipModuleLoadData`,
dispatches a vecadd kernel on gfx942 hardware, and verifies numerical
correctness.  This exercises both integration layers:

- **Layer 1** (LD_PRELOAD shim): patches `e_flags` so HIP accepts the ELF
- **Layer 2** (HSA runtime): detects original ISA from MSGPACK metadata, triggers Salmon

**Build the shim and test** (standalone, not part of the CMake project):

```bash
ROCR_BUILD=../../../../../build   # adjust to your ROCR build dir

# Build the LD_PRELOAD shim
g++ -shared -fPIC -O2 -o libsalmon_intercept.so tests/salmon_intercept.cpp -ldl

# Build the test
hipcc -std=c++17 -O2 tests/salmon_hip_test.cpp -o salmon_hip_test
```

**Generate the test code object** (if `tests/vecadd_gfx950.co` doesn't exist):

```bash
cat > /tmp/vecadd.hip << 'EOF'
#include <hip/hip_runtime.h>
extern "C" __global__ void vecadd(const float* A, const float* B, float* C, int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  if (i < N) C[i] = A[i] + B[i];
}
EOF

hipcc --genco --offload-arch=gfx950 -o /tmp/vecadd_gfx950_bundled.co /tmp/vecadd.hip
/opt/rocm/llvm/bin/clang-offload-bundler --type=o \
  --targets=hipv4-amdgcn-amd-amdhsa--gfx950 \
  --input=/tmp/vecadd_gfx950_bundled.co \
  --output=tests/vecadd_gfx950.co --unbundle
```

**Run:**

```bash
LD_PRELOAD=./libsalmon_intercept.so \
LD_LIBRARY_PATH=$ROCR_BUILD/rocr/lib \
HSA_HOTSWAP_ISA_OVERRIDE=gfx942 \
HSA_HOTSWAP_IR_RAISER=1 \
HSA_HOTSWAP_RULES=/dev/null \
./salmon_hip_test tests/vecadd_gfx950.co
```

Expected output:

```
salmon_intercept: active, target=gfx942
salmon_intercept: patched e_flags for gfx942 (5648 bytes)
=== Salmon HIP Integration Test ===
  ...
  PASS: 1024/1024 elements correct
  Salmon transpiled gfx950 -> gfx942 and executed correctly.
```

### 4. GPU execution tests — same-ISA and cross-ISA (GPU + HIP required)

These are CMake targets that raise a kernel and run it on the GPU, verifying
correctness.  Add HIP to the CMake configuration:

```bash
cmake .. -G Ninja \
  -DCMAKE_PREFIX_PATH="/opt/rocm;$HOME/shared-llvm" \
  -DLLVM_INSTALL_DIR=$HOME/shared-llvm \
  -Dhip_DIR=/opt/rocm/lib/cmake/hip \
  -DCMAKE_CXX_COMPILER=clang++

ninja
```

Adjust `/opt/rocm` to your ROCm install path (e.g., `/opt/rocm-7.2.1`).

```bash
./ir_gpu_test           # same-ISA vecadd (gfx942 → gfx942)
./mfma_gpu_test         # same-ISA MFMA GEMM (gfx942 → gfx942)
./cross_arch_gpu_test   # cross-ISA (gfx1200/1250 → gfx942)
./gfx1250_gpu_test      # gfx1250 Triton kernels → gfx942
```

**Note:** `cross_arch_gpu_test` has rocBLAS HSACO paths hardcoded to
`/opt/rocm-7.2.1/...` in `CMakeLists.txt`.  If you have a different ROCm
version, edit the `NATIVE_HSACO` and `SOURCE_HSACO` variables before
configuring.

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
paged-attention, topk-softmax, etc.):

| Metric | Value |
|--------|-------|
| Kernels raised | **23 / 27 (85.2%)** |
| Remaining failures | 4 (IR verification: CFG reconstruction edge case) |

Successfully raised kernel families:
- **GEMM** (bf16, fp4, fp8 block-scale, int8)
- **FMHA forward** (bf16, with/without causal mask, GQA)
- **FMHA backward** (bf16)
- **MoE** (fp8 block-scale, gate/up fusion)
- **MLA** (bf16 attention decode)
- **Paged attention** (bf16 no-quant)
- **TopK softmax** (f32 and bf16 variants)

### Known limitations

| Category | Description |
|----------|-------------|
| **BUFFER_LOAD_DWORD_LDS** | Direct-to-LDS buffer loads are now supported. |
| **IR verification failures** | 4 f8_block_scale kernels fail IR verification due to a CFG reconstruction edge case ("terminator in the middle of a basic block"). |
| **WMMA (gfx1250)** | Wave Matrix Multiply support is being added by a separate effort. |
| **LDS_DIRECT operand** | The LDS_DIRECT pseudo-register (used in some FMHA backward kernels) triggers a non-fatal warning but does not block raising when the register is used in non-critical paths. |

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

## Project structure

```
transpiler/
├── CMakeLists.txt              # Standalone CMake project
├── README.md                   # This file
├── raiser.hpp / raiser.cpp     # Orchestration: disassemble → IR → mem2reg
├── pipeline.hpp / pipeline.cpp # IR → llc → llvm-mc → ld.lld → HSACO
├── code_object_utils.hpp/.cpp  # ELF parsing, kernel metadata extraction
├── amdgpu_formats.hpp          # SIInstrFlags alias + diagnostic format labels
├── semop.hpp                   # Architecture-neutral semantic opcodes
├── isa_profile.hpp             # Per-target architectural properties
├── decoded_inst.hpp            # Disassembled instruction representation
├── parsed_reg.hpp              # Register classification helpers
├── mc_state.hpp/.cpp           # LLVM MC layer encapsulation
├── opcode_map.hpp/.cpp         # MC opcode → SemOp mapping
├── raise_context.hpp/.cpp      # Shared raiser state (RaiseContext)
├── reg_file.hpp                # SSA register file (AllocaRegFile)
├── kernarg_layout.hpp          # Kernel argument metadata
├── handlers.hpp                # Handler function declarations
├── handle_*.cpp                # Per-format instruction handlers (12 files)
├── tests/
│   ├── batch_raise_test.cpp    # Batch raise rate test (no GPU)
│   ├── hotswap_load_test.cpp   # HSA-level integration test (GPU)
│   ├── salmon_intercept.cpp    # LD_PRELOAD shim for HIP integration
│   ├── salmon_hip_test.cpp     # End-to-end HIP integration test (GPU)
│   ├── vecadd_gfx950.co        # Pre-compiled gfx950 vecadd test kernel
│   ├── ir_gpu_test.cpp         # Same-ISA GPU test
│   ├── mfma_gpu_test.cpp       # MFMA GEMM GPU test
│   ├── mfma_gemm.hip           # MFMA kernel source (compile with hipcc)
│   ├── cross_arch_gpu_test.cpp # Cross-ISA GPU test
│   └── gfx1250_gpu_test.cpp    # gfx1250 → gfx942 GPU test
└── docs (*.md)                 # Design documents
```

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

## PyTorch / HIP integration

Salmon integrates into the AMD GPU software stack at **two layers**, so that
PyTorch (or any HIP application) can transparently load and execute kernels
compiled for a different ISA.

### Architecture

```
 ┌────────────────────────────────────────────────────────────────┐
 │  PyTorch / HIP application                                    │
 │  hipModuleLoadData / __hipRegisterFatBinary                   │
 └────────────────────┬─────────────────────────────────────────-─┘
                      │
                      ▼
 ┌────────────────────────────────────────────────────────────────┐
 │  Layer 1: HIP fat binary intercept  (hip_fatbin.cpp in CLR)   │
 │                                                                │
 │  COMGR extracts code object from fat binary:                  │
 │    Pass 1: query native ISA (e.g. gfx942) → not found        │
 │    Pass 2: query cross-gen ISA (e.g. gfx950) → found         │
 │                                                                │
 │  Patch ELF metadata so HIP's ISA check accepts it:            │
 │    e_flags:  0x4e (gfx950) → 0x42 (gfx942)                   │
 │    .note:    "gfx950" → "gfx942"                              │
 │                                                                │
 │  *** REUSED FROM LEGACY HOTSWAP — no Salmon changes needed ** │
 └────────────────────┬─────────────────────────────────────────-─┘
                      │
                      ▼
 ┌────────────────────────────────────────────────────────────────┐
 │  Layer 2: HSA runtime hook  (loader/executable.cpp in ROCR)   │
 │                                                                │
 │  ExecutableImpl::LoadCodeObject detects ISA mismatch           │
 │  (original ISA preserved in ELF .note section)                │
 │                                                                │
 │  When HSA_HOTSWAP_IR_RAISER=1:                                │
 │    Salmon raises binary → LLVM IR → llc → lld → new HSACO    │
 │    Replaces the input ELF entirely (fresh code object)        │
 │                                                                │
 │  *** THIS IS SALMON ***                                       │
 └────────────────────┬─────────────────────────────────────────-─┘
                      │
                      ▼
 ┌────────────────────────────────────────────────────────────────┐
 │  GPU executes re-compiled code on target hardware             │
 └────────────────────────────────────────────────────────────────┘
```

### What Salmon reuses from the legacy hotswap

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

### Environment variables

| Variable | Required? | Description |
|----------|:---------:|-------------|
| `HSA_HOTSWAP_ISA_OVERRIDE` | **Yes** | Target ISA (e.g. `gfx942`).  Enables the hotswap hook. |
| `HSA_HOTSWAP_IR_RAISER` | **Yes** | Set to `1` to activate Salmon.  Without this, the legacy transpiler handles mismatches. |
| `HSA_HOTSWAP_RULES` | No | Path to JSON rules file.  Set to `/dev/null` if no rules are needed but the engine must be enabled. |
| `HSA_SALMON_DUMP_DIR` | No | Directory for intermediate files.  Each invocation creates a unique `salmon-XXXXXX/` subdirectory.  When unset, intermediates go to a temp dir that is cleaned up on exit. |
| `HSA_SALMON_DUMP_INPUT` | No | Set to `1` to also save the input code object (`input.co`) and per-kernel disassembly (`.dis`) alongside the raised IR. |

### Running PyTorch with Salmon

Once both layers are built (HIP with fat binary intercept + ROCR with Salmon):

```bash
export HSA_HOTSWAP_ISA_OVERRIDE=gfx942
export HSA_HOTSWAP_IR_RAISER=1
export HSA_HOTSWAP_RULES=/dev/null

python my_model.py  # gfx950 kernels transparently raised and re-compiled
```

### Building the full stack

Full PyTorch integration requires:

1. **ROCR runtime with Salmon** — Built with `-DROCR_ENABLE_HOTSWAP=ON
   -DROCR_ENABLE_IR_RAISER=ON` and a recent LLVM (see top of this README).
2. **HIP/CLR with fat binary intercept** — The `hip_fatbin.cpp` changes from
   the legacy hotswap branch.  These are in the `clr` repository
   (`hipamd/src/hip_fatbin.cpp`), not in the ROCR tree.
3. **PyTorch wheel** — Built against the above ROCR + HIP.

See `hotswap/docs/hotswap-wheel-integration.md` for the full TheRock-based
build procedure.

### Testing without the HIP layer

See `hotswap_load_test` in the [Running the tests](#running-the-tests) section
above.  It loads code objects directly via the HSA API, bypassing HIP entirely.
