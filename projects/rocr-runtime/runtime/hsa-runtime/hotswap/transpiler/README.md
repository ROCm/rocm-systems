# Hotswap Transpiler — Build & Run Instructions

## Overview

The hotswap transpiler lifts native AMDGPU machine code to LLVM IR for
cross-ISA binary translation.  It is a standalone CMake project that links
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

### batch_raise_test (no GPU needed)

The main test.  Raises every kernel in a code object and reports the raise rate:

```bash
# Single code object:
./batch_raise_test /path/to/kernel.co

# Directory (recursively finds all .co/.hsaco files):
./batch_raise_test /path/to/kernel_directory/

# Override ISA (normally auto-detected from ELF metadata):
./batch_raise_test /path/to/kernel.co gfx942
```

Example output:

```
=== BATCH RAISE SUMMARY ===
Total kernels:  27
Raised:         27  (100.0%)
Failed:          0
```

### GPU execution tests (optional)

These require HIP, ROCm, and an AMD GPU.  Add HIP to the CMake configuration:

```bash
cmake .. -G Ninja \
  -DCMAKE_PREFIX_PATH="/opt/rocm;$HOME/shared-llvm" \
  -DLLVM_INSTALL_DIR=$HOME/shared-llvm \
  -Dhip_DIR=/opt/rocm/lib/cmake/hip \
  -DCMAKE_CXX_COMPILER=clang++

ninja
```

Adjust `/opt/rocm` to your ROCm install path (e.g., `/opt/rocm-7.2.1`).

Then run:

```bash
./ir_gpu_test           # same-ISA vecadd
./mfma_gpu_test         # same-ISA MFMA GEMM
./cross_arch_gpu_test   # cross-ISA (gfx1200/1250 → gfx942)
./gfx1250_gpu_test      # gfx1250 Triton kernels → gfx942
```

**Note:** `cross_arch_gpu_test` has rocBLAS HSACO paths hardcoded to
`/opt/rocm-7.2.1/...` in `CMakeLists.txt`.  If you have a different ROCm
version, edit the `NATIVE_HSACO` and `SOURCE_HSACO` variables before
configuring.

---

## Project structure

```
transpiler/
├── CMakeLists.txt              # Standalone CMake project
├── README.md                   # This file
├── raiser.hpp / raiser.cpp     # Orchestration: disassemble → IR → mem2reg
├── pipeline.hpp / pipeline.cpp # IR → llc → llvm-mc → ld.lld → HSACO
├── code_object_utils.hpp/.cpp  # ELF parsing, kernel metadata extraction
├── amdgpu_formats.hpp          # TSFlags → FormatKind classification
├── semop.hpp                   # Architecture-neutral semantic opcodes
├── isa_profile.hpp             # Per-target architectural properties
├── decoded_inst.hpp            # Disassembled instruction representation
├── parsed_reg.hpp              # Register classification helpers
├── mc_state.hpp/.cpp           # LLVM MC layer encapsulation
├── opcode_map.hpp/.cpp         # MC opcode → SemOp mapping
├── canonicalize.hpp/.cpp       # Mnemonic normalization across ISAs
├── raise_context.hpp/.cpp      # Shared raiser state (RaiseContext)
├── reg_file.hpp                # SSA register file (AllocaRegFile)
├── kernarg_layout.hpp          # Kernel argument metadata
├── handlers.hpp                # Handler function declarations
├── handle_*.cpp                # Per-format instruction handlers (12 files)
├── tests/
│   ├── batch_raise_test.cpp    # Batch raise (no GPU)
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
