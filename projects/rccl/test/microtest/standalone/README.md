# `test/microtest/standalone` — CPU-only, no-ROCm build of the microtests

This is an **optional, standalone** way to build and run the `test/microtest`
suite (`rccl-UnitTestsMicro`) with **plain clang++ and no ROCm toolchain** — e.g.
on a stock `ubuntu:24.04`/WSL2 box with no GPU and no `/opt/rocm`.

It exists alongside (does not replace) the in-tree `test/microtest/CMakeLists.txt`,
which builds the same tests via the full RCCL/hipcc build. Pull whichever pieces
are useful.

## What it does

- Compiles `p2p-test.cc` + the `fakes/` layer + the real (hipified) `p2p.cc`
  with **clang++** (matches ROCm's LLVM, so `llvm-cov`/`llvm-profdata` line up).
- Links **gtest + fmt only** (both via FetchContent) — no `librccl.so`, no HIP runtime.
- Shadows only the ROCm **system** headers (`<hip/*>`, `<hsa/*>`, `<cuda*>`) with
  stubs in `stubs/`; the real RCCL headers resolve from the hipified tree.
- Builds with llvm source-based coverage by default (`-DMICRO_COVERAGE=OFF` to skip).
- Result on WSL2 Ubuntu 26.04 / clang-21: **30/30 tests pass, ~1 ms, `ldd` shows no HIP/ROCm/HSA.**

## Prerequisites

- clang++ / llvm (clang-20+; tested on clang-21).
- A **hipified source snapshot** from a prior ROCm build — i.e.
  `projects/rccl/build/<release|debug>/hipify/src` and `.../include`. On a machine
  with no ROCm you cannot run hipify, so copy this tree from a build box. Point the
  build at it with `-DRCCL_BUILD_DIR=/path/to/projects/rccl/build/release` (default:
  `../../build/release`).
- Network access for the gtest/fmt FetchContent (or pre-populate `_deps`).

## Build & run

```bash
cd projects/rccl/test/microtest/standalone
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
      -DRCCL_BUILD_DIR=/path/to/projects/rccl/build/release
cmake --build build -j"$(nproc)"
./build/rccl-UnitTestsMicro
```

Coverage:
```bash
cd build
LLVM_PROFILE_FILE=micro.profraw ./rccl-UnitTestsMicro
llvm-profdata merge -sparse micro.profraw -o micro.profdata
llvm-cov report ./rccl-UnitTestsMicro -instr-profile=micro.profdata
```

## The one source change that makes this possible

`src/include/nccl_device/hip_compat.h` set `NCCL_DEVICE_COMPILE 1` whenever
`__HIP_PLATFORM_AMD__` is defined. A host build must define `__HIP_PLATFORM_AMD__`
(to select HIP paths), which then dragged the `nccl_device/*` **device** template
bodies (guarded by `NCCL_CHECK_CUDACC = NCCL_DEVICE_COMPILE`) into the host
compile — they cannot compile with a host compiler. This PR keys
`NCCL_DEVICE_COMPILE` on an actual device-compile macro
(`__HIP_DEVICE_COMPILE__` / `__CUDA_ARCH__`) while keeping the platform flag
separate. This is correct for real hipcc builds too (the device pass still sets
it to 1) and is the load-bearing change for any host-only build.

## Caveats (read before adopting)

- **Not fully self-contained on a no-ROCm box:** it still needs the hipified
  source snapshot (see Prerequisites). A future improvement is a checked-in
  hipified snapshot or a sed-based hipify step.
- The stub HIP headers (`stubs/hip/hip_runtime.h`) declare the HIP entry points
  the fakes define and provide types/enums/device-intrinsics. They guard the
  clang-native builtins (`__hip_atomic_*`, `__builtin_nontemporal_*`) behind
  `#ifndef __clang__` so g++ also works.
- This is host-side seam/unit coverage, not integration coverage — keep the
  ROCm-linked and GPU/MPI suites for end-to-end behavior.
