# `test/microtest/standalone` — lightweight host-only build of the microtests

This is an **optional, standalone** way to build and run the `test/microtest`
suite (`rccl-UnitTestsMicro`) **without configuring/building all of librccl** — it
compiles just the tests + fakes + the hipified unit-under-test sources.

**ROCm is a prerequisite.** Per epic AICOMRCCL-1661 ("ROCm toolchain is
available"), this build uses `hipcc` in host-only mode (`--offload-host-only`)
against the **real ROCm headers**. There is no CPU-only / g++ path and no stubbed
`<hip/*>` / `<hsa/*>` / `<cuda*>` headers.

It exists alongside (does not replace) the in-tree `test/microtest/CMakeLists.txt`,
which builds the same tests as part of the full RCCL/hipcc build. Pull whichever
pieces are useful.

## What it does

- Compiles `p2p-test.cc` + the `fakes/` layer + the real (hipified) `p2p.cc`
  with **hipcc `--offload-host-only`** — a normal host compile (no amdgcn device
  pass) that still resolves the real ROCm headers. `llvm-cov`/`llvm-profdata` from
  the same ROCm toolchain line up with the instrumented binary.
- Links **gtest + fmt only** (both via FetchContent), and passes `-no-hip-rt` so it
  links **neither `librccl.so` nor the HIP runtime** — every HIP symbol the tests
  reach is provided by `fakes/` + `micro_link_stubs.cc`.
- Builds with llvm source-based coverage by default (`-DMICRO_COVERAGE=OFF` to skip).
- Result on WSL2 Ubuntu / ROCm 7.2.4: **30/30 tests pass, ~1 ms, `ldd` shows no
  HIP/ROCm/HSA/RCCL.**

## Prerequisites

- **ROCm with `hipcc`.** Auto-detected via `ROCM_PATH` (default `/opt/rocm`), or
  pass `-DCMAKE_CXX_COMPILER=/path/to/hipcc` / `-DROCM_PATH=/opt/rocm-7.x`.
- A **hipified source snapshot** from a prior RCCL build — i.e.
  `projects/rccl/build/<release|debug>/hipify` and `.../include`. Run the normal
  RCCL build once to produce it (this is the step that runs `hipify`), then point
  this build at it with `-DRCCL_BUILD_DIR=/path/to/projects/rccl/build/release`
  (default: `../../build/release`).
- Network access for the gtest/fmt FetchContent (or pre-populate `_deps`).

## Build & run

```bash
cd projects/rccl/test/microtest/standalone
cmake -B build -DCMAKE_BUILD_TYPE=Release \
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

## Caveats (read before adopting)

- **Needs the hipified source snapshot** (see Prerequisites): this build compiles
  the hipified UUT sources directly and does not run `hipify` itself.
- This is host-side seam/unit coverage, not integration coverage — keep the
  ROCm-linked and GPU/MPI suites for end-to-end behavior.
