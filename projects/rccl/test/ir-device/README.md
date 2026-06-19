# RCCL IR Device Tests (`librccl_device.bc`)

## Description

This directory contains a pytest harness that builds and runs the GoogleTest
binary defined in `bindings/ir/test/IR_test.cpp`. That test exercises every
active API exported by `nccl_device_wrapper__impl.h` on real GPU hardware:

| Group | API |
|-------|-----|
| A | `ncclGetPeerPointerTeam` |
| B1–B5 | `ncclCoopAnyInit{Thread,Warp,Lanes,WarpSpan,Cta}` + `ncclCoopSize/ThreadRank/NumThreads` |
| B6 | `ncclCoopSync` (all five coop types) |
| B7 | `ncclLsaBarrierSession_C` structural check (runtime calls skipped) |

Unlike the rest of `rccl-UnitTests`, this test cannot be built by the default
RCCL CMake invocation: it links `librccl_device.bc` into the AMDGPU device-side
LTO and must be compiled by `hipcc` at `-O0` with special offload-linker flags.
The bitcode itself only exists when RCCL is configured with `-DEMIT_LLVM_IR=ON`.

The harness follows the same pattern as `test/ext-plugins`: the test artifact
is built on demand from a session fixture, and the suite is **skipped with a
clear reason** (rather than failing) when its prerequisites are missing.

The session fixture builds **both** layers on demand:
1. If `librccl_device.bc` is absent, it is emitted via the standalone
   `bindings/ir` Makefile (`make EMIT_LLVM_IR=1 …`) — the equivalent of a
   `-DEMIT_LLVM_IR=ON` build, but reusing the hipify-staged headers from an
   earlier RCCL build instead of recompiling RCCL.
2. The GoogleTest binary is then compiled against that bitcode.

## Prerequisites

You only need RCCL to have been built **once** so the hipify-staged
`nccl_device` headers and generated `nccl.h`/`rccl.h` exist (a normal
`-DBUILD_TESTS=ON` build is enough — `EMIT_LLVM_IR` is **not** required at
configure time because the harness emits the bitcode itself):

```bash
cmake -B build/release -DBUILD_TESTS=ON .
cmake --build build/release --target rccl
```

If you prefer, configuring with `-DEMIT_LLVM_IR=ON -DBITCODE_LIB_ARCH=gfx942`
and building the `llvm_ir` target pre-produces the bitcode, and the harness
will reuse it. GoogleTest is taken from `GTEST_ROOT` (the RCCL build installs
it under `build/release/gtest`, or point at a system install such as `/usr`).

## Configuration (environment variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `RCCL_DIR` | repo root (derived) | RCCL source root |
| `RCCL_BUILD` | `$RCCL_DIR/build/release` | RCCL CMake build dir |
| `ROCM_PATH` | `/opt/rocm` | ROCm install root |
| `ARCH` | `gfx942` | offload / bitcode architecture |
| `GTEST_ROOT` | `$RCCL_BUILD/gtest` | GoogleTest install prefix |
| `IR_OUTDIR` | `<cwd>/ir_test_build` | output dir for the compiled binary |

## Running

```bash
cd test/ir-device
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Build + run all IR device tests
pytest -v --cache-clear

# Run a single group
pytest -v -m peer_pointer
pytest -v -m coop
pytest -v -m lsa_barrier
```

Build and per-test logs are written to `logs/`.

## CI integration

This suite is intended to be run the same way `test/ext-plugins` is — a
`pytest` step in the test workflow, on a runner that built RCCL with
`-DEMIT_LLVM_IR=ON`. On runners that did not, every case auto-skips, so the
step is safe to add unconditionally.

Alternatively, because the `EMIT_LLVM_IR` flag can be passed through the
`test_runner` build configuration (`cmake_options`), the compiled binary can
be registered in a `test_runner` JSON config as an `is_gtest: true` test with
`binary: IR_test.exe` and `test_filter: IRDeviceTest.*`.
