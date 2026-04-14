# HotSwap Agent Rules

## Docker-Only Execution

All build, test, and execution commands MUST run inside the Docker container.
NEVER run cmake, ninja, make, gcc, clang, hipcc, python test scripts, or any
compiled binary directly on the host. The host has no LLVM AMDGPU backend,
no ROCm, and no AMD GPU.

The ONLY host-side commands allowed are:
- `docker build` / `docker run` / `docker exec`
- `git` operations (status, diff, add, commit)
- File editing (the source is volume-mounted into the container)
- `ls`, `cat`, `head`, `tail` for inspecting local files

## Getting Started

From the repo root (`rocm-systems/`):

```bash
# 1. Build the Docker image (installs ROCm LLVM, cmake, ninja, etc.)
docker build -t hotswap-test projects/rocr-runtime/runtime/hsa-runtime/hotswap/

# 2. Start the container (with GPU passthrough)
docker run -d --name hotswap-dev \
  --device /dev/kfd --device /dev/dri \
  --group-add video --ipc=host --shm-size 8G \
  -v "$(pwd)":/workspace \
  -w /workspace/projects/rocr-runtime/runtime/hsa-runtime/hotswap \
  hotswap-test

# 3. Configure and build
docker exec hotswap-dev cmake -B build -S . -GNinja \
  -DLLVM_DIR=/opt/rocm/llvm/lib/cmake/llvm \
  -DCMAKE_BUILD_TYPE=Release
docker exec hotswap-dev ninja -C build
```

If the container already exists, restart it with `docker start hotswap-dev`.
If no AMD GPU is available, omit the `--device` and `--group-add` flags.

All commands inside the container use `docker exec hotswap-dev bash -c "..."`.
Always use `bash -c` — never rely on file execute permissions.

## Running Tests

```bash
# C++ unit tests (no GPU needed):
docker exec hotswap-dev ./build/hotswap_test
docker exec hotswap-dev ./build/lifter_test

# MVE GPU round-trip test (requires GPU):
docker exec hotswap-dev ./build/mve_gpu_test

# Python mnemonic translation tests (no GPU needed, requires llvm-mc):
docker exec hotswap-dev python3 tests/test_transpiler.py

# Python end-to-end transpiler test (requires llvm-mc):
docker exec hotswap-dev python3 tests/test_transpiler_e2e.py

# Or use the convenience script that builds + runs everything:
docker exec hotswap-dev bash run_tests.sh          # full suite
docker exec hotswap-dev bash run_tests.sh --no-gpu  # Tier 1 only

# Rebuild after code changes:
docker exec hotswap-dev ninja -C build
```

## File Layout

```
hotswap/
  hotswap.cpp, hotswap.hpp         -- Rewrite engine + retarget + ELF patcher
  hotswap_rules.cpp, .hpp          -- JSON rule parser (zero external deps)
  trampoline.cpp, .hpp             -- Out-of-line code patching
  transpiler.cpp, .hpp             -- Cross-family gfx1250 -> gfx942 transpiler
  lifter.cpp, .hpp                 -- Binary lifter: MCInst -> waveasm MLIR
  ssa_construction.cpp             -- Physical -> virtual SSA register pass
  cross_target.cpp, .hpp           -- Cross-target mnemonic mapping
  wave_width.cpp, .hpp             -- Wave width translation (wave32 -> wave64)
  emit_assembly.cpp, .hpp          -- waveasm IR -> AMDGPU assembly text
  pipeline.cpp, .hpp               -- Full MLIR pipeline orchestration
  code_object_builder.cpp, .hpp    -- ELF extraction + code object rebuild
  CMakeLists.txt                   -- Standalone build (library + test binary)
  Dockerfile                       -- ROCm dev image for testing
  run_tests.sh                     -- Test runner (--no-gpu for Tier 1 only)
  AGENTS.md                        -- This file
  WORK_ITEMS.md                    -- Task tracking
  waveasm/                         -- waveasm MLIR dialect (submodule)
  cmake/                           -- CMake helper scripts
  tests/
    hotswap_test.cpp               -- C++ unit tests (no GPU needed)
    lifter_test.cpp                -- MLIR pipeline unit tests (no GPU needed)
    mve_gpu_test.cpp               -- GPU round-trip MVE test (requires GPU)
    mve_vecadd.hip                 -- MVE test kernel source
    test_transpiler.py             -- Mnemonic translation validation (needs llvm-mc)
    test_transpiler_e2e.py         -- End-to-end transpile round-trip (needs llvm-mc)
    *.hip                          -- GPU kernel test sources
    test_rules.json                -- Example rewrite rules
```

## Build Artifacts

The build directory is `hotswap/build/`. It is created inside the container
via the volume mount, so it persists across `docker exec` calls. To clean:

```bash
docker exec hotswap-dev rm -rf build
```

## Environment Variables

Inside the container, these are pre-set or have defaults:
- `ROCM_PREFIX` -- defaults to `/opt/rocm`
- `LLVM_MC` -- defaults to `/opt/rocm/llvm/bin/llvm-mc`
- `LLVM_OBJDUMP` -- defaults to `/opt/rocm/llvm/bin/llvm-objdump`

## Error Handling Policy

**No fallbacks. No silent failures.** If something doesn't work, fail loudly
with a clear error message so it can be fixed. Specifically:

- Never catch exceptions and silently continue (`if Exception: pass`).
- Never substitute a "best effort" fallback when the correct path fails.
- Never emit `waveasm.raw` as a silent fallback — if an instruction can't be
  properly lifted or translated, report it as an error.
- Never skip a failing pass and hope downstream compensates.
- Assertions and `llvm::report_fatal_error` are preferred over quiet degradation.

The goal is correctness. A loud crash on one kernel is better than a silent
miscompilation on many.

## Key Constraint

DO NOT install packages on the host. DO NOT compile on the host. DO NOT run
test binaries on the host. Everything goes through `docker exec hotswap-dev`.
