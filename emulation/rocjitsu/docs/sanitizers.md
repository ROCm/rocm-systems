# Waitcheck and ConSan quick start

[![sanitizer artifacts](https://github.com/ROCm/rocm-systems/actions/workflows/rocjitsu-sanitizer-artifacts.yml/badge.svg?branch=shared%2Frocjitsu%2Fsanitizers)](https://github.com/ROCm/rocm-systems/actions/workflows/rocjitsu-sanitizer-artifacts.yml?query=branch%3Ashared%2Frocjitsu%2Fsanitizers)

The badge tracks the latest build of the sanitizer integration branch. Click it
for prebuilt binaries, or jump to [Download a prebuilt build](#download-a-prebuilt-build).

RocJITsu's combined HSA-tools hook runs two stages in order whenever an AMDGPU
code object is loaded:

1. **Waitcheck** analyzes every kernel in the original code object and reports
   missing AMDGPU waits. Its diagnostics are non-fatal.
2. **ConSan** instruments the same code object and loads the instrumented
   replacement when possible.

## Target support

| Target | Waitcheck | ConSan |
| --- | --- | --- |
| `gfx942` | Yes | Yes |
| `gfx950` | Yes | Yes |
| `gfx1100` | Yes | Yes |
| `gfx1150` | Yes | — |
| `gfx1151` | Yes | — |
| `gfx1200` | Yes | — |
| `gfx1201` | Yes | Yes |
| `gfx1250` | Yes | Yes |

Support is for native code objects; neither tool translates between GPU ISAs.
On waitcheck-only targets, the combined hook reports wait hazards and leaves
the original code object uninstrumented. “Yes” denotes the supported semantic
forms in the [ConSan capability matrix](consan/CAPABILITIES.md), not every ISA
memory operation.

ConSan takes the active workgroup-LDS capacity from the runtime agent rather
than baking a gfx942 size into its instrumentation. Simulator and offline tests
use the selected RocJITsu JSON configuration, including `lds_size_kb`, as their
source of truth.

## Build

From the `emulation/rocjitsu` source directory:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target rocjitsu_dbi_hooks rj_waitcheck
```

The two useful artifacts are:

```text
build/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so
build/tools/rj_waitcheck
```

See [Building RocJITsu](building.md) for dependencies and additional build
options.

## Download a prebuilt build

Every push to `shared/rocjitsu/sanitizers` publishes those same two artifacts
through the
[`rocjitsu-sanitizer-artifacts`](https://github.com/ROCm/rocm-systems/actions/workflows/rocjitsu-sanitizer-artifacts.yml?query=branch%3Ashared%2Frocjitsu%2Fsanitizers)
workflow, so you can skip the build entirely:

```sh
python3 emulation/rocjitsu/scripts/download_sanitizer_artifacts.py
```

That fetches the newest successful build. Pass `--run <id>` for a specific
build, `--branch` for a different branch, `--list` to inspect a run without
downloading, and `--dest` to choose the output directory. The script needs only
the Python standard library.

GitHub does not serve Actions artifacts anonymously, so a token with
`actions:read` scope is required even though the repository is public. The
script reads `--token`, then `$GITHUB_TOKEN`, then `$GH_TOKEN`, and finally
falls back to `gh auth token`; running `gh auth login` once is usually enough.

The bundle unpacks to a flattened layout and its SHA-256 digests are verified
on download:

```text
rocjitsu-sanitizers/
├── bin/rj_waitcheck
├── lib/librocjitsu_dbi_hooks.so
├── MANIFEST.json      # commit, run ID, build type, compiler, glibc baseline
└── sha256sums.txt
```

Point the steps below at those paths instead of a local `build/` tree:

```sh
export ROCJITSU_SANITIZER_HOOK="$PWD/rocjitsu-sanitizers/lib/librocjitsu_dbi_hooks.so"
```

The binaries are built in the same `therock_build_manylinux_x86_64` image the
rest of the ROCm builds use (AlmaLinux 8, glibc 2.28) with `libstdc++` linked
statically, so they carry no `libstdc++` dependency and run against any host
`glibc` at least as new as the `minimum_glibc` value recorded in
`MANIFEST.json`. Artifacts are retained for 30 days; build from source for
anything older.

## Run waitcheck and ConSan together

```sh
export ROCJITSU_BUILD="$PWD/build"
export ROCJITSU_SANITIZER_HOOK="$ROCJITSU_BUILD/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so"

env \
  HSA_TOOLS_DISABLE_REGISTER=1 \
  HSA_TOOLS_LIB="$ROCJITSU_SANITIZER_HOOK" \
  ./application
```

That is the complete ordinary setup. Do not also load the standalone waitcheck
hook or set `ROCJITSU_WAITCHECK*` variables. The combined hook always runs an
exhaustive load-time waitcheck first. A waitcheck diagnostic is printed with a
`rocjitsu-waitcheck:` prefix, then ConSan continues with DBI. Add
`RJ_CONSAN_LOG=1` only when you want verbose pass and instrumentation summaries.

Record/Replay is the default. Select another analysis with
`RJ_CONSAN_MODE=inline-shadow`, `sampled`, or `supercollider`. For a focused
test where incomplete instrumentation must fail, add
`RJ_CONSAN_POLICY=strict`; race diagnostics themselves remain non-fatal.

For ConSan engines, diagnostics, coverage, and expert controls, continue with
the [ConSan tutorial](consan/TUTORIAL.md) or [ConSan usage reference](consan/USAGE.md).

## Run waitcheck on a saved object

Waitcheck can also inspect a code object without running it:

```sh
"$ROCJITSU_BUILD/tools/rj_waitcheck" path/to/kernel.hsaco
```

It also accepts HIP fat binaries, executables, shared libraries, and directory
corpora. See the [waitcheck guide](waitcheck/README.md) for runtime-only use,
target selection, kernel filtering, corpus scans, and its C API.

## Minimal gfx1250 simulator repro

This runs native `gfx1250` code in the RocJITsu simulator; no DBT is involved.
Save the following as `/tmp/gfx1250-sanitizer-repro.hip`:

```cpp
#include <hip/hip_runtime.h>

#include <cstdint>

__global__ void lds_race(uint32_t *out) {
  __shared__ uint32_t value[64];
  const uint32_t tid = threadIdx.x;
  value[tid] = tid;
  if (tid >= 32)
    out[tid] = value[tid - 32]; // Wave 1 reads wave 0 without a barrier.
}

__global__ void missing_wait(const uint32_t *in, uint32_t *out) {
  const uint32_t *address = in + threadIdx.x;
  uint32_t value;
  asm volatile("global_load_b32 %0, %1, off\n\t"
               "v_mov_b32 %0, %0\n\t" // Deliberate use before the wait.
               "s_wait_loadcnt 0"
               : "=&v"(value)
               : "v"(address)
               : "memory");
  out[threadIdx.x] = value;
}

int main() {
  uint32_t *device = nullptr;
  if (hipMalloc(&device, 64 * sizeof(uint32_t)) != hipSuccess)
    return 1;
  if (hipMemset(device, 0, 64 * sizeof(uint32_t)) != hipSuccess)
    return 1;
  lds_race<<<1, 64>>>(device);
  missing_wait<<<1, 64>>>(device, device);
  const hipError_t result = hipDeviceSynchronize();
  const hipError_t cleanup = hipFree(device);
  return result == hipSuccess && cleanup == hipSuccess ? 0 : 1;
}
```

From the `rocm-systems` repository root, build it with the workspace's TheRock
SDK and run it through the combined hook:

```sh
export BUILD="$PWD/../build"
export ROCM_SDK_ROOT="$(../venv/bin/rocm-sdk path --root)"
export LD_LIBRARY_PATH="$ROCM_SDK_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

ninja -C "$BUILD" rocjitsu rocjitsu_dbi_hooks
../venv/bin/hipcc --offload-arch=gfx1250 -O1 -g \
  /tmp/gfx1250-sanitizer-repro.hip -o /tmp/gfx1250-sanitizer-repro

env \
  HSA_TOOLS_LIB="$BUILD/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so" \
  RJ_CONSAN_LOG=1 \
  "$BUILD/tools/rocjitsu/rocjitsu" \
    --config emulation/rocjitsu/configs/gfx1250.json -- \
    /tmp/gfx1250-sanitizer-repro 2>&1 | \
  grep -E 'rocjitsu-waitcheck:.*(missing|consumer)|ConSan MOI auto replay diagnostic|ConSan analysis verdict'
```

Expected output (IDs, registers, and code offsets may vary):

```text
rocjitsu-waitcheck: .text+0x...: missing s_wait_loadcnt <= 0 ... global_load_b32 ...
rocjitsu-waitcheck:   consumer: v_mov_b32_e32 ...
[rocjitsu-dbi-hooks] ConSan MOI auto replay diagnostic ... kind=1 ... first_lds=[0,4) second_lds=[0,4) first_kind=2 second_kind=1 ...
[rocjitsu-dbi-hooks] ConSan analysis verdict applicable=true analysis_complete=true static_complete=true dynamic_complete=true ...
```

The waitcheck lines identify the missing load wait. The ConSan line is a
write/read conflict (`first_kind=2`, `second_kind=1`) on the same four LDS
bytes. Remove the final `grep` to see the complete instrumentation log.
