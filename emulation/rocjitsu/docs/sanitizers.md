# Waitcheck and ConSan quick start

RocJITsu's combined HSA-tools hook runs two stages in order whenever an AMDGPU
code object is loaded:

1. **Waitcheck** analyzes every kernel in the original code object and reports
   missing AMDGPU waits. Its diagnostics are non-fatal.
2. **ConSan** instruments the same code object and loads the instrumented
   replacement when possible.

## Target support

| Target | Waitcheck | ConSan |
| --- | --- | --- |
| `gfx942` | Yes | — |
| `gfx950` | Yes | Partial |
| `gfx1100` | Yes | — |
| `gfx1150` | Yes | — |
| `gfx1151` | Yes | — |
| `gfx1200` | Yes | — |
| `gfx1201` | Yes | Yes |
| `gfx1250` | Yes | Yes |

Support is for native code objects; neither tool translates between GPU ISAs.
On waitcheck-only targets, the combined hook reports wait hazards and leaves
the original code object uninstrumented. “Partial” means native instrumentation
exists for only a subset of ConSan forms; “Yes” still denotes supported-form
coverage, not every ISA memory operation.

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
