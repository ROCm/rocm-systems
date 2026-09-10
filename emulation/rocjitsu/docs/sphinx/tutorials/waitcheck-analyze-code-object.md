---
myst:
    html_meta:
        "description": "Step-by-step tutorial for analyzing an AMDGPU code object for wait hazards using the rocJITsu waitcheck tool."
        "keywords": "rocJITsu, waitcheck, wait hazard, s_waitcnt, AMDGPU, code object, ROCm, static analysis"
---
# Analyze a code object for wait hazards with waitcheck

AMD GPU kernels coordinate memory operations through explicit wait
instructions such as `s_waitcnt` and the GFX12 split `s_wait_*` family.
When a wait is missing or too weak, the hardware might read a register
before the memory operation that produces its value has completed. These
*wait hazards* cause silent data corruption that is difficult to
reproduce and diagnose at runtime.

rocJITsu's **waitcheck** tool performs a static, forward dataflow
analysis over the final encoded instruction stream in an AMDGPU code
object. It tracks outstanding wait-counter events across the
control-flow graph and reports every instruction pair where a consumer
uses a result before the relevant counter has been waited on strongly
enough.

Use this tutorial when you need to verify that a compiled HIP kernel, a
kernel generator's output, or a handwritten assembly routine contains
correct waits before you deploy it to hardware.

## Supported targets

Waitcheck supports the following targets: `gfx942` (CDNA3), `gfx950`
(CDNA4), `gfx1100` (RDNA3), `gfx1200` (RDNA4), `gfx1201` (RDNA4), and
`gfx1250`.

## Hazard classes

Waitcheck detects three broad classes of wait hazards:

### Split counter hazards

On GFX12 targets, memory operations retire through separate counters:
`loadcnt`, `storecnt`, `dscnt`, `kmcnt`, `samplecnt`, `bvhcnt`, and
`expcnt`. A hazard is reported when an instruction consumes a register
that was written by an in-flight memory operation whose specific counter
has not been waited on.

### Legacy counter hazards

On CDNA3 and CDNA4 targets, `s_waitcnt` encodes combined `vmcnt`,
`lgkmcnt`, and `expcnt` fields. RDNA3 adds a separate `vscnt`. A hazard
is reported when the encoded wait value is not strong enough (the
counter threshold is too high) to retire the operation that produces the
consumed value.

### ALU dependency hazards

`s_wait_alu` dependency-counter hazards cover cases such as
`depctr_vm_vsrc`, `depctr_va_sdst`, `depctr_va_vcc`, and
`depctr_sa_sdst`, where a scalar or vector ALU result is consumed before
the pipeline stage that produces it has drained. Embedded wait fields in
DSDIR and VINTERP instructions are also tracked.

## Step 1: compile a HIP kernel

Compile a kernel targeting one of the supported architectures. The
`--offload-arch` flag must match the target you intend to analyze:

``` bash
hipcc -O2 -o vector_add vector_add.hip --offload-arch=gfx950
```

The resulting `vector_add` executable is a HIP fat binary that contains
one or more embedded AMDGPU code objects.

## Step 2: run waitcheck

Pass the compiled binary to the `rj_waitcheck` CLI:

``` bash
build/tools/rj_waitcheck vector_add --target gfx950
```

If the binary contains multiple code objects for the same target, list
them first and select one by index:

``` bash
build/tools/rj_waitcheck vector_add --list-code-objects
build/tools/rj_waitcheck vector_add --target gfx950 --code-object-index 0
```

When the analysis finds no hazards, the tool exits with code `0` and
produces no diagnostic output.

## Step 3: interpret the report

When a hazard is found, waitcheck prints a diagnostic that identifies
the *producer* instruction (the memory or ALU operation that writes a
result), the *consumer* instruction (the instruction that reads that
result too early), and the wait that is missing or insufficient between
them.

A typical diagnostic identifies:

-   The section-relative offset of the producer and consumer (for
    example `.text+0x1a0` and `.text+0x1b8`).
-   The producer and consumer mnemonics.
-   The counter and threshold that would resolve the hazard.

Use `rj_co` to inspect the surrounding instructions when you need more
context around a reported offset:

``` bash
rj_co vector_add --target gfx950 \
  --disassemble-window .text+0x1a0 --context-bytes 128
```

## Step 4: apply the fix and re-analyze

Insert or strengthen the appropriate wait instruction in your source,
generator, or assembly. For a legacy-counter target the fix is typically
an `s_waitcnt` with a tighter counter value; for a GFX12 target the fix
is the corresponding split wait (`s_wait_loadcnt`, `s_wait_dscnt`, and
so on).

Rebuild the code object and rerun waitcheck:

``` bash
hipcc -O2 -o vector_add vector_add.hip --offload-arch=gfx950
build/tools/rj_waitcheck vector_add --target gfx950
```

Repeat until waitcheck reports no hazards (exit code `0`).

## Corpus sweeps

For large collections of code objects --- such as a PyTorch
`site-packages` tree or a Tensile library --- use the `--exhaustive`
flag with a target to sweep every kernel entry point:

``` bash
build/tools/rj_waitcheck /path/to/site-packages/torch \
  --exhaustive --target gfx950 --summary-only -j16
```

`--exhaustive` implies `--recursive --all-code-objects` and requires
every selected code object to decode and analyze successfully.

## Runtime checking

To check kernels at dispatch time without modifying the application,
load the waitcheck HSA tools library:

``` bash
HSA_TOOLS_LIB="build/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_waitcheck_hooks.so" \
  ROCJITSU_WAITCHECK_FAIL=1 \
  ./vector_add
```

With `ROCJITSU_WAITCHECK_FAIL=1`, the tool aborts before submitting a
kernel that contains a wait hazard. Without that flag, hazards are
reported to stderr and the kernel is submitted normally.

## Next steps

-   See [waitcheck hazard analyzer reference](../reference/waitcheck.md) for the
    full CLI option reference, environment variables, exit codes, and
    runtime tool configuration.
-   See [Inspect and disassemble a code object with the C API](inspect-code-object.md)
    for loading and disassembling code objects through the rocJITsu C
    API.
