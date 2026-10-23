---
myst:
    html_meta:
        "description": "Reference for the rocJITsu waitcheck hazard analyzer, covering CLI invocation, supported architectures, hazard classes, report format, exit codes, and limitations."
        "keywords": "rocJITsu, waitcheck, hazard, s_waitcnt, wait counter, AMD, ROCm, GPU, code object, static analysis"
---

# waitcheck hazard analyzer reference

waitcheck is a static object-code checker for AMDGPU wait hazards, distributed with rocJITsu. It analyzes final HSA code objects and reports missing or too-weak waits in the encoded instruction stream. waitcheck operates at the ISA level: the encoded instructions plus target features are the source of truth.

For a guided walkthrough, see [Analyze a code object for wait hazards with waitcheck](/tutorials/waitcheck-analyze-code-object.md).

## Offline CLI

The `rj_waitcheck` command analyzes code objects, HIP fat binaries, host executables, and directory trees.

### Invocation

``` text
rj_waitcheck INPUT [options]
```

`INPUT` is a file path or a directory (when `--recursive` is used).

#### Single-file analysis

``` shell
rj_waitcheck path/to/input.co
```

Select a target and code-object index when an executable contains multiple device images:

``` shell
rj_waitcheck app_or_fatbin --target gfx1250 --code-object-index 0
```

List the code objects in an input:

``` shell
rj_waitcheck app_or_fatbin --list-code-objects
```

#### Directory and corpus sweeps

``` shell
rj_waitcheck path/to/corpus \
  --recursive --all-code-objects --skip-unsupported --no-fail \
  --max-diagnostics 0 --stop-after-first-diagnostic --summary-only
```

Exhaustive target-specific sweep:

``` shell
rj_waitcheck /path/to/site-packages/torch \
  --exhaustive --target gfx950 --summary-only -j16 --slowest-kernels 10
```

Preserve every diagnostic as machine-readable JSONL:

``` shell
rj_waitcheck /path/to/site-packages/torch \
  --exhaustive --target gfx942 --summary-only --no-fail -j12 \
  --diagnostics-jsonl gfx942-diagnostics.raw.jsonl
```

### Options

| Option | Meaning |
| --- | --- |
| `--target gfx942\|gfx950\|gfx1100\|gfx1200\|gfx1201\|gfx1250` | Select one supported target from an executable input. |
| `--code-object-index N` | Select the Nth code object for the selected target. |
| `--kernel-entry OFFSET` | Analyze only the descriptor whose `.text` entry byte offset matches `OFFSET`. |
| `--all-code-objects` | Analyze all supported code objects in each input. |
| `--recursive` | Expand directory inputs into recursive file sweeps. |
| `--exhaustive` | Strict target-specific recursive sweep with code-object and kernel completeness totals. Requires `--target`. Implies `--recursive --all-code-objects`. |
| `--progress` | Show exhaustive kernel progress even when standard error is not an interactive terminal. |
| `--no-progress` | Disable exhaustive kernel progress, including on an interactive terminal. |
| `-j N`, `--jobs N` | Analyze up to N kernels concurrently. Default is 1, maximum is 16. |
| `--slowest-kernels N` | Report the N slowest kernels after an all-code-object or exhaustive sweep. |
| `--diagnostics-jsonl PATH` | Losslessly write one JSON object per retained per-kernel diagnostic. Requires `--all-code-objects` or `--exhaustive`. |
| `--skip-unsupported` | Skip unparsable inputs, inputs with no supported code object, or unsupported analysis failures. |
| `--max-diagnostics N` | Limit collected and printed diagnostics. Use `0` to suppress diagnostic payloads while preserving counts. |
| `--stop-after-first-diagnostic` | Stop each code object after the first observed hazard. |
| `--summary-only` | Print only final batch totals. |
| `--no-fail` | Return success even when hazards are reported. |
| `--list-code-objects` | List extractable code objects in the input and exit. |

## Supported architectures

waitcheck supports the following targets:

-   `gfx942` (CDNA3)
-   `gfx950` (CDNA4)
-   `gfx1100` (RDNA3)
-   `gfx1200` (RDNA4)
-   `gfx1201` (RDNA4)
-   `gfx1250`

See [Supported GPU architectures](/reference/supported-architectures.md) for the full list of architectures supported by rocJITsu.

## Hazard classes

The analyzer models the following wait-counter hazard classes on supported targets:

### Split wait-counter hazards

On GFX12 targets, the analyzer tracks split `loadcnt`, `storecnt`, `dscnt`, `kmcnt`, `samplecnt`, `bvhcnt`, and `expcnt` hazards.

### Legacy combined wait hazards

On CDNA3 and CDNA4, the analyzer tracks legacy `VMcnt`, `LGKMcnt`, and `EXPcnt` waits. On RDNA3, it additionally tracks the separate `VScnt`.

### Out-of-order scalar-memory completion

On RDNA4, scalar-memory operations can complete out of order. This requires `kmcnt(0)` for dependencies on a particular SMEM result.

### `s_wait_alu` dependency-counter hazards

The analyzer tracks `s_wait_alu` `depctr` hazards including `depctr_vm_vsrc`, `depctr_va_sdst`, `depctr_va_vcc`, and `depctr_sa_sdst`.

### Embedded wait fields

The analyzer recognizes embedded wait fields in `DSDIR` (`wait_vm_vsrc`, `wait_va_vdst`), `VINTERP` (`wait_exp`), and gfx1250 `s_set_vgpr_msb` high-VGPR bank selection.

### CFG-aware analysis

The forward dataflow analysis handles CFG joins, skipped paths, and loop-carried hazards.

A diagnostic is reported when a later instruction uses, overwrites, orders after, or ends the program before the relevant event has been waited on strongly enough.

## Report format

waitcheck prints diagnostics to standard error. Each diagnostic identifies the producer instruction, the consumer instruction, and the missing or too-weak wait between them. Locations are reported as section-relative offsets such as `.text+0xd1b44`.

With `--diagnostics-jsonl`, each diagnostic is written as one JSON object per line. Each record carries the exact input path, code-object index, kernel name and entry point, producer and consumer instructions, and a replay command.

When `--summary-only` is used, only final batch totals are printed. For `--exhaustive` sweeps, the summary reports completed versus discovered code objects and kernel descriptors as `code-objects=C/D kernels=C/D`. A complete sweep has matching counts and `analysis-errors=0`.

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Analysis succeeded and no hazards were found, or `--no-fail` was set. |
| `1` | Command-line usage error. |
| `2` | Input selection, parsing, or analysis error, including an incomplete `--exhaustive` sweep. |
| `4` | One or more hazards were found. |

## Runtime HSA tool

`librocjitsu_waitcheck_hooks.so` checks the final code-object reader passed to the ROCR runtime loader via HSA tools.

``` shell
HSA_TOOLS_DISABLE_REGISTER=1 \
HSA_TOOLS_LIB="build/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_waitcheck_hooks.so" \
  ROCJITSU_WAITCHECK_FAIL=1 \
  ./app
```

`HSA_TOOLS_DISABLE_REGISTER=1` selects the environment-driven HSA tools path on ROCR builds that also support rocprofiler registration.

When combining waitcheck with the rocJITsu DBT tool, list waitcheck first so it checks the final translated code:

``` shell
HSA_TOOLS_DISABLE_REGISTER=1 \
HSA_TOOLS_LIB="build/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_waitcheck_hooks.so build/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_hooks.so" \
  ROCJITSU_WAITCHECK_FAIL=1 \
  ./app
```

### Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `ROCJITSU_WAITCHECK` | `1` | Set to `0` to disable checking while leaving the HSA tool loaded. |
| `ROCJITSU_WAITCHECK_MODE` | `dispatch` | `dispatch` checks a kernel immediately before its first AQL dispatch and caches the result. `eager` exhaustively checks every kernel when its code object is loaded. |
| `ROCJITSU_WAITCHECK_FAIL` | `0` | Set to `1` to stop on a missing wait. Dispatch mode aborts before submitting the bad packet; eager mode rejects the load with `HSA_STATUS_ERROR_INVALID_CODE_OBJECT`. |
| `ROCJITSU_WAITCHECK_SUMMARY` | `0` | Set to `1` to print load, dispatch, check, cache, pass, and hazard counters at shutdown or process exit. |

In dispatch mode, descriptors are indexed at code-object load, mapped to runtime `kernel_object` values after executable freeze, and not decoded or analyzed until the kernel is actually submitted. This avoids whole-library analysis costs for large code objects. Use `ROCJITSU_WAITCHECK_MODE=eager` for load-only validation or when the application does not create an interceptable HSA queue.

## Known limitations

-   waitcheck validates final encoded programs. It does not rewrite code.
-   The analyzer is only as complete as the ISA metadata available to rocJITsu. Missing implicit operands, instruction-class flags, wait-counter effects, or target predicates can cause false negatives or false positives.
-   Hazards whose architectural fix is instruction spacing, `s_nop`, or `s_delay_alu` are separate from wait-counter validation unless they also expose a wait-like dependency.
-   Relocatable ELF objects (`.o` files) are rejected because they are compiler intermediates rather than final loadable code objects.
-   Targets outside the supported set listed above are out of scope.
