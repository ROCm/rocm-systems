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

The `rj_waitcheck` command analyzes final AMDGPU ELF code objects, host ELF files
containing HIP bundles, and directory trees of those files. A standalone
compiler `.bundle` file is not an ELF input; extract its `.hsaco` payload before
passing it to waitcheck.

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
rj_waitcheck host_elf_or_code_object --target gfx1250 --code-object-index 0
```

List the code objects in an input:

``` shell
rj_waitcheck host_elf_or_code_object --list-code-objects
```

List exact kernel descriptors as JSONL, or extract AMDGPU ELFs for inspection:

``` shell
rj_waitcheck host_elf_or_code_object --list-kernels
rj_waitcheck host_elf_or_code_object --target gfx1250 --code-object-index 0 \
  --extract-code-object selected.co
rj_waitcheck host_elf_or_code_object --target gfx1250 \
  --extract-code-objects extracted/
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
| `--target TARGET` | Select a target from the supported architectures listed below. |
| `--code-object-index N` | Select the Nth code object for the selected target. |
| `--kernel-entry OFFSET` | Analyze each merged entry/wave group at this `.text` byte offset. Same-wave descriptors are conservatively grouped, including register allocation. Use `--wave-size` to select one wave mode. This flag cannot select an individual descriptor for a dispatch-specific check. Offline callers iterating `waitcheck_kernels()` should pass each `WaitcheckKernelInfo` to the core overload to avoid repeating groups. |
| `--wave-size 32\|64` | Select a descriptor wave size at `--kernel-entry`, or supply the wave size for a descriptorless entry. Requires `--kernel-entry`; a descriptorless entry remains incomplete without VGPR allocation. Descriptors keep their encoded wave size. |
| `--all-code-objects` | Analyze all supported code objects in each input. |
| `--recursive` | Expand directory inputs into recursive file sweeps. |
| `--exhaustive` | Strict target-specific recursive sweep with code-object and kernel completeness totals. Requires `--target`. Implies `--recursive --all-code-objects`. |
| `--progress` | Show exhaustive kernel progress even when standard error is not an interactive terminal. |
| `--no-progress` | Disable exhaustive kernel progress, including on an interactive terminal. |
| `-j N`, `--jobs N` | Analyze up to N kernels concurrently within each input's all-code-object analysis. Inputs are scanned in sequence. Default is 1, maximum is 16. |
| `--slowest-kernels N` | Report the N slowest kernels after an all-code-object or exhaustive sweep. |
| `--diagnostics-jsonl PATH` | Losslessly write one JSON object per retained per-kernel diagnostic. Requires `--all-code-objects` or `--exhaustive`. |
| `--check-counter-parity` | Compare encoded wait fields with modeled required counts. |
| `--counter-parity-jsonl PATH` | Write counter-parity findings as JSONL. Requires `--all-code-objects` or `--exhaustive`. |
| `--skip-unsupported` | Skip unparsable inputs, inputs with no supported code object, or unsupported analysis failures. |
| `--max-diagnostics N` | Limit collected and printed diagnostics. Use `0` to suppress diagnostic payloads while preserving counts. |
| `--stop-after-first-diagnostic` | Stop each code object after the first observed hazard. |
| `--summary-only` | Print only final batch totals. |
| `--no-fail` | Return success even when hazards are reported. |
| `--list-code-objects` | List extractable code objects in the input and exit. |
| `--list-kernels` | List every exact kernel descriptor as JSONL and exit. The code-object row includes `kernel_count` for exact descriptors and `analysis_group_count` for merged entry/wave groups. Each descriptor row names its `analysis_group_name` and all `analysis_group_descriptor_vaddrs`; exhaustive summaries count groups. |
| `--extract-code-object PATH` | Write the selected AMDGPU ELF to `PATH` and exit. |
| `--extract-code-objects DIR` | Parse one input once, write each selected AMDGPU ELF to `DIR`, and emit its target, index, and path as JSONL. Existing files are reused only when their bytes match. |

## Supported architectures

waitcheck supports the following targets:

-   `gfx942` (CDNA3)
-   `gfx950` (CDNA4)
-   `gfx1100` (RDNA3)
-   `gfx1150` and `gfx1151` (RDNA3.5)
-   `gfx1200` (RDNA4)
-   `gfx1201` (RDNA4)
-   `gfx1250` and `gfx1251` (CDNA5)

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

waitcheck prints analysis summaries and diagnostics to standard output. Progress and analysis errors are printed to standard error. Each diagnostic identifies the producer instruction, the consumer instruction, and the missing or too-weak wait between them. Locations are reported as section-relative offsets such as `.text+0xd1b44`.

With `--diagnostics-jsonl`, each diagnostic is written as one JSON object per line. Each record carries the exact input path, code-object index, producer and consumer instructions, and a replay command. Kernel name and entry point are present when the code object supplies a kernel descriptor.

When `--summary-only` is used, only final batch totals are printed. For `--exhaustive` sweeps, the summary reports completed versus discovered code objects and merged entry/wave groups as `code-objects=C/D kernels=C/D`. A complete sweep has matching counts and `analysis-errors=0`.

## Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Analysis succeeded and no hazards were found, or `--no-fail` was set. |
| `1` | Command-line usage error. |
| `2` | Input selection, parsing, or analysis error, including an incomplete `--exhaustive` sweep. |
| `4` | One or more hazards were found. |

## Known limitations

-   waitcheck validates final encoded programs. It does not rewrite code.
-   The analyzer is only as complete as the ISA metadata available to rocJITsu. Missing implicit operands, instruction-class flags, wait-counter effects, or target predicates can cause false negatives or false positives.
-   Hazards whose architectural fix is instruction spacing, `s_nop`, or `s_delay_alu` are separate from wait-counter validation unless they also expose a wait-like dependency.
-   Relocatable ELF objects (`.o` files) are rejected because they are compiler intermediates rather than final loadable code objects.
-   Targets outside the supported set listed above are out of scope.
