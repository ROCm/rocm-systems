# Waitcheck

Waitcheck is an object-code checker for AMDGPU wait hazards. It treats the final
encoded instruction stream as the contract: if a kernel generator emits ISA
directly, there is no LLVM MIR state to preserve or consult. Waitcheck inspects
final HSA code objects and reports missing or too-weak waits in the program that
the hardware will execute.

It is intended for two workflows:

- `rj_waitcheck`, an offline CLI for code objects, HIP fat binaries, and
  recursive corpus sweeps.
- `librocjitsu_waitcheck.so`, an `LD_PRELOAD` shim that checks code objects as
  ROCR creates code-object readers.

The LLVM parity map is used as a regression checklist for known wait patterns,
not as a requirement that kernels came from LLVM. See
[`../waitcheck-llvm-parity.md`](../waitcheck-llvm-parity.md).

## Build

From the RocJITsu workspace:

```sh
cmake --build build --target rj_waitcheck rocjitsu_waitcheck_shim
```

The expected build products are:

- `build/tools/rj_waitcheck`
- `build/lib/rocjitsu/src/rocjitsu/kmd/linux/librocjitsu_waitcheck.so`

## Offline CLI

Analyze one input:

```sh
build/tools/rj_waitcheck path/to/input.co
```

Select a target and code-object index when an executable contains multiple
device images:

```sh
build/tools/rj_waitcheck app_or_fatbin --target gfx1250 --code-object-index 0
```

List supported code objects in an input:

```sh
build/tools/rj_waitcheck app_or_fatbin --list-code-objects
```

Sweep a directory or corpus:

```sh
build/tools/rj_waitcheck "$HOME/rocjitsu/rocjitsu-corpus/corpus" \
  --recursive --all-code-objects --skip-unsupported --no-fail \
  --max-diagnostics 0 --stop-after-first-diagnostic --summary-only
```

Useful options:

| Option | Meaning |
| --- | --- |
| `--target gfx950|gfx1200|gfx1201|gfx1250` | Select one supported target from an executable input. |
| `--code-object-index N` | Select the Nth code object for the selected target. |
| `--all-code-objects` | Analyze all supported code objects in each input. |
| `--recursive` | Expand directory inputs into recursive file sweeps. |
| `--skip-unsupported` | Skip unparsable inputs, inputs with no supported code object, or unsupported analysis failures. |
| `--max-diagnostics N` | Limit collected and printed diagnostics. Use `0` to suppress diagnostic payloads while preserving counts. |
| `--stop-after-first-diagnostic` | Stop each code object after the first observed hazard. Useful for large sweeps. |
| `--summary-only` | Print only final batch totals. |
| `--no-fail` | Return success even when hazards are reported. Useful for measurement runs. |

Exit codes:

- `0`: analysis succeeded and no hazards were found, or `--no-fail` was set.
- `1`: command-line usage error.
- `2`: input selection, parsing, or analysis error.
- `4`: one or more hazards were found.

## gfx950 Tensile E2E

The optional `rj_waitcheck_gfx950_tensile_e2e` target builds a small TensileLite
gfx950 corpus and checks the final loadable `Kernels.so-*.hsaco` sidecars:

```sh
ROCM_VENV="$HOME/rocjitsu/gfx1250-dbt/venv" \
TENSILELITE_ROOT="$HOME/rocjitsu/rocjitsu-corpus/results-deps/upstream-rocm-libraries/projects/hipblaslt/tensilelite" \
PYTHON="$PWD/build/waitcheck-e2e/tensile-gfx950/.venv/bin/python" \
cmake --build build --target rj_waitcheck_gfx950_tensile_e2e
```

Use `ROCM_VENV` for a TheRock SDK venv, `ROCM_PATH` for a normal ROCm tree, or a
`rocm-sdk` executable on `PATH`. The selected Python must have TensileLite's
Python dependencies and `rocisa` installed.

By default the target builds one GEMM config and one sparse GEMM config. Override
the list with colon-separated paths relative to the TensileLite root:

```sh
WAITCHECK_TENSILE_CONFIGS="Tensile/Tests/common/gemm/gfx950/bf16_cvt.yaml:Tensile/Tests/common/gradient/gfx950/bf16_gradient_bias.yaml" \
cmake --build build --target rj_waitcheck_gfx950_tensile_e2e
```

This target intentionally skips Tensile intermediate `.o` files and
`TensileLibrary_gfx950.co` containers. Those artifacts need separate triage
because they are not the final sidecar HSACOs loaded as individual kernels.

## Runtime Preload

Preload the shim into a process that uses ROCR/HSA:

```sh
LD_PRELOAD="$PWD/build/lib/rocjitsu/src/rocjitsu/kmd/linux/librocjitsu_waitcheck.so" \
  ROCJITSU_WAITCHECK_FAIL=1 \
  ./app
```

Environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `ROCJITSU_WAITCHECK` | `1` | Set to `0` to disable checking while leaving the shim preloaded. |
| `ROCJITSU_WAITCHECK_FAIL` | `0` | Set to `1` to reject supported code objects with missing waits by returning `HSA_STATUS_ERROR_INVALID_CODE_OBJECT`. |

The shim prints diagnostics to stderr. With `ROCJITSU_WAITCHECK_FAIL=0`, it
reports hazards but chains to the real runtime reader.

## How It Works

The checker parses `AmdGpuCodeObject` images and analyzes executable kernel
entry points. It decodes instructions, builds basic blocks, runs a forward
dataflow analysis over the object-code CFG, tracks outstanding wait-counter
events, and reports a diagnostic when a later instruction uses, overwrites,
orders after, or ends the program before the relevant event has been waited on
strongly enough.

This is intentionally an ISA-level analysis. For final-code wait hazards, the
encoded instructions plus target features are the source of truth. Compared to
an LLVM MC-level view, missing fidelity is an engineering issue in the RocJITsu
ISA metadata, not a fundamental limitation of analyzing encoded ISA. Metadata
such as explicit operands, implicit operands, instruction classes, wait-counter
effects, embedded wait fields, branch targets, and target predicates can be
generated or added to the ISA layer.

The runtime shim intercepts:

- `hsa_code_object_reader_create_from_memory`
- `hsa_code_object_reader_create_from_file`
- `hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size`

It also patches AMD loader extension tables returned through
`hsa_system_get_extension_table` and `hsa_system_get_major_extension_table`, so
clients that call the offset-size reader through the extension table are checked
too.

The current analyzer models gfx12 and gfx950 object-visible wait behavior including:

- split `loadcnt`, `storecnt`, `dscnt`, `kmcnt`, `samplecnt`, `bvhcnt`, and
  `expcnt` hazards;
- `s_waitcnt` and combined gfx12 wait encodings;
- `s_wait_alu` depctr hazards for `depctr_vm_vsrc`, `depctr_va_sdst`,
  `depctr_va_vcc`, and `depctr_sa_sdst`;
- DSDIR embedded `wait_vm_vsrc` and `wait_va_vdst` fields;
- VINTERP embedded `wait_exp`;
- gfx1250 `s_set_vgpr_msb` high-VGPR bank selection;
- CFG joins, skipped paths, and loop-carried hazards.

Supported targets are `gfx950`, `gfx1200`, `gfx1201`, and `gfx1250`.

## Limitations

Waitcheck is a post-link object-code checker. It does not rewrite code and it
does not try to reconstruct an intermediate compiler representation. Its job is
to validate the final encoded program.

Known boundaries:

- Waitcheck can only reason from facts present in, or derivable from, final
  code-object bytes and target metadata.
- The analyzer is only as complete as RocJITsu's ISA metadata. Missing implicit
  operands, instruction-class flags, wait-counter effects, or target predicates
  can cause false negatives or false positives until the metadata is improved.
- Hazards whose architectural fix is instruction spacing, `s_nop`, or
  `s_delay_alu` are separate from wait-counter validation unless they also
  expose a final wait-like dependency.
- Compiler-specific questions such as whether LLVM preserved, removed, or
  intentionally avoided a redundant wait are not modeled. Correct final waits
  are accepted; missing final waits are reported.
- Targets outside gfx12/RDNA4 and gfx950/CDNA4 are out of scope for this
  prototype.
- Unsupported or undecodable code objects are analysis failures. For corpus
  measurement, use `--skip-unsupported`; for preload enforcement, supported
  analysis failures fail only when `ROCJITSU_WAITCHECK_FAIL=1`.

## Tests

Focused validation:

```sh
ctest --test-dir build -R 'Waitcheck|RjWaitcheck|WaitcheckPreload' --output-on-failure
```

The main coverage lives in:

- `tests/analysis/waitcheck_test.cpp`
- `tests/tools/rj_waitcheck_smoke_test.cpp`
- `tests/tools/waitcheck_preload_smoke_test.cpp`

The parity map records which LLVM waitcnt/hazard lit areas are represented by
object-code fixtures:

```sh
emulation/rocjitsu/docs/waitcheck-llvm-parity.md
```
