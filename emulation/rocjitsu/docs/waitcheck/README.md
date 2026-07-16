# Waitcheck

Waitcheck is an object-code checker for AMDGPU wait hazards. It treats the final
encoded instruction stream as the contract: if a kernel generator emits ISA
directly, there is no LLVM MIR state to preserve or consult. Waitcheck inspects
final HSA code objects and reports missing or too-weak waits in the program that
the hardware will execute.

It is intended for two workflows:

- `rj_waitcheck`, an offline CLI for code objects, HIP fat binaries, and
  recursive corpus sweeps.
- `librocjitsu_waitcheck_hooks.so`, a ROCR HSA tools library that checks the
  final code-object reader passed to the runtime loader.

The LLVM parity map is used as a regression checklist for known wait patterns,
not as a requirement that kernels came from LLVM. See
[`../waitcheck-llvm-parity.md`](../waitcheck-llvm-parity.md).

## Build

From the RocJITsu workspace:

```sh
cmake --build build --target rj_waitcheck rocjitsu_waitcheck_hooks
```

The expected build products are:

- `build/tools/rj_waitcheck`
- `build/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_waitcheck_hooks.so`

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
build/tools/rj_waitcheck path/to/corpus \
  --recursive --all-code-objects --skip-unsupported --no-fail \
  --max-diagnostics 0 --stop-after-first-diagnostic --summary-only
```

Require a complete target-specific sweep, suitable for an installed PyTorch
tree or another release corpus:

```sh
build/tools/rj_waitcheck /path/to/site-packages/torch \
  --exhaustive --target gfx950 --summary-only -j16
```

`--exhaustive` implies `--recursive --all-code-objects`. Files that do not
contain the selected target are ignored, while a selected code object that
cannot be decoded or fully analyzed is counted as an analysis error and makes
the command fail. The final summary reports completed/discovered code objects
and kernel descriptors as `code-objects=C/D kernels=C/D`; a complete sweep has
matching counts and `analysis-errors=0`. Unlike a bounded measurement sweep,
exhaustive mode rejects `--skip-unsupported` and
`--stop-after-first-diagnostic` because either option could hide unchecked
kernels. When standard error is an interactive terminal, exhaustive mode first
counts the selected code objects and kernel descriptors and then displays
progress as `kernels C/D code-objects C/D`. Redirected and procedural runs are
silent by default; use `--progress` to force the display or `--no-progress` to
disable it. Code objects are independent analysis units, so `-j N` runs up to
`N` of them concurrently. The default is `-j1`, and the maximum is `-j16` to
bound memory use.

Useful options:

| Option | Meaning |
| --- | --- |
| `--target gfx942|gfx950|gfx1100|gfx1200|gfx1201|gfx1250` | Select one supported target from an executable input. |
| `--code-object-index N` | Select the Nth code object for the selected target. |
| `--kernel-entry OFFSET` | Analyze only the descriptor whose `.text` entry byte offset matches `OFFSET`. Use this to mirror dispatch-lazy runtime checking on a massive code object. |
| `--all-code-objects` | Analyze all supported code objects in each input. |
| `--recursive` | Expand directory inputs into recursive file sweeps. |
| `--exhaustive` | Strict target-specific recursive sweep with code-object and kernel completeness totals. Requires `--target`. |
| `--progress` | Show exhaustive kernel progress even when standard error is not an interactive terminal. |
| `--no-progress` | Disable exhaustive kernel progress, including on an interactive terminal. |
| `-j N`, `--jobs N` | Analyze up to N code objects concurrently. The default is 1 and the maximum is 16. |
| `--skip-unsupported` | Skip unparsable inputs, inputs with no supported code object, or unsupported analysis failures. |
| `--max-diagnostics N` | Limit collected and printed diagnostics. Use `0` to suppress diagnostic payloads while preserving counts. |
| `--stop-after-first-diagnostic` | Stop each code object after the first observed hazard. Useful for large sweeps. |
| `--summary-only` | Print only final batch totals. |
| `--no-fail` | Return success even when hazards are reported. Useful for measurement runs. |

Exit codes:

- `0`: analysis succeeded and no hazards were found, or `--no-fail` was set.
- `1`: command-line usage error.
- `2`: input selection, parsing, or analysis error, including an incomplete
  `--exhaustive` sweep.
- `4`: one or more hazards were found.

## Generated Or Standalone Kernel Code

Waitcheck is a static checker. You do not need a matching GPU to analyze a
kernel; you only need the final AMDGPU code object, fat binary, or executable
that contains the code object. This makes it useful for code produced by kernel
generators, handwritten assembly, reduced reproducers, and compiler test cases.

If your build already leaves a loadable `.hsaco`, `.co`, HIP fat binary, or host
executable, run waitcheck on that artifact directly:

```sh
rj_waitcheck generated-kernels.hsaco --target gfx950 --max-diagnostics 64
```

If the artifact contains several AMDGPU images, list them first and select the
one you want:

```sh
rj_waitcheck generated-app --list-code-objects
rj_waitcheck generated-app --target gfx950 --code-object-index 0
```

If your generator emits AMDGPU assembly, assemble and link it into a code object
first. The exact command depends on the assembly format, but a typical flow is:

```sh
llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx950 \
  -filetype=obj generated-kernel.s -o generated-kernel.o
ld.lld -shared generated-kernel.o -o generated-kernel.hsaco
rj_waitcheck generated-kernel.hsaco --target gfx950 --max-diagnostics 64
```

The assembly must contain the normal AMDHSA kernel descriptor and metadata needed
to create a loadable code object. If your source is HIP or another high-level
language, use the compiler flags your build normally uses to keep the generated
code object or saved device binary, then pass that artifact to `rj_waitcheck`.

For generated-code triage, `rj_co` is often useful next to `rj_waitcheck`:

```sh
rj_co generated-kernel.hsaco --target gfx950 --list-kernels
rj_co generated-kernel.hsaco --target gfx950 \
  --disassemble-window .text+0xd1b44 --context-bytes 384
rj_co generated-kernel.hsaco --target gfx950 \
  --waitcheck --repro-diagnostic 0 --max-diagnostics 64
```

The diagnostic location is a section-relative offset such as `.text+0xd1b44`.
Use `--disassemble-window` to inspect the producer, the consumer, and any waits
between them. Use `--repro-diagnostic N` when you need a small markdown block
with the command, diagnostic, and nearby ISA for a bug report or review.

A typical fix/verify loop is:

1. Generate or compile the code object.
2. Run `rj_waitcheck` on the final artifact.
3. Inspect each reported producer/consumer pair with `rj_co`.
4. Add or strengthen the relevant `s_waitcnt`, `s_wait_*`, or embedded wait in
   the generated ISA.
5. Regenerate the code object and rerun `rj_waitcheck`.

For corpus-sized generated output, use a bounded first pass:

```sh
rj_waitcheck path/to/generated/artifacts \
  --recursive --all-code-objects --skip-unsupported --no-fail \
  --stop-after-first-diagnostic --summary-only
```

Then rerun individual artifacts without `--summary-only` when you need the full
diagnostic details. Waitcheck rejects relocatable ELF objects because they are
compiler intermediates rather than final loadable code objects. With
`--skip-unsupported`, recursive corpus sweeps skip those intermediates and keep
scanning loadable `.co` and `.hsaco` files.

## gfx950 Tensile E2E

The optional `rj_waitcheck_gfx950_tensile_e2e` target builds a small TensileLite
gfx950 corpus and checks both final loadable artifacts: the
`TensileLibrary_gfx950.co` library and its `Kernels.so-*.hsaco` sidecars:

```sh
ROCM_VENV=/path/to/therock/venv \
TENSILELITE_ROOT=/path/to/TensileLite \
PYTHON=/path/to/python \
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

This target intentionally skips Tensile intermediate `.o` files. The library
`.co` is a loadable ELF code object (and is named by Tensile's generated client
configuration), so waitcheck analyzes it rather than treating it as a container.
Waitcheck analyzes each kernel entry point independently so large generated
libraries do not retain whole-library CFG state in memory at once.

For a library with many descriptors, use `rj_co --list-kernels` to obtain one
entry offset and reproduce the runtime-sized analysis directly:

```sh
rj_waitcheck libtorch_hip.so --target gfx950 --code-object-index 76 \
  --kernel-entry 0x25d400
```

## Runtime HSA Tool

Load the checker through ROCR's HSA tools interface:

```sh
HSA_TOOLS_DISABLE_REGISTER=1 \
HSA_TOOLS_LIB="$PWD/build/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_waitcheck_hooks.so" \
  ROCJITSU_WAITCHECK_FAIL=1 \
  ./app
```

`HSA_TOOLS_DISABLE_REGISTER=1` selects the environment-driven HSA tools path on
ROCR builds that also support rocprofiler registration. Without it, a successful
registration path can take precedence over `HSA_TOOLS_LIB`.

ROCR reads `HSA_TOOLS_LIB` as a space-separated list and installs tools from
left to right. A later tool is the outer layer. To check the final code produced
by the RocJITsu DBT tool, put waitcheck first and DBT last:

```sh
HSA_TOOLS_DISABLE_REGISTER=1 \
HSA_TOOLS_LIB="$PWD/build/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_waitcheck_hooks.so $PWD/build/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_hooks.so" \
  ROCJITSU_WAITCHECK_FAIL=1 \
  ./app
```

The `rocjitsu` launcher preserves an existing space-separated tool list and
appends `librocjitsu_hooks.so`, so the usual DBT invocation is:

```sh
HSA_TOOLS_LIB="$PWD/build/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_waitcheck_hooks.so" \
  ROCJITSU_WAITCHECK_FAIL=1 \
  build/tools/rocjitsu/rocjitsu --config configs/guest_gfx950_on_gfx1201.json -- ./app
```

Environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `ROCJITSU_WAITCHECK` | `1` | Set to `0` to disable checking while leaving the HSA tool loaded. |
| `ROCJITSU_WAITCHECK_MODE` | `dispatch` | `dispatch` checks a kernel immediately before its first AQL dispatch and caches the result. `eager` exhaustively checks every kernel when its code object is loaded. |
| `ROCJITSU_WAITCHECK_FAIL` | `0` | Set to `1` to stop on a missing wait. Dispatch mode aborts before submitting the bad packet; eager mode rejects the load with `HSA_STATUS_ERROR_INVALID_CODE_OBJECT`. |
| `ROCJITSU_WAITCHECK_SUMMARY` | `0` | Set to `1` to print load/dispatch/check/cache/pass/hazard counters at shutdown or process exit. |

The tool prints diagnostics to stderr. With `ROCJITSU_WAITCHECK_FAIL=0`, it
reports each selected kernel's hazards once and submits the packet. Dispatch
mode indexes descriptors at load, maps them to runtime `kernel_object` values
after executable freeze, and does not decode or build a CFG until the kernel is
actually submitted. This keeps large PyTorch and Tensile code objects from
paying whole-library analysis costs. Use `ROCJITSU_WAITCHECK_MODE=eager` for
load-only validation or when the application never creates an interceptable
HSA queue.

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

The runtime tool patches these core API table entries through chain-safe
`OnLoad`/`OnUnload` callbacks:

- `hsa_code_object_reader_create_from_memory`
- `hsa_code_object_reader_create_from_file`
- `hsa_code_object_reader_destroy`
- `hsa_executable_load_agent_code_object`
- `hsa_ven_amd_loader_code_object_reader_create_from_file_with_offset_size`

It also patches AMD loader extension tables returned through
`hsa_system_get_extension_table` and `hsa_system_get_major_extension_table`, so
clients that call the offset-size reader through the extension table are checked
too. Reader creation records the backing bytes; analysis happens at executable
load. This timing lets an inner waitcheck layer inspect the replacement reader
created by an outer DBT or DBI tool rather than checking only the original
input.

The current analyzer models gfx942/CDNA3, gfx950/CDNA4, gfx1100/RDNA3, and
gfx12 object-visible wait behavior including:

- split `loadcnt`, `storecnt`, `dscnt`, `kmcnt`, `samplecnt`, `bvhcnt`, and
  `expcnt` hazards;
- out-of-order RDNA4 scalar-memory completion, which requires `kmcnt(0)` for
  dependencies on a particular SMEM result;
- `s_waitcnt` and combined gfx12 wait encodings;
- `s_wait_alu` depctr hazards for `depctr_vm_vsrc`, `depctr_va_sdst`,
  `depctr_va_vcc`, and `depctr_sa_sdst`;
- DSDIR embedded `wait_vm_vsrc` and `wait_va_vdst` fields;
- VINTERP embedded `wait_exp`;
- gfx1250 `s_set_vgpr_msb` high-VGPR bank selection;
- CFG joins, skipped paths, and loop-carried hazards;
- legacy CDNA3/CDNA4 VMcnt/LGKMcnt/EXPcnt waits and RDNA3's separate VScnt;
- gfx1100 SOPK single-counter wait forms.

Supported targets are `gfx942`, `gfx950`, `gfx1100`, `gfx1200`, `gfx1201`, and
`gfx1250`.

The `RjWaitcheck.LlvmKernel.*` tests compile the same ordinary HIP vector-add
kernel with AMD Clang for gfx942 and gfx1100 and require both resulting code
objects to scan clean. They also compile a deliberately wait-perturbed HIP
kernel for each target and require waitcheck to diagnose the missing
`s_waitcnt lgkmcnt(0)`. This keeps the cross-target validation on real
LLVM-produced code objects rather than synthetic ELF fixtures.

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
- Targets outside gfx942/CDNA3, gfx950/CDNA4, gfx1100/RDNA3, and gfx12/RDNA4
  are out of scope for this prototype.
- Unsupported or undecodable code objects are analysis failures. For corpus
  measurement, use `--skip-unsupported`; for runtime enforcement, supported
  analysis failures fail only when `ROCJITSU_WAITCHECK_FAIL=1`.

## Tests

Focused validation:

```sh
ctest --test-dir build -R 'Waitcheck|RjWaitcheck' --output-on-failure
```

The main coverage lives in:

- `tests/analysis/waitcheck_test.cpp`
- `tests/tools/rj_waitcheck_smoke_test.cpp`
- `tests/tools/waitcheck_hooks_unit_test.cpp`
- `tests/hip_waitcheck_hazard_test.cpp`

The parity map records which LLVM waitcnt/hazard lit areas are represented by
object-code fixtures:

```sh
emulation/rocjitsu/docs/waitcheck-llvm-parity.md
```
