# rj_dbt_translate

`rj_dbt_translate` inspects or translates an AMDGPU code object with the DBT
pipeline. It accepts either a standalone AMDGPU code object or a host object
containing bundled AMDGPU code objects.

This tool is mostly meant as a debugging tool for DBT developers and agents.
Start with `--output-mode diff` when investigating translation behavior; it
shows what changed without requiring you to compare full disassemblies by hand.

## Usage

```text
rj_dbt_translate INPUT --input-target TARGET --output-target TARGET [options]
```

Required arguments:

- `INPUT`: input file path. For host objects, the tool extracts an embedded
  AMDGPU code object for the selected input target.
- `--input-target TARGET`: input LLVM machine name, such as `gfx950`.
- `--output-target TARGET`: output LLVM machine name, such as `gfx1200`.

Options:

- `--input-revision REVISION` and `--output-revision REVISION`: silicon
  revisions (`a0` or `b0`). Both are required for `gfx1250`; other targets do
  not accept them.
- `--code-object-index N`: code-object index for executable inputs. Defaults to
  `0`.
- `--output-mode MODE`: output format. `disasm` prints translated
  disassembly, `code-object` writes the translated code-object bytes, and
  `diff` prints a compact source-to-target translation report. Defaults to
  `disasm`.
- `--debug-conservative-liveness N`: leave liveness dataflow unchanged, but make
  VGPR scratch allocation skip every register below `N`. Pass the
  descriptor-declared ordinary VGPR count when checking whether a semantic
  lowering clobbers guest VGPRs.
- `--debug-continue-after-failure`: keep scanning instructions after recoverable
  translation failures so one run can report multiple diagnostics. The output
  code object is still left unchanged when any error diagnostic is emitted.
- `--skip-failed-kernels`: replace kernels that fail translation with
  non-dispatchable `s_endpgm` stubs so diagnostic modes can continue inspecting
  other kernels. Executable code-object output is rejected when any kernel was
  skipped.
- `--verify-idempotence`: translate the first output again with the same policy
  and require the complete ELF bytes to stay unchanged. The command rejects
  requests whose input and output architecture families differ. A second-pass
  failure or byte difference makes the command fail. This option cannot be
  combined with `--skip-failed-kernels` or `--list-code-objects`.
- `--verify-rewrite-discharge`: scan the final gfx1250 B0-to-A0 output with the
  applicability checks owned by its opcode and operand rewrite registry. The
  command fails if any registered rewrite remains actionable. This option
  cannot be combined with `--skip-failed-kernels` or `--list-code-objects`.
- `--list-code-objects`: list extractable code objects and exit.
- `--help`: print command-line help.

Supported target names are `gfx942`, `gfx950`, `gfx1200`, `gfx1201`, `gfx1250`,
and `gfx1251`.

## Output

All selected output is written to stdout. Use shell redirection when a file is
needed:

```sh
rj_dbt_translate vector_add.o --input-target gfx950 --output-target gfx1200 \
  --output-mode code-object > vector_add.gfx1200.co
```

Structured translation diagnostics and validation errors are written to stderr.
Error diagnostics make the command fail.

## Idempotence Verification

Use `--verify-idempotence` to test whether an offline translation is a strict
byte-level fixed point. For example, to test the gfx1250 B0-to-A0 path:

```sh
rj_dbt_translate input.gfx1250.co \
  --input-target gfx1250 --input-revision b0 \
  --output-target gfx1250 --output-revision a0 \
  --verify-idempotence --output-mode code-object > output.gfx1250-a0.co
```

The verifier keeps the first translated ELF in memory, submits it to the same
translation request, and compares the first and second outputs byte-for-byte
before writing stdout. On a mismatch, stderr identifies the first differing
executable code-section location when possible, falling back to the complete
ELF image.

Verification roughly doubles translation work. While comparing the result it
also retains both translated byte buffers and their parsed code-object copies,
so peak memory can approach four copies of a large code object.

This is deliberately stricter than checking whether the second pass applies
another instruction legalization. Relocation, branch layout, symbol sizes, and
other ELF materialization are part of the result, so changes in any of them make
verification fail. A failure does not by itself mean that the first output is
invalid; it means the translation is not a byte-level fixed point. This mode is
intended to expose such instability while developing or auditing the translator.

## Rewrite-discharge Verification

Use `--verify-rewrite-discharge` to check that gfx1250 B0-to-A0 translation
discharged the applicability conditions of its registered opcode and operand
rewrites:

```sh
rj_dbt_translate input.gfx1250.co \
  --input-target gfx1250 --input-revision b0 \
  --output-target gfx1250 --output-revision a0 \
  --verify-rewrite-discharge --output-mode code-object > output.gfx1250-a0.co
```

The verifier decodes the final executable stream after all translation and ELF
materialization have finished. It reconstructs executable entry boundaries and
stream-checks instruction-local applicability predicates without retaining the
complete decoded stream. If a candidate uses predecessor or successor context,
the verifier reconstructs the full CFG and runs every predicate there instead.
It reports any residual site as an error at its final `.text` offset. One profile
registry supplies both opcode-keyed and operand-driven rewrites to lowering and
verification. Every rule in an audited profile must explicitly register either
a residual predicate and its required decode context, or a no-success contract;
a no-success rule that later begins emitting output fails its registry contract
instead of silently escaping verification.

This is deliberately a registry-level guarantee. It does not certify separate
pipeline augmentations that are not selected through the rewrite registry, such
as WMMA completion-wait insertion; those remain covered by their existing
implementation-specific tests.

Translation and final verification require every runtime-loaded executable byte
to reside in one `SHF_ALLOC | SHF_EXECINSTR` section named `.text`. Inputs with
another allocated executable section, or with executable bytes only in a
differently named section, fail closed instead of leaving an unaudited stream in
the output object.

Executable entries use the same preservation contract as translation and ELF
patching: kernel descriptors, relocation-backed function tables, allocated
target-less dynamic zero-addend `R_AMDGPU_ABS32_LO`, `R_AMDGPU_ABS32_HI`,
`R_AMDGPU_ABS64`, and `R_AMDGPU_ABS32` relocations to ordinary text symbols, and
`R_AMDGPU_RELATIVE64` addends into `.text`. The explicit-symbol forms store,
respectively, the low 32 bits, high 32 bits, full 64 bits, or truncated 32 bits
of the relocated symbol value.

ROCr derives dynamic `STT_OBJECT`, `STT_AMDGPU_HSA_KERNEL`, and `STT_FUNC`
addresses from `st_value`; only `STT_FUNC` is treated here as a
relocation-backed executable entry. ROCr instead resolves `STT_NOTYPE` as an
external agent symbol by name, so a text-defined `STT_NOTYPE` relocation is not
accepted as a local code reference. Global or weak `STT_FUNC` symbols without a
supported runtime reference remain symbol-table metadata rather than independent
execution roots.

ROCr skips explicit-target/static relocation sections for supported code objects,
so they retain the translator's broader metadata-preservation policy rather than
the dynamic-loader allowlist above. An allocated, zero-addend explicit-target
`RELA` reference to a text `STT_FUNC` creates an executable entry for every
non-`R_AMDGPU_NONE` relocation type. An explicit-target `STT_NOTYPE` reference
also creates an entry for `R_AMDGPU_ABS64`; other ordinary non-section text
symbols require only symbol remapping. Explicit-target `R_AMDGPU_NONE` records
are ignored.

An `ET_DYN` `SHT_RELA` section is malformed when its nonzero `sh_info` is out of
range or names an `SHT_NULL` section. Translation rejects that section metadata
before inspecting individual relocation records. The later `R_AMDGPU_NONE` and
text-relocation compatibility checks therefore skip malformed sections; the
earlier structural rejection owns the diagnostic.

Relocation sections explicitly targeting non-allocated sections are likewise
metadata and do not create executable entries. If a runtime-referenced source
entry is emitted at multiple output placements, materialization fails closed:
neither a named symbol nor a relative addend can safely select one kernel-local
clone on behalf of every caller.

This is an opt-in offline development audit. The implementation lives beside
translation because it needs the final materialized ELF and decoded CFG, while
`rj_dbt_translate` is the enabling consumer. Runtime translation leaves the
check disabled; when explicitly requested, an unavailable profile or failed
scan is reported as an ordinary translation error.

This check proves only that the current rewrite payloads clear the conditions
recognized by their corresponding applicability checks. It does not establish
that those checks are complete or correct, prove semantic equivalence, or
validate runtime prerequisites. `--verify-idempotence` can be enabled alongside
this option when both the residual-trigger property and byte-level fixed-point
property are desired.

### Generated Artifact Markers

Translated control-flow artifacts use `s_nop` immediates `0x1250` and `0x1251`
to mark canonical long transfers and branch-island pools. These values belong
to DBT's reserved marker range. A marker word alone is never authoritative:
recognition also validates the complete instruction sequence, decoded CFG
adjacency, and artifact-specific bounds before preserving generated code.

Add any future marker through the shared range constants in
`kernel_text_layout.h`, give it a canonical structural recognizer, and add
near-miss tests proving that marker-shaped guest instructions are not accepted
on their own.

## Diff Mode

`diff` mode is the primary debugging mode. It prints a compact translation
report with enough context to answer the usual DBT questions:

- Did the instruction change, lower, expand, or get copied unchanged?
- Which source words produced the change?
- Which target words did the translator emit?
- Did expanded code emit in-place, or into a kernel-local `.text` cave?
- Did source or translated decode validation fail?

Run it with:

```sh
rj_dbt_translate vector_add.o --input-target gfx950 --output-target gfx1200 \
  --output-mode diff
```

The report starts with source and translated code-object summaries, then lists
the shown instruction translations. Identity translations are omitted unless
their words changed or they were emitted through the code cave, so the report
stays focused on places worth inspecting.

Each shown translation uses this order:

```text
source_words: bf8cc07f
source: s_waitcnt vmcnt(63) expcnt(7) lgkmcnt(0)
target_words: bfc900f0 bfc70000
target: s_wait_storecnt_dscnt 240
target: s_wait_kmcnt 0
```

Read `source_words` and `target_words` as the exact machine words, not as a
re-encoding of the printed assembly. This is useful when checking literal
operands, opcode substitution, wait-counter lowering, and instruction-size
changes. Multiple `target:` lines mean one source instruction lowered to a
target instruction sequence.

## Examples

Print translated disassembly:

```sh
rj_dbt_translate vector_add.o --input-target gfx950 --output-target gfx1200
```

Print a compact translation diff:

```sh
rj_dbt_translate vector_add.o --input-target gfx950 --output-target gfx1200 \
  --output-mode diff
```

List bundled code objects in an executable input:

```sh
rj_dbt_translate vector_add.o --list-code-objects
```
