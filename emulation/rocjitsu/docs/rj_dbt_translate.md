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
- `--list-code-objects`: list extractable code objects and exit.
- `--help`: print command-line help.

Supported target names are `gfx942`, `gfx950`, `gfx1200`, `gfx1201`, and
`gfx1250`.

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
