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
- `--list-code-objects`: list extractable code objects and exit.
- `--help`: print command-line help.

Supported target names are `gfx942`, `gfx950`, `gfx1200`, and `gfx1201`.

## Output

All selected output is written to stdout. Use shell redirection when a file is
needed:

```sh
rj_dbt_translate vector_add.o --input-target gfx950 --output-target gfx1200 \
  --output-mode code-object > vector_add.gfx1200.co
```

Structured translation diagnostics and validation errors are written to stderr.
Error diagnostics make the command fail.

## Diff Mode

`diff` mode is the primary debugging mode. It prints a compact translation
report with enough context to answer the usual DBT questions:

- Did the instruction change, lower, expand, or get copied unchanged?
- Which source words produced the change?
- Which target words did the translator emit?
- Did expanded code stay in `.text`, or move into `.rj_translations`?
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
