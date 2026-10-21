---
myst:
    html_meta:
        "description": "Command reference for the rj_dbt_translate tool in rocJITsu, covering all CLI flags, output modes, supported targets, and exit codes."
        "keywords": "rocJITsu, rj_dbt_translate, DBT, dynamic binary translation, CLI, AMDGPU, ROCm, code object, disassembly"
---
# `rj_dbt_translate` command reference

`rj_dbt_translate` inspects or translates an AMDGPU code object using
the rocJITsu dynamic binary translation (DBT) pipeline. It accepts
either a standalone AMDGPU code object or a host object containing
bundled AMDGPU code objects. The tool is primarily intended for
debugging DBT behavior.

For the underlying C API used by the translator, see
[API reference: code object](api-code-object.md). For a
hands-on translation walkthrough, see
[Translate a CDNA4 kernel to RDNA4 using rj_dbt_translate](../tutorials/translate-cdna4-to-rdna4.md).

## Synopsis

``` text
rj_dbt_translate INPUT --input-target TARGET --output-target TARGET [options]
```

## Required arguments

| Argument | Description |
| --- | --- |
| `INPUT` | Input file path. For host objects, the tool extracts an embedded AMDGPU code object for the selected input target. |
| `--input-target TARGET` | Input LLVM machine name (for example, `gfx950`). |
| `--output-target TARGET` | Output LLVM machine name (for example, `gfx1200`). |

## Options

| Option | Description |
| --- | --- |
| `--code-object-index N` | Code-object index for executable inputs that contain multiple device images. Defaults to `0`. |
| `--output-mode MODE` | Output format. Accepted values: `disasm` (print translated disassembly to stdout; this is the default), `code-object` (write the translated code-object bytes to stdout), `diff` (print a compact source-to-target translation report to stdout). |
| `--debug-conservative-liveness N` | Leave liveness dataflow unchanged but make VGPR scratch allocation skip every register below `N`. Pass the descriptor-declared ordinary VGPR count when checking whether a semantic lowering clobbers guest VGPRs. |
| `--debug-continue-after-failure` | Keep scanning instructions after recoverable translation failures so that a single run can report multiple diagnostics. The output code object is left unchanged when any error diagnostic is emitted. |
| `--list-code-objects` | List extractable code objects in the input and exit. |
| `--help` | Print command-line help and exit. |

## Supported target names

The following LLVM machine names are accepted by `--input-target` and
`--output-target`:

-   `gfx942`
-   `gfx950`
-   `gfx1200`
-   `gfx1201`

For a full list of architecture identifiers and their mappings, see
[Supported GPU architectures](supported-architectures.md).

## Output conventions

-   All selected output (disassembly, translated bytes, or diff report)
    is written to **stdout**.
-   Structured translation diagnostics and validation errors are written
    to **stderr**.
-   Error diagnostics cause the command to exit with a non-zero status.

### Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Translation or inspection succeeded without errors. |
| `1` | An error occurred (command-line usage error, input parsing failure, translation error, or validation failure). |

## Shell redirection examples

Write a translated code object to a file:

``` bash
rj_dbt_translate vector_add.o \
  --input-target gfx950 --output-target gfx1200 \
  --output-mode code-object > vector_add.gfx1200.co
```

Capture diagnostics separately from translated output:

``` bash
rj_dbt_translate vector_add.o \
  --input-target gfx950 --output-target gfx1200 \
  --output-mode diff 2> diagnostics.log
```

## Diff mode

`diff` mode is the primary debugging mode. It prints a compact
translation report that answers common DBT questions:

-   Did the instruction change, lower, expand, or get copied unchanged?
-   Which source words produced the change?
-   Which target words did the translator emit?
-   Did expanded code emit in-place or into a kernel-local `.text` cave?
-   Did source or translated decode validation fail?

Identity translations are omitted unless their words changed or they
were emitted through a code cave, keeping the report focused on places
worth inspecting.

Each shown translation uses the following format:

``` text
source_words: bf8cc07f
source: s_waitcnt vmcnt(63) expcnt(7) lgkmcnt(0)
target_words: bfc900f0 bfc70000
target: s_wait_storecnt_dscnt 240
target: s_wait_kmcnt 0
```

The `source_words` and `target_words` lines show exact machine words,
not a re-encoding of the printed assembly. Multiple `target:` lines
indicate that one source instruction lowered to a target instruction
sequence.

Run diff mode with:

``` bash
rj_dbt_translate vector_add.o \
  --input-target gfx950 --output-target gfx1200 \
  --output-mode diff
```

## Usage examples

Print translated disassembly (default output mode):

``` bash
rj_dbt_translate vector_add.o \
  --input-target gfx950 --output-target gfx1200
```

List bundled code objects in an executable input:

``` bash
rj_dbt_translate vector_add.o --list-code-objects
```

Print a compact translation diff:

``` bash
rj_dbt_translate vector_add.o \
  --input-target gfx950 --output-target gfx1200 \
  --output-mode diff
```

## Related pages

-   [API reference: code object](api-code-object.md) --- C API
    for code objects and the `rj_code_translate` function
-   [Supported GPU architectures](supported-architectures.md)
    --- full list of supported GPU architectures
-   [Translate a CDNA4 kernel to RDNA4 using rj_dbt_translate](../tutorials/translate-cdna4-to-rdna4.md)
    --- tutorial for translating a CDNA4 code object to RDNA4
-   [Inspect and disassemble a code object with the C API](../tutorials/inspect-code-object.md) ---
    tutorial for inspecting code objects
-   [rocjitsu CLI reference](rocjitsu-cli.md) ---
    `rocjitsu` main CLI reference
