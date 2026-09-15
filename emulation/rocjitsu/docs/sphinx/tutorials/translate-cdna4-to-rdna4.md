---
myst:
    html_meta:
        "description": "Tutorial walking through translating a CDNA4 (gfx950) kernel to RDNA4 (gfx1200) using the rj_dbt_translate tool in rocJITsu."
        "keywords": "rocJITsu, ROCm, DBT, dynamic binary translation, CDNA4, RDNA4, gfx950, gfx1200, rj_dbt_translate, code object"
---
# Translate a CDNA4 kernel to RDNA4 using `rj_dbt_translate`

This tutorial walks through using the `rj_dbt_translate` command-line
tool to translate an AMD GPU code object compiled for CDNA4 (gfx950) so
that it can run on RDNA4 (gfx1200). You will learn how to run the tool
in diff mode, interpret the translation actions it reports, and extract
a translated code object.

`rj_dbt_translate` is the primary debugging interface for the rocJITsu
dynamic binary translator (DBT). Use it whenever you need to inspect how
the translator handles individual instructions across ISA boundaries ---
for example, when verifying that a kernel you compiled for one
architecture can be correctly translated to another.

For a broader look at the DBT architecture, see
[Architecture and component layers](../conceptual/architecture.md). For a complete
reference of `rj_dbt_translate` options, see
[rj_dbt_translate command reference](../reference/rj-dbt-translate.md).

## Compile a HIP kernel for gfx950

Start with a HIP kernel compiled for gfx950. The following command
produces a host executable that bundles an embedded CDNA4 code object:

``` bash
hipcc -o vector_add.o vector_add.hip --offload-arch=gfx950
```

You can verify that the executable contains a gfx950 code object by
listing the bundled images:

``` bash
rj_dbt_translate vector_add.o --list-code-objects
```

## Run `rj_dbt_translate` in diff mode

Diff mode is the most informative output for understanding what the
translator does to each instruction. Run it by specifying the input and
output targets along with `--output-mode diff`:

``` bash
rj_dbt_translate vector_add.o \
  --input-target gfx950 \
  --output-target gfx1200 \
  --output-mode diff
```

The report begins with summaries of the source and translated code
objects (section sizes, instruction counts, decode failures), then lists
each instruction translation that changed from the source.

Identity translations --- instructions that pass through unchanged ---
are omitted from the diff output so that the report focuses on the
interesting cases.

## Interpret the translation report

Each entry in the diff output follows this structure:

``` text
source_words: bf8cc07f
source: s_waitcnt vmcnt(63) expcnt(7) lgkmcnt(0)
target_words: bfc900f0 bfc70000
target: s_wait_storecnt_dscnt 240
target: s_wait_kmcnt 0
```

The `source_words` and `target_words` lines show the raw machine-code
dwords. The `source` line is the disassembly of the original gfx950
instruction, and each `target` line is a disassembled instruction in the
translated output. Multiple `target` lines indicate that a single source
instruction lowered to a sequence of target instructions.

### Translation actions

The DBT classifies every source instruction into one of the following
legalization actions before translating it:

| Action | Description |
| --- | --- |
| **Identity** | The instruction encoding is identical on source and target. The instruction word is copied verbatim. Identity translations are hidden in diff mode unless their bytes changed for another reason. |
| **Substitute** | The instruction has the same encoding layout on both architectures but uses a different opcode value. The translator swaps the opcode field while preserving the rest of the encoding. |
| **Lower** | The instruction has a semantic equivalent on the target, but the encoding differs enough that a field-by-field copy is not sufficient. The translator emits a target-native instruction sequence. Waitcnt splitting and barrier translation are common examples. |
| **Expand** | No direct target equivalent exists. The translator emits a software emulation sequence. MFMA matrix instructions translating from CDNA to RDNA fall into this category because RDNA uses WMMA instead. |

### Waitcnt splitting

CDNA4 uses a single `s_waitcnt` instruction that encodes `vmcnt`,
`expcnt`, and `lgkmcnt` counters in one immediate. RDNA4 (GFX12)
replaced `s_waitcnt` with separate per-counter wait instructions such as
`s_wait_storecnt_dscnt`, `s_wait_kmcnt`, `s_wait_loadcnt`, and others.

When the translator encounters an `s_waitcnt` on a CDNA4 source, it
lowers it into the corresponding split-wait sequence for the RDNA4
target. This is one of the most common lowering actions you will see in
the diff output.

### Semantic lowering

Instructions whose encoding format changed between CDNA4 and RDNA4 go
through semantic lowering. For example, CDNA4 memory instructions using
the `ENC_FLAT`, `ENC_MUBUF`, or `ENC_DS` encoding families translate
into RDNA4's `ENC_VFLAT`, `ENC_VBUFFER`, or `ENC_VDS` families. The
translator decodes the source instruction fields into a neutral
representation, then re-encodes them into the target format, remapping
field names and coherency bits as needed.

## Extract the translated code object

Once you are satisfied with the diff output, extract the translated code
object by switching to `code-object` output mode and redirecting to a
file:

``` bash
rj_dbt_translate vector_add.o \
  --input-target gfx950 \
  --output-target gfx1200 \
  --output-mode code-object > vector_add.gfx1200.co
```

The resulting file is a standalone AMDGPU code object for gfx1200. You
can inspect it with `rj_dbt_translate` itself by treating it as a new
input:

``` bash
rj_dbt_translate vector_add.gfx1200.co \
  --input-target gfx1200 \
  --output-target gfx1200 \
  --output-mode disasm
```

This prints the disassembly of the translated code object, letting you
verify that individual instructions were encoded correctly.

## Diagnose translation issues

If the translator encounters instructions it cannot handle, it emits
structured diagnostics to stderr. Two options help when investigating
problems:

| Option | Description |
| --- | --- |
| `--debug-continue-after-failure` | Continues scanning instructions after a recoverable translation failure, so that a single run can surface multiple diagnostics. |
| `--debug-conservative-liveness N` | Forces the VGPR scratch allocator to skip every register below index `N`. This is useful for checking whether a semantic lowering clobbers guest VGPRs when you know the kernel's declared ordinary VGPR count. |

## Next steps

-   Read the full [rj_dbt_translate command reference](../reference/rj-dbt-translate.md) for all supported options and targets.
-   See [Architecture and component layers](../conceptual/architecture.md) for an
    overview of how the DBT fits into the rocJITsu component stack.
-   Try [Inspect and disassemble a code object with the C API](inspect-code-object.md)
    to explore code objects with the rocJITsu C API.
