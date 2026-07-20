# Vega Rocjitsu Gaps

Architecture: Vega

Manual source: `workspace_docs/amdgpu-isa-manuals/vega/README.md`

Rocjitsu source: `emulation/rocjitsu`

## Coverage

| Area | Status | Notes |
| --- | --- | --- |
| Vega VOP3P packed math | Audited for target/source availability | Checked whether rocjitsu has a Vega arch id, decoder target, generated arch directory, config parser entry, or DBT target. |
| Remaining Vega rocjitsu surface | Not started | Full decoder/autogen/runtime fuzzing remains blocked until a Vega target source exists. |

## Gaps

### VEGA-RJ-001: Rocjitsu has no Vega ISA target to decode or emulate the manual-defined VOP3P slice

Manual evidence:

- Vega section 6.7 defines packed math and the VOP3P opcode inventory at
  `vega/README.md:1499` through `:1516`.
- Vega section 12.10 gives detailed VOP3P packed-math and MIX semantics at
  `vega/README.md:4130` through `:4168`.
- Vega section 13.3.6 defines the VOP3P encoding fields at
  `vega/README.md:6954` through `:7024`.

Rocjitsu evidence:

- `lib/rocjitsu/include/rocjitsu/code/rj_code.h:25` through `:55` enumerates
  CDNA, RDNA, gfx1250, and RISC-V architectures but no Vega or Vega7
  architecture id.
- `lib/rocjitsu/src/rocjitsu/isa/decoder.cpp:42` through `:64` dispatches
  decoders for CDNA, RDNA, and gfx1250 only.
- `lib/rocjitsu/src/rocjitsu/config/config_loader.cpp:724` through `:752`
  parses config architecture strings for CDNA, RDNA, gfx1250, and RISC-V only.
- `tools/dbt_translate_main.cpp:42` through `:50` exposes DBT target rows only
  for gfx942, gfx950, gfx1200, and gfx1201.
- The generated ISA source tree under
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu` contains CDNA, RDNA, gfx1250,
  and shared directories, but no Vega/Vega7 directory.

Impact:

Rocjitsu cannot currently decode, disassemble, fuzz, or emulate Vega packed-math
VOP3P instructions from the manual. Per-instruction packed-math behavior cannot
be compared against an llvm oracle until Vega target metadata and generated ISA
support exist.

## No-Gap Notes

- No per-instruction rocjitsu no-gap conclusions were recorded for Vega in this
  slice because rocjitsu has no Vega ISA target surface to execute or decode.
