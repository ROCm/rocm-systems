# Vega7 Manual vs XML Gaps

Architecture: Vega7

Manual source: `workspace_docs/amdgpu-isa-manuals/vega7/README.md`

XML source: no matching checked-in Vega7 XML was found under
`shared/machine-readable-isa/isa`

## Coverage

| Manual section | Status | Notes |
| --- | --- | --- |
| 6.7 Packed Math | Audited for source availability and VOP3P packed-math slice only | Checked packed opcode inventory and MIX wording. |
| 12.10 VOP3P Instructions | Audited for source availability and packed-math slice only | Checked packed 16-bit arithmetic, MIX, and DOT2 instruction definitions. |
| 13.3.6 VOP3P | Audited for source availability and field inventory only | Checked `OPSEL_HI2`, `OPSEL_HI`, `CLMP`, source selector table, and opcode table. |
| Remaining Vega7 manual sections | Not started | Full chapter-by-chapter audit remains. |

## Gaps

### VEGA7-XML-001: The checked-in machine-readable ISA set has no Vega7 XML source

Manual evidence:

- Vega7 section 6.7 defines packed math, says it uses VOP3P, lists packed I16,
  U16, F16, and `V_MAD_MIX_F32` opcodes, and says MIX is a single MAD using
  VOP3P at `vega7/README.md:1513` through `:1530`.
- Vega7 section 12.10 gives detailed VOP3P opcode semantics for packed I16/U16,
  packed F16, MIX, and DOT2 opcodes at `vega7/README.md:4205` through `:4246`.
- Vega7 section 13.3.6 defines VOP3P fields, source encodings, `OPSEL`,
  `OPSEL_HI2`, `OPSEL_HI`, `CLMP`, and the VOP3P opcode table at
  `vega7/README.md:7071` through `:7142`.

XML evidence:

- The checked-in XML directory contains CDNA1, CDNA2, CDNA3, CDNA4, RDNA1,
  RDNA2, RDNA3, RDNA3.5, RDNA4, and gfx1250 XMLs, but no
  `amdgpu_isa_vega7*.xml`, `amdgpu_isa_gfx908.xml`, or equivalent
  Vega7-family XML file.

Impact:

For Vega7, every manual fact in this audited packed-math slice is unavailable to
XML-only consumers. This includes the Vega7 DOT2 VOP3P opcodes, not just the
common packed I16/U16/F16 and MIX set.

## No-Gap Notes

- No semantic no-gap conclusions were recorded for Vega7 XML in this slice
  because there is no checked-in Vega7 XML source to validate against.
