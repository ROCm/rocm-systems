# Vega Manual vs XML Gaps

Architecture: Vega

Manual source: `workspace_docs/amdgpu-isa-manuals/vega/README.md`

XML source: no matching checked-in Vega XML was found under
`shared/machine-readable-isa/isa`

## Coverage

| Manual section | Status | Notes |
| --- | --- | --- |
| 6.7 Packed Math | Audited for source availability and VOP3P packed-math slice only | Checked packed opcode inventory and MIX wording. |
| 12.10 VOP3P Instructions | Audited for source availability and packed-math slice only | Checked packed 16-bit arithmetic and MIX instruction definitions. |
| 13.3.6 VOP3P | Audited for source availability and field inventory only | Checked `OPSEL_HI2`, `OPSEL_HI`, `CLMP`, source selector table, and opcode table. |
| Remaining Vega manual sections | Not started | Full chapter-by-chapter audit remains. |

## Gaps

### VEGA-XML-001: The checked-in machine-readable ISA set has no Vega XML source

Manual evidence:

- Vega section 6.7 defines packed math, says it uses VOP3P, lists packed I16,
  U16, F16, and `V_MAD_MIX_F32` opcodes, and says MIX is a single MAD using
  VOP3P at `vega/README.md:1499` through `:1516`.
- Vega section 12.10 gives detailed VOP3P opcode semantics for packed I16/U16,
  packed F16, and MIX at `vega/README.md:4130` through `:4168`.
- Vega section 13.3.6 defines VOP3P fields, source encodings, `OPSEL`,
  `OPSEL_HI2`, `OPSEL_HI`, and `CLMP` at `vega/README.md:6954` through
  `:7024`.

XML evidence:

- The checked-in XML directory contains CDNA1, CDNA2, CDNA3, CDNA4, RDNA1,
  RDNA2, RDNA3, RDNA3.5, RDNA4, and gfx1250 XMLs, but no
  `amdgpu_isa_vega*.xml`, `amdgpu_isa_gfx900.xml`, or equivalent Vega-family
  XML file.

Impact:

For Vega, every manual fact in this audited packed-math slice is unavailable to
XML-only consumers. This is broader than an individual missing field: the
architecture has no checked-in machine-readable source to compare or generate
from in this repository revision.

## No-Gap Notes

- No semantic no-gap conclusions were recorded for Vega XML in this slice
  because there is no checked-in Vega XML source to validate against.
