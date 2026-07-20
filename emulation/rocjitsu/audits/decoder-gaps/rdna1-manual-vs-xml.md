# RDNA1 Manual vs XML Gaps

Architecture: RDNA1

Manual source: `workspace_docs/amdgpu-isa-manuals/rdna1/README.md`

XML source: `shared/machine-readable-isa/isa/amdgpu_isa_rdna1.xml`

## Coverage

| Manual section | Status | Notes |
| --- | --- | --- |
| 6.7 Packed Math | Audited for VOP3P packed-math slice only | Checked packed opcode inventory, VOP3P use, MIX membership, and DPP adjacency. |
| 12.10 VOP3P Instructions | Audited for packed-math slice only | Checked ordinary packed I16/U16/F16 and MIX instruction definitions against XML metadata. |
| 13.3 Vector ALU Formats | Audited for VOP3P field inventory only | Checked `OPSEL_HI2`, `OPSEL_HI`, `CLMP`, and opcode table coverage. |
| Remaining RDNA1 manual sections | Not started | Full chapter-by-chapter audit remains. |

## Gaps

### RDNA1-XML-001: MIX-specific VOP3P selector and modifier overloads are prose-only

Manual evidence:

- RDNA1 section 6.7 says `V_MAD_MIX_*` uses VOP3P but is not packed math at
  `rdna1/README.md:1660` through `:1677`.
- The MIX instruction definitions say source size/location is controlled by
  `OPSEL`, choosing between full FP32, low FP16, and high FP16 inputs, and that
  `NEG_HI` acts as an absolute-value modifier at `rdna1/README.md:4309`
  through `:4311`.

XML evidence:

- The generic VOP3P field description says `NEG_HI` negates the high operation
  at `amdgpu_isa_rdna1.xml:3250`, and says `OP_SEL`/`OP_SEL_HI` choose lower or
  upper 16-bit inputs at `:3270` through `:3290`.
- `V_FMA_MIX_F32`, `V_FMA_MIXLO_F16`, and `V_FMA_MIXHI_F16` are present at
  `amdgpu_isa_rdna1.xml:115975`, `:116239`, and `:116503`, but their entries
  do not encode the MIX-only selector mapping or `NEG_HI` absolute-value
  behavior.

Impact:

The generic XML field descriptions imply the wrong interpretation for MIX. XML
consumers need a hard-coded MIX override or manual-derived metadata.

## No-Gap Notes

- The RDNA1 XML has a distinct `OP_SEL_HI_2` field at
  `amdgpu_isa_rdna1.xml:3290`, matching the VOP3P split high-selector field in
  the manual at `rdna1/README.md:6633` and `:6645`.
- The XML carries packed VOP3P entries for the audited ordinary packed-F16
  forms, including `V_PK_FMA_F16` at `amdgpu_isa_rdna1.xml:113399`.
- `V_PK_FMAC_F16` is not missing from VOP3P: the manual defines it as a VOP2
  form at `rdna1/README.md:3703`, and the XML lists it under `ENC_VOP2` and
  VOP2 DPP encodings at `amdgpu_isa_rdna1.xml:87071`.
- RDNA1 does not have the later RDNA3/RDNA3.5 packed inline-constant subsection
  in the audited manual. I did not carry the gfx11 F16 inline-constant finding
  into RDNA1 without a manual or oracle source.
- The RDNA1 manual says DPP is compatible only with VOP1 and VOP2 at
  `rdna1/README.md:1683` through `:1695`; the audited ordinary VOP3P packed
  entries do not expose VOP3P DPP encodings.
