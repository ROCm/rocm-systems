# RDNA1 Rocjitsu Gaps

Architecture: RDNA1

Manual source: `workspace_docs/amdgpu-isa-manuals/rdna1/README.md`

Rocjitsu source: `emulation/rocjitsu`

## Coverage

| Area | Status | Notes |
| --- | --- | --- |
| RDNA1 VOP3P packed math | Audited statically | Checked generated packed F16/I16/U16 helpers, MIX helpers, selector handling, clamp behavior, and DPP reachability for this slice. |
| Remaining RDNA1 rocjitsu surface | Not started | Full decoder/autogen/runtime fuzzing remains. |

## Gaps

### RDNA1-RJ-001: Ordinary packed I16/U16/F16 VOP3P arithmetic ignores `CLMP`

Manual/XML evidence:

- The RDNA1 VOP3P field table defines `CLMP` as "1 = clamp result" at
  `rdna1/README.md:6628` through `:6636`.
- The XML VOP3P field description also defines generic VOP3P `CLAMP` behavior
  at `amdgpu_isa_rdna1.xml:3220`.

Rocjitsu evidence:

- Ordinary packed-F16 helpers execute without testing `inst.inst_.clamp`, for
  example `V_PK_ADD_F16`, `V_PK_FMA_F16`, `V_PK_MAX_F16`, `V_PK_MIN_F16`, and
  `V_PK_MUL_F16` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:16664`
  through `:16698`, `:16839` through `:16884`, `:17089` through `:17127`,
  `:17192` through `:17230`, and `:17316` through `:17350`.
- Ordinary packed integer helpers wrap or shift 16-bit values without checking
  `inst.inst_.clamp`, for example `V_PK_ADD_I16`, `V_PK_ADD_U16`,
  `V_PK_MAD_I16`, and `V_PK_SUB_I16` at `execute_shared.h:16744` through
  `:16813`, `:17008` through `:17052`, and `:17428` through `:17464`.
- MIX helpers in the same shared file do apply `inst.inst_.clamp` at
  `execute_shared.h:11470` through `:11595`.

Impact:

Packed arithmetic results remain wrapped or unclamped when the VOP3P `CLMP` bit
is set.

### RDNA1-RJ-002: `V_FMA_MIX*` uses multiply-add instead of fused FMA

Manual/XML evidence:

- RDNA1 instruction definitions 32-34 describe `V_FMA_MIX_F32`,
  `V_FMA_MIXLO_F16`, and `V_FMA_MIXHI_F16` as fused multiply-add operations at
  `rdna1/README.md:4309` through `:4311`.
- The XML entries for `V_FMA_MIX_F32`, `V_FMA_MIXLO_F16`, and
  `V_FMA_MIXHI_F16` are present at `amdgpu_isa_rdna1.xml:115975`, `:116239`,
  and `:116503` and describe fused multiply-add behavior.

Rocjitsu evidence:

- `execute_v_fma_mix_f32_vop3p`, `execute_v_fma_mixhi_f16_vop3p`, and
  `execute_v_fma_mixlo_f16_vop3p` compute `float result = a * b + c;` at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:11470`
  through `:11595`.

Impact:

One-rounding FMA cases can differ from hardware and the RDNA1 ISA.

## No-Gap Notes

- `V_PK_FMAC_F16` is generated as VOP2, matching the manual VOP2 definition at
  `rdna1/README.md:3703` and the XML VOP2 entry at
  `amdgpu_isa_rdna1.xml:87071`.
- MIX execution implements the special selector mapping, treats `NEG_HI` as
  absolute value, applies `CLMP`, and preserves the untouched destination half
  for MIXLO/MIXHI at
  `lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu/shared/execute_shared.h:11470`
  through `:11595`.
- The ordinary `V_PK_*` VOP3P constructors do not contain the gfx11 VOP3P DPP
  scaffolding, matching the RDNA1 manual statement that DPP is compatible only
  with VOP1 and VOP2.
- The RDNA1 manual audited here does not contain the RDNA3/RDNA3.5 packed
  inline-constant rule. I did not classify the ordinary packed-F16 inline
  constant path as an RDNA1 bug without manual prose or oracle coverage.
