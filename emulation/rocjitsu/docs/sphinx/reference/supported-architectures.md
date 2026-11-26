---
myst:
    html_meta:
        "description": "Supported GPU architectures for rocjitsu, including GFX target IDs, ISA families, and concrete target selection."
        "keywords": "rocjitsu, ROCm, GPU architecture, CDNA, RDNA, RISC-V, GFX target, ISA, simulation, DBT, DBI"
---

# Supported GPU architectures

rocjitsu supports simulation, dynamic binary translation (DBT), and dynamic binary instrumentation (DBI) across multiple AMD GPU architecture generations and a RISC-V target. The tables below list the supported architecture groups and concrete CDNA5 targets together with their ISA family grouping. Individual tools can support different subsets, as noted below and in their command reference.

## Architecture table

| Architecture | GFX target | ISA family |
| --- | --- | --- |
| CDNA1 | gfx908 | GFX9 |
| CDNA2 | gfx90a | GFX9 |
| CDNA3 | gfx942 | GFX9 |
| CDNA4 | gfx950 | GFX9 |
| RDNA1 | GFX10.1 | GFX10 |
| RDNA2 | GFX10.3 | GFX10 |
| RDNA3 | GFX11 | GFX11 |
| RDNA3.5 | GFX11.5 | GFX11 |
| RDNA4 | GFX12 | GFX12 |
| CDNA5 | gfx1250, gfx1251 | GFX12.5 |
| RV32I | --- | RV |
| RV64I | --- | RV |

## Target selection

The public code API uses `rj_code_target_id_t` for decoder creation,
code-object filtering, instruction and basic-block lists, and the guest and
host of binary translation. Select the concrete target, such as
`ROCJITSU_CODE_TARGET_GFX1251`, to preserve its instruction legality and
semantic capabilities. Family names such as CDNA5 describe shared internal
implementations; they are not a second public enum.

`rj_code_decoder_create_for_target()` also accepts canonical target strings
and provider aliases. Existing family aliases retain their provider's default
selection (for example, `cdna5` selects gfx1250). Prefer concrete names when
inspecting code for a known processor. RV64I has a decoder target binding;
RV32I remains unsupported.

For all target values, see [API reference: code object](/reference/api-code-object.md).

gfx1250 and gfx1251 both have concrete functional-simulator bindings. The
checked-in gfx1251 configuration is a minimal synthetic topology for functional
execution, not a product model or a claim of cycle/timing fidelity. Decoder-only
consumers use a separate model provider whose instructions have no execution
callbacks.

## ISA family groupings

rocjitsu groups architectures into ISA families that share encoding structure and instruction-set characteristics. The DBT legalization tables and cross-ISA analysis use these groupings to determine translation strategies.

### GFX9 (CDNA)

CDNA1, CDNA2, CDNA3, and CDNA4 share the GFX9 encoding family. All four architectures use a Wave64 wavefront, a 9-bit primary encoding field, and the same scalar and vector encoding formats (`ENC_SOP1`, `ENC_VOP3`, `ENC_FLAT`, and others). CDNA3 and CDNA4 add MFMA matrix instructions and AccVGPR support not present in CDNA1.

### GFX10 (RDNA1 and RDNA2)

RDNA1 and RDNA2 share the GFX10 encoding family with Wave32 as the default wavefront size and support for Wave64.

### GFX11 (RDNA3 and RDNA3.5)

RDNA3 and RDNA3.5 share the GFX11 encoding family. This generation introduces WMMA matrix instructions, VOPD dual-issue encodings, and a restructured `S_WAITCNT` layout.

### GFX12 and GFX12.5 (RDNA4 and CDNA5)

RDNA4 uses the GFX12 ISA family. CDNA5 targets gfx1250 and gfx1251 use
GFX12.5. Their memory encodings are structured as independent `ENC_VFLAT`,
`ENC_VGLOBAL`, `ENC_VSCRATCH`, `ENC_VBUFFER`, and `ENC_VDS` formats, and
split `S_WAIT_*` instructions replace `S_WAITCNT`.

### RV (RISC-V)

The RV32I and RV64I architectures represent 32-bit and 64-bit RISC-V integer base ISAs. RV64I is available for simulation; RV32I has no registered implementation.
