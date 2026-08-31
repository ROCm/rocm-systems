---
myst:
    html_meta:
        "description": "Supported GPU architectures for rocJITsu, including GFX target IDs, ISA families, and architecture enum mappings."
        "keywords": "rocJITsu, ROCm, GPU architecture, CDNA, RDNA, RISC-V, GFX target, ISA, simulation, DBT, DBI"
---

# Supported GPU architectures

rocJITsu supports simulation, dynamic binary translation (DBT), and dynamic binary instrumentation (DBI) across multiple AMD GPU architecture generations and a RISC-V target. The tables below list every supported target together with its public API enum values and ISA family grouping.

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
| gfx1250 | gfx1250 | GFX12 |
| RV32I | --- | RV |
| RV64I | --- | RV |

## Architecture enum

The `rj_code_arch_e` enumeration identifies an ISA architecture throughout the rocJITsu C API. Use these values when creating decoders, specifying translation options, or querying architecture properties. For the full enumeration and the rest of the code object and instruction API, see [API reference: code object](/reference/api-code-object.md).

## Target ID enum

The `rj_code_target_id_t` enumeration identifies a specific GPU target within an architecture. Use these values when filtering code objects inside an executable or creating instruction lists. For the full enumeration, see [API reference: code object](/reference/api-code-object.md).

## ISA family groupings

rocJITsu groups architectures into ISA families that share encoding structure and instruction-set characteristics. The DBT legalization tables and cross-ISA analysis use these groupings to determine translation strategies.

### GFX9 (CDNA)

CDNA1, CDNA2, CDNA3, and CDNA4 share the GFX9 encoding family. All four architectures use a Wave64 wavefront, a 9-bit primary encoding field, and the same scalar and vector encoding formats (`ENC_SOP1`, `ENC_VOP3`, `ENC_FLAT`, and others). CDNA3 and CDNA4 add MFMA matrix instructions and AccVGPR support not present in CDNA1.

### GFX10 (RDNA1 and RDNA2)

RDNA1 and RDNA2 share the GFX10 encoding family with Wave32 as the default wavefront size and support for Wave64.

### GFX11 (RDNA3 and RDNA3.5)

RDNA3 and RDNA3.5 share the GFX11 encoding family. This generation introduces WMMA matrix instructions, VOPD dual-issue encodings, and a restructured `S_WAITCNT` layout.

### GFX12 (RDNA4 and gfx1250)

RDNA4 and gfx1250 share the GFX12 encoding family. Memory encodings are restructured into independent `ENC_VFLAT`, `ENC_VGLOBAL`, `ENC_VSCRATCH`, `ENC_VBUFFER`, and `ENC_VDS` formats. The `S_WAITCNT` instruction is replaced by split `S_WAIT_*` instructions.

### RV (RISC-V)

The RV32I and RV64I architectures represent 32-bit and 64-bit RISC-V integer base ISAs. These targets are available for simulation.
