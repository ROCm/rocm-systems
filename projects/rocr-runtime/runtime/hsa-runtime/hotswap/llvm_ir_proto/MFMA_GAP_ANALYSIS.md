# MFMA Kernel Instruction Gap Analysis

## Source Kernels

### Hand-crafted MFMA GEMM (`mfma_gemm.hip`)
- `mfma_gemm_16x16`: uses `v_mfma_f32_16x16x16_f16`, 64 threads per tile
- `mfma_gemm_32x32`: uses `v_mfma_f32_32x32x8_f16`, 64 threads per tile
- Compiled for gfx942 with `-O2`

## Instruction Frequency (both kernels combined)

| Count | Instruction | Status |
|-------|------------|--------|
| 313 | `s_nop` | **HANDLED** (skip) |
| 49 | `v_mov_b32_e32` | NEW |
| 39 | `v_lshl_add_u64` | **HANDLED** |
| 39 | `v_ashrrev_i32_e32` | **HANDLED** |
| 28 | `v_add_u32_e32` | **HANDLED** |
| 26 | `s_cbranch_execz` | **HANDLED** |
| 24 | `v_cmp_gt_i32_e32` | **HANDLED** |
| 24 | `s_and_saveexec_b64` | **HANDLED** |
| 22 | `s_or_b64` | NEW |
| 20 | `global_store_dword` | **HANDLED** |
| 12 | `s_waitcnt` | **HANDLED** (skip) |
| 11 | `v_or_b32_e32` | NEW |
| 6 | `v_mad_u64_u32` | NEW |
| 6 | `v_cmp_gt_i32_e64` | NEW (e64 variant) |
| 6 | `s_lshl_b32` | NEW |
| 6 | `s_and_b64` | NEW |
| 5 | `s_branch` | NEW |
| 4 | `s_load_dwordx4` | **HANDLED** |
| 4 | `s_cbranch_scc1` | NEW |
| 4 | `global_load_dwordx2` | NEW (multi-dword) |
| 3 | `v_mul_lo_u32` | NEW |
| 3 | `v_and_b32_e32` | NEW |
| 2 | `v_lshrrev_b32_e32` | NEW |
| 2 | `v_add3_u32` | NEW |
| 2 | `s_mov_b64` | NEW |
| 2 | `s_mov_b32` | NEW |
| 2 | `s_load_dwordx2` | **HANDLED** |
| 2 | `s_endpgm` | **HANDLED** |
| 2 | `s_cmp_ge_i32` | NEW |
| 2 | `s_add_i32` | NEW |
| 1 | **`v_mfma_f32_32x32x8_f16`** | **NEW (MFMA)** |
| 1 | **`v_mfma_f32_16x16x16_f16`** | **NEW (MFMA)** |
| 1 | `v_lshlrev_b32_e32` | NEW |
| 1 | `v_lshl_add_u32` | NEW |
| 1 | `s_cmp_lt_i32` | NEW |
| 1 | `s_cmp_gt_i32` | NEW |
| 1 | `s_cbranch_vccnz` | NEW |
| 1 | `s_andn2_b64` | NEW |

## Summary

- **Already handled**: 11 instruction types (s_nop, s_waitcnt, s_load_dwordx2/x4, s_and_saveexec_b64, s_cbranch_execz, v_add_u32, v_cmp_gt_i32_e32, v_ashrrev_i32, v_lshl_add_u64, global_load/store_dword, s_endpgm)
- **New instructions needed**: 27 instruction types

### By category:
- **MFMA** (2): `v_mfma_f32_16x16x16_f16`, `v_mfma_f32_32x32x8_f16`
- **Scalar ALU** (6): `s_add_i32`, `s_mov_b32`, `s_mov_b64`, `s_lshl_b32`, `s_or_b64`, `s_and_b64`, `s_andn2_b64`
- **Scalar compare** (3): `s_cmp_gt_i32`, `s_cmp_lt_i32`, `s_cmp_ge_i32`
- **Branch** (3): `s_branch`, `s_cbranch_scc1`, `s_cbranch_vccnz`
- **Vector ALU** (8): `v_mov_b32`, `v_or_b32`, `v_and_b32`, `v_lshrrev_b32`, `v_lshlrev_b32`, `v_add3_u32`, `v_mul_lo_u32`, `v_mad_u64_u32`, `v_lshl_add_u32`
- **Vector compare e64** (1): `v_cmp_gt_i32_e64`
- **Global memory** (1): `global_load_dwordx2`

### Key observation
This kernel does NOT use LDS or buffer operations because it's a direct global-memory MFMA kernel without tiling through shared memory. This makes it an ideal first target — we can add MFMA support without needing LDS/buffer infrastructure first.
