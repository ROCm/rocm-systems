# Corpus skip tests by gfx target

This document records every corpus case in each target's skip-list JSON,
grouped by observed outcome. The JSON files are the source of truth for active
CI skips, and their case sets match this document.

Classification buckets:

- **functional failure**: failed for a reason unrelated to a timeout
- **long-running test**: exceeded the standard timeout (~15s) but passed with
  an extended timeout (~30s)
- **possible hang or freeze**: exceeded the extended timeout (~30s), either
  because execution remained too slow or stopped making progress
- **flaky**: passed normally but failed intermittently

## Pinned revisions

| component | commit |
| --- | --- |
| ROCjitsu | `81ffb57e24c682d15e10d2b90d82bcf4c45a9f67` |
| `rocjitsu-test-corpus` | `ce5da512188dd40de0bc7da298ec11b587d8fdd3` |

## gfx942

### functional failure: 52

- `cts.gfx942.fpsan.fpsan_amdgcn_math_extra_test`
- `cts.gfx942.fpsan.fpsan_atomic_test`
- `cts.gfx942.fpsan.fpsan_hip_device_test`
- `cts.gfx942.fpsan.fpsan_mfma_cdna3_bf16_16x16x4_test`
- `cts.gfx942.fpsan.fpsan_mfma_cdna3_bf16_32x32x4_test`
- `cts.gfx942.fpsan.fpsan_mfma_cdna3_bf16_wide_test`
- `cts.gfx942.fpsan.fpsan_mfma_cdna3_f16_16x16x4_test`
- `cts.gfx942.fpsan.fpsan_mfma_cdna3_f16_32x32x4_test`
- `cts.gfx942.fpsan.fpsan_mfma_cdna3_f16_wide_test`
- `cts.gfx942.fpsan.fpsan_mfma_cdna3_f32_32x32_test`
- `cts.gfx942.fpsan.fpsan_mfma_cdna3_f32_test`
- `cts.gfx942.fpsan.fpsan_mfma_cdna3_fp8_16x16x32_bf8_test`
- `cts.gfx942.fpsan.fpsan_mfma_cdna3_fp8_32x32x16_bf8_test`
- `cts.gfx942.fpsan.fpsan_mfma_cdna3_fp8_32x32x16_fp8_test`
- `cts.gfx942.fpsan.fpsan_mfma_cdna3_fp8_test`
- `cts.gfx942.fpsan.fpsan_mfma_cdna3_xf32_test`
- `cts.gfx942.fpsan.fpsan_smfmac_cdna3_test`
- `hip_matmul_matvec::m256_n1_k1024`
- `hip_streamk_simple::m256_n256_k256`
- `hip_streamk_two_tile::m256_n256_k256`
- `llama.gfx942.backend_ops.MUL_MAT.017115bdd15b`
- `llama.gfx942.backend_ops.MUL_MAT.38225d8678d3`
- `llama.gfx942.backend_ops.MUL_MAT.48f89386e558`
- `llama.gfx942.backend_ops.MUL_MAT.5a436936375c`
- `llama.gfx942.backend_ops.MUL_MAT.65b2027ab982`
- `llama.gfx942.backend_ops.MUL_MAT.9e9ba8458bfc`
- `llama.gfx942.backend_ops.MUL_MAT.9f4865116d4b`
- `llama.gfx942.backend_ops.MUL_MAT.a8125e58a34b`
- `llama.gfx942.backend_ops.MUL_MAT.b3d523cee895`
- `llama.gfx942.backend_ops.MUL_MAT.ce78b3dca900`
- `llama.gfx942.backend_ops.MUL_MAT.fe08e494b3c4`
- `llama.gfx942.backend_ops.MUL_MAT_ID.2c2d40514bc8`
- `llama.gfx942.backend_ops.MUL_MAT_ID.3cdbbdcd11b4`
- `llama.gfx942.backend_ops.MUL_MAT_ID.87069c59f324`
- `llama.gfx942.backend_ops.MUL_MAT_ID.965309d54fb6`
- `llama.gfx942.backend_ops.MUL_MAT_ID.c9c4af69fd12`
- `llama.gfx942.backend_ops.MUL_MAT_VEC_FUSION.64a84ca8e243`
- `llama.gfx942.backend_ops.OPT_STEP_ADAMW.53563ed7ba67`
- `llama.gfx942.backend_ops.OUT_PROD.7eb05239af23`
- `llama.gfx942.backend_ops.ROPE.637c25a9ca18`
- `llama.gfx942.backend_ops.ROPE_SET_ROWS.466fbec03e98`
- `llama.gfx942.backend_ops.ROPE_SET_ROWS.9fab018c060b`
- `llama.gfx942.backend_ops.SOLVE_TRI.4b2665c93c4f`
- `llama.gfx942.backend_ops.SOLVE_TRI.efce0cf0e0d3`
- `llama.gfx942.backend_ops.SSM_SCAN.83b46ddc1f72`
- `llama.gfx942.backend_ops.SSM_SCAN.8ffce5cff50e`
- `llama.gfx942.backend_ops.SSM_SCAN.b63c27fe2feb`
- `llama.gfx942.backend_ops.UPSCALE.f4497e2354a8`
- `llama.gfx942.backend_ops.XIELU.925694f8c916`
- `llama.gfx942.backend_ops.XIELU.dc6e5386ba70`
- `llama_mul_mat_vec_q::default`
- `llama_rms_norm::default`

### flaky: 1

- `iree.gfx942.e2e.pack_i8.static_pack_vnni_lhs_large`

### long-running test: 3

- `llama.gfx942.backend_ops.MUL_MAT_ID.4c6f0c369254`
- `llama.gfx942.backend_ops.MUL_MAT_ID.501e10751a9a`
- `llama.gfx942.backend_ops.MUL_MAT_ID.dbedbadd142a`

### possible hang or freeze: 5

- `llama.gfx942.backend_ops.LIGHTNING_INDEXER.28d7def07117`
- `llama.gfx942.backend_ops.MUL_MAT.462ccb738b08`
- `llama.gfx942.backend_ops.MUL_MAT_ID_FUSION.b3d0befaa714`
- `llama.gfx942.backend_ops.MUL_MAT_ID_FUSION.dfb1ed5ca4e9`
- `llama.gfx942.backend_ops.MUL_MAT_ID_FUSION.f6f7601c1c97`

## gfx950

### functional failure: 25

- `cts.gfx950.fpsan.fpsan_amdgcn_math_extra_test`
- `cts.gfx950.fpsan.fpsan_atomic_test`
- `cts.gfx950.fpsan.fpsan_cvt_scalef32_fp4_gfx950_test`
- `cts.gfx950.fpsan.fpsan_cvt_scalef32_fp6_gfx950_test`
- `cts.gfx950.fpsan.fpsan_cvt_scalef32_sr_gfx950_test`
- `cts.gfx950.fpsan.fpsan_hip_device_test`
- `cts.gfx950.fpsan.fpsan_mfma_gfx950_scaled_test`
- `cts.gfx950.fpsan.fpsan_mfma_gfx950_smfmac_test`
- `cts.gfx950.fpsan.fpsan_mfma_gfx950_test`
- `llama.gfx950.backend_ops.MUL_MAT.5a436936375c`
- `llama.gfx950.backend_ops.MUL_MAT.9f4865116d4b`
- `llama.gfx950.backend_ops.MUL_MAT.b3d523cee895`
- `llama.gfx950.backend_ops.OPT_STEP_ADAMW.53563ed7ba67`
- `llama.gfx950.backend_ops.OUT_PROD.7eb05239af23`
- `llama.gfx950.backend_ops.ROPE.637c25a9ca18`
- `llama.gfx950.backend_ops.ROPE_SET_ROWS.466fbec03e98`
- `llama.gfx950.backend_ops.ROPE_SET_ROWS.9fab018c060b`
- `llama.gfx950.backend_ops.SOLVE_TRI.4b2665c93c4f`
- `llama.gfx950.backend_ops.SOLVE_TRI.efce0cf0e0d3`
- `llama.gfx950.backend_ops.SSM_SCAN.83b46ddc1f72`
- `llama.gfx950.backend_ops.SSM_SCAN.8ffce5cff50e`
- `llama.gfx950.backend_ops.SSM_SCAN.b63c27fe2feb`
- `llama.gfx950.backend_ops.UPSCALE.f4497e2354a8`
- `llama.gfx950.backend_ops.XIELU.925694f8c916`
- `llama.gfx950.backend_ops.XIELU.dc6e5386ba70`

### flaky: 1

- `iree.gfx950.e2e.pack_i8.static_pack_vnni_lhs_large`

### long-running test: 2

- `llama.gfx950.backend_ops.MUL_MAT_ID.501e10751a9a`
- `llama.gfx950.backend_ops.MUL_MAT_ID.dbedbadd142a`

### possible hang or freeze: 8

- `llama.gfx950.backend_ops.LIGHTNING_INDEXER.28d7def07117`
- `llama.gfx950.backend_ops.MUL_MAT.462ccb738b08`
- `llama.gfx950.backend_ops.MUL_MAT_ID.4c6f0c369254`
- `llama.gfx950.backend_ops.MUL_MAT_ID_FUSION.b3d0befaa714`
- `llama.gfx950.backend_ops.MUL_MAT_ID_FUSION.dfb1ed5ca4e9`
- `llama.gfx950.backend_ops.MUL_MAT_ID_FUSION.f6f7601c1c97`
- `rocblas_sgemm::fuzz_iter4`
- `rocblas_sgemm::large_2048x2048`

## gfx1100

### functional failure: 44

- `cts.gfx1100.fpsan.fpsan_amdgcn_math_extra_test`
- `cts.gfx1100.fpsan.fpsan_atomic_test`
- `cts.gfx1100.fpsan.fpsan_classify_w64_test`
- `cts.gfx1100.fpsan.fpsan_hip_device_test`
- `cts.gfx1100.fpsan.fpsan_wave_gfx11_w64_test`
- `cts.gfx1100.fpsan.fpsan_wmma_gfx11_acc16_w64_test`
- `cts.gfx1100.fpsan.fpsan_wmma_gfx11_tied_bf16_w64_test`
- `cts.gfx1100.fpsan.fpsan_wmma_gfx11_tied_w64_test`
- `cts.gfx1100.fpsan.fpsan_wmma_gfx11_w64_test`
- `cts.gfx1100.fpsan.fpsan_xlane_w64_test`
- `llama.gfx1100.backend_ops.FLASH_ATTN_EXT.0839462e9aa3`
- `llama.gfx1100.backend_ops.FLASH_ATTN_EXT.09f30cac3778`
- `llama.gfx1100.backend_ops.FLASH_ATTN_EXT.0ffa0a7d7dc5`
- `llama.gfx1100.backend_ops.FLASH_ATTN_EXT.128bf0f8e11b`
- `llama.gfx1100.backend_ops.FLASH_ATTN_EXT.199f26f43bd8`
- `llama.gfx1100.backend_ops.FLASH_ATTN_EXT.1d03df547de7`
- `llama.gfx1100.backend_ops.FLASH_ATTN_EXT.27a842fb2b89`
- `llama.gfx1100.backend_ops.FLASH_ATTN_EXT.2b8c93acdf59`
- `llama.gfx1100.backend_ops.FLASH_ATTN_EXT.2dce6045f861`
- `llama.gfx1100.backend_ops.FLASH_ATTN_EXT.3011acacd940`
- `llama.gfx1100.backend_ops.FLASH_ATTN_EXT.308462dc53da`
- `llama.gfx1100.backend_ops.FLASH_ATTN_EXT.3cb3400551f5`
- `llama.gfx1100.backend_ops.FLASH_ATTN_EXT.3e3b001724ad`
- `llama.gfx1100.backend_ops.FLASH_ATTN_EXT.3f4c31a640cf`
- `llama.gfx1100.backend_ops.FLASH_ATTN_EXT.413b9f5893c3`
- `llama.gfx1100.backend_ops.FLASH_ATTN_EXT.42d8b5a9a8e6`
- `llama.gfx1100.backend_ops.FLASH_ATTN_EXT.56b8d3aa98d7`
- `llama.gfx1100.backend_ops.FLASH_ATTN_EXT.5e2b94d34560`
- `llama.gfx1100.backend_ops.FLASH_ATTN_EXT.6db37d5b0bbd`
- `llama.gfx1100.backend_ops.FLASH_ATTN_EXT.74edc47aa39a`
- `llama.gfx1100.backend_ops.FLASH_ATTN_EXT.78086d5cc78b`
- `llama.gfx1100.backend_ops.FLASH_ATTN_EXT.7fc0eebb6b21`
- `llama.gfx1100.backend_ops.FLASH_ATTN_EXT.8722d0937d99`
- `llama.gfx1100.backend_ops.FLASH_ATTN_EXT.926dbedf6cda`
- `llama.gfx1100.backend_ops.FLASH_ATTN_EXT.a12f4ee82c79`
- `llama.gfx1100.backend_ops.FLASH_ATTN_EXT.a845b89cd4f0`
- `llama.gfx1100.backend_ops.FLASH_ATTN_EXT.afceae33b0b7`
- `llama.gfx1100.backend_ops.FLASH_ATTN_EXT.b951a29db5dd`
- `llama.gfx1100.backend_ops.FLASH_ATTN_EXT.c94907bea454`
- `llama.gfx1100.backend_ops.FLASH_ATTN_EXT.da5c646e799e`
- `llama.gfx1100.backend_ops.FLASH_ATTN_EXT.e024795e26c2`
- `llama.gfx1100.backend_ops.FLASH_ATTN_EXT.e397786c8f00`
- `llama.gfx1100.backend_ops.FLASH_ATTN_EXT.e5e7712c889c`
- `llama.gfx1100.backend_ops.FLASH_ATTN_EXT.eb5de146e0b8`

### long-running test: 2

- `cts.gfx1100.fpsan.fpsan_amdgcn_ldexp_test`
- `cts.gfx1100.fpsan.fpsan_wmma_gfx11_tied_bf16_test`

### possible hang or freeze: 8

- `rocblas_sgemm::large_2048x2048`
- `llama.gfx1100.backend_ops.LIGHTNING_INDEXER.28d7def07117`
- `llama.gfx1100.backend_ops.MUL_MAT_ID.4c6f0c369254`
- `llama.gfx1100.backend_ops.MUL_MAT_ID.501e10751a9a`
- `llama.gfx1100.backend_ops.MUL_MAT_ID.dbedbadd142a`
- `llama.gfx1100.backend_ops.MUL_MAT_ID_FUSION.b3d0befaa714`
- `llama.gfx1100.backend_ops.MUL_MAT_ID_FUSION.dfb1ed5ca4e9`
- `llama.gfx1100.backend_ops.MUL_MAT_ID_FUSION.f6f7601c1c97`

## gfx1201

### functional failure: 145

- `cts.gfx1201.fpsan.fpsan_amdgcn_ldexp_test`
- `cts.gfx1201.fpsan.fpsan_amdgcn_math_extra_test`
- `cts.gfx1201.fpsan.fpsan_atomic_test`
- `cts.gfx1201.fpsan.fpsan_classify_gfx12_w64_test`
- `cts.gfx1201.fpsan.fpsan_hip_device_test`
- `cts.gfx1201.fpsan.fpsan_swmmac_gfx12_w64_16_test`
- `cts.gfx1201.fpsan.fpsan_swmmac_gfx12_w64_fp8_bf8_test`
- `cts.gfx1201.fpsan.fpsan_swmmac_gfx12_w64_fp8_bf8bf8_test`
- `cts.gfx1201.fpsan.fpsan_swmmac_gfx12_w64_fp8_mixed_test`
- `cts.gfx1201.fpsan.fpsan_swmmac_gfx12_w64_fp8_test`
- `cts.gfx1201.fpsan.fpsan_swmmac_gfx12_w64_test`
- `cts.gfx1201.fpsan.fpsan_wave_gfx12_w64_test`
- `cts.gfx1201.fpsan.fpsan_wmma_gfx12_acc16_w64_test`
- `cts.gfx1201.fpsan.fpsan_wmma_gfx12_w64_fp8_bf8_test`
- `cts.gfx1201.fpsan.fpsan_wmma_gfx12_w64_fp8_mixed_test`
- `cts.gfx1201.fpsan.fpsan_wmma_gfx12_w64_fp8_test`
- `cts.gfx1201.fpsan.fpsan_wmma_gfx12_w64_test`
- `cts.gfx1201.fpsan.fpsan_xlane_gfx12_w64_test`
- `cts.gfx1201.int_isa.int_isa_rdna4_ballot_w64_test`
- `cts.gfx1201.int_isa.int_isa_rdna4_bit_permute_test`
- `cts.gfx1201.int_isa.int_isa_rdna4_compare_w64_test`
- `cts.gfx1201.int_isa.int_isa_rdna4_ds_permute_w64_test`
- `cts.gfx1201.int_isa.int_isa_rdna4_integer_dot_test`
- `cts.gfx1201.int_isa.int_isa_rdna4_lane_read_w64_test`
- `cts.gfx1201.int_isa.int_isa_rdna4_lerp_test`
- `cts.gfx1201.int_isa.int_isa_rdna4_mbcnt_w64_test`
- `cts.gfx1201.int_isa.int_isa_rdna4_packed_convert_test`
- `cts.gfx1201.int_isa.int_isa_rdna4_packed_sad_test`
- `cts.gfx1201.int_isa.int_isa_rdna4_permlane_w64_test`
- `cts.gfx1201.int_isa.int_isa_rdna4_reduction_w64_test`
- `cts.gfx1201.int_isa.int_isa_rdna4_swmmac_i32_16x16x32_iu4_test`
- `cts.gfx1201.int_isa.int_isa_rdna4_swmmac_i32_16x16x32_iu4_w64_test`
- `cts.gfx1201.int_isa.int_isa_rdna4_swmmac_i32_16x16x32_iu8_w64_test`
- `cts.gfx1201.int_isa.int_isa_rdna4_swmmac_i32_16x16x64_iu4_w64_test`
- `cts.gfx1201.int_isa.int_isa_rdna4_wmma_i32_16x16x16_iu4_w64_test`
- `cts.gfx1201.int_isa.int_isa_rdna4_wmma_i32_16x16x16_iu8_w64_test`
- `cts.gfx1201.int_isa.int_isa_rdna4_wmma_i32_16x16x32_iu4_w64_test`
- `hip_matmul_matvec::m256_n1_k1024`
- `llama.gfx1201.backend_ops.GATED_DELTA_NET.05b63a4322c6`
- `llama.gfx1201.backend_ops.GATED_DELTA_NET.5615f586b1ca`
- `llama.gfx1201.backend_ops.GATED_DELTA_NET.5a41ca1b14ff`
- `llama.gfx1201.backend_ops.GATED_DELTA_NET.6f2073224190`
- `llama.gfx1201.backend_ops.GATED_DELTA_NET.71f1f886bdeb`
- `llama.gfx1201.backend_ops.GATED_DELTA_NET.7bad0bbc2812`
- `llama.gfx1201.backend_ops.GATED_DELTA_NET.7cb9b00606cb`
- `llama.gfx1201.backend_ops.GATED_DELTA_NET.bafb44a169ef`
- `llama.gfx1201.backend_ops.GATED_DELTA_NET.bbcb01f66416`
- `llama.gfx1201.backend_ops.GATED_DELTA_NET.bdcf60346537`
- `llama.gfx1201.backend_ops.GATED_DELTA_NET.d35f9c8ff08d`
- `llama.gfx1201.backend_ops.GATED_DELTA_NET.f029d1f5b7d0`
- `llama.gfx1201.backend_ops.GATED_DELTA_NET.f1ead0d94ad7`
- `llama.gfx1201.backend_ops.MUL_MAT.0bfa3a774759`
- `llama.gfx1201.backend_ops.MUL_MAT.12d6ef2eb102`
- `llama.gfx1201.backend_ops.MUL_MAT.1430f54270e4`
- `llama.gfx1201.backend_ops.MUL_MAT.1c37272d4207`
- `llama.gfx1201.backend_ops.MUL_MAT.21597cf05866`
- `llama.gfx1201.backend_ops.MUL_MAT.2740bcfad2c3`
- `llama.gfx1201.backend_ops.MUL_MAT.2bb9de5014b7`
- `llama.gfx1201.backend_ops.MUL_MAT.2f847bd0f72c`
- `llama.gfx1201.backend_ops.MUL_MAT.3b9e72804d6c`
- `llama.gfx1201.backend_ops.MUL_MAT.3e329e5b475c`
- `llama.gfx1201.backend_ops.MUL_MAT.3f2054def40b`
- `llama.gfx1201.backend_ops.MUL_MAT.427bd256bceb`
- `llama.gfx1201.backend_ops.MUL_MAT.4292d4689b42`
- `llama.gfx1201.backend_ops.MUL_MAT.4468e13bf5f3`
- `llama.gfx1201.backend_ops.MUL_MAT.4529cc72e450`
- `llama.gfx1201.backend_ops.MUL_MAT.462ccb738b08`
- `llama.gfx1201.backend_ops.MUL_MAT.4791e7191823`
- `llama.gfx1201.backend_ops.MUL_MAT.56a3ebecef08`
- `llama.gfx1201.backend_ops.MUL_MAT.5a2e0296c82c`
- `llama.gfx1201.backend_ops.MUL_MAT.67a24d655c21`
- `llama.gfx1201.backend_ops.MUL_MAT.67c2b413a2ab`
- `llama.gfx1201.backend_ops.MUL_MAT.68fbfd9c3ec5`
- `llama.gfx1201.backend_ops.MUL_MAT.720ff672c8f0`
- `llama.gfx1201.backend_ops.MUL_MAT.757a865e7da5`
- `llama.gfx1201.backend_ops.MUL_MAT.7bcc11c86355`
- `llama.gfx1201.backend_ops.MUL_MAT.7fe4bfa5cc2f`
- `llama.gfx1201.backend_ops.MUL_MAT.965b6be4d678`
- `llama.gfx1201.backend_ops.MUL_MAT.9a612f547630`
- `llama.gfx1201.backend_ops.MUL_MAT.9c792f73463d`
- `llama.gfx1201.backend_ops.MUL_MAT.9f4865116d4b`
- `llama.gfx1201.backend_ops.MUL_MAT.a78a57b1b164`
- `llama.gfx1201.backend_ops.MUL_MAT.adeb0e6159c1`
- `llama.gfx1201.backend_ops.MUL_MAT.af983043c5ee`
- `llama.gfx1201.backend_ops.MUL_MAT.b4ed18f10c28`
- `llama.gfx1201.backend_ops.MUL_MAT.bc1240d0e862`
- `llama.gfx1201.backend_ops.MUL_MAT.be4d79581943`
- `llama.gfx1201.backend_ops.MUL_MAT.c58ade49b398`
- `llama.gfx1201.backend_ops.MUL_MAT.ccf3f0be1c60`
- `llama.gfx1201.backend_ops.MUL_MAT.d5d0d2415b9b`
- `llama.gfx1201.backend_ops.MUL_MAT.d8793c3beaab`
- `llama.gfx1201.backend_ops.MUL_MAT.d8d5f3069b98`
- `llama.gfx1201.backend_ops.MUL_MAT.da65cf942bdb`
- `llama.gfx1201.backend_ops.MUL_MAT.dae71e0e75e6`
- `llama.gfx1201.backend_ops.MUL_MAT.e52b5d823289`
- `llama.gfx1201.backend_ops.MUL_MAT.e5d32efbe803`
- `llama.gfx1201.backend_ops.MUL_MAT.eb1434025ead`
- `llama.gfx1201.backend_ops.MUL_MAT.f678c43ff6f4`
- `llama.gfx1201.backend_ops.MUL_MAT.f8941f74071b`
- `llama.gfx1201.backend_ops.MUL_MAT.faa09334a5c1`
- `llama.gfx1201.backend_ops.MUL_MAT.fbddae989c1d`
- `llama.gfx1201.backend_ops.MUL_MAT_ID.297df3b5f822`
- `llama.gfx1201.backend_ops.MUL_MAT_ID.4d3788c41bef`
- `llama.gfx1201.backend_ops.MUL_MAT_ID.5bba4e9ee5fa`
- `llama.gfx1201.backend_ops.MUL_MAT_ID.6c5a0824e9d0`
- `llama.gfx1201.backend_ops.MUL_MAT_ID.ce7f91ceb634`
- `llama.gfx1201.backend_ops.MUL_MAT_ID.debb5fffb653`
- `llama.gfx1201.backend_ops.MUL_MAT_ID.ef8683b758e2`
- `llama.gfx1201.backend_ops.OUT_PROD.7eb05239af23`
- `llama.gfx1201.backend_ops.OUT_PROD.f35a3e5a1843`
- `llama.gfx1201.backend_ops.RMS_NORM_BACK.738357150297`
- `llama.gfx1201.backend_ops.RMS_NORM_BACK.b0eca1729a88`
- `llama.gfx1201.backend_ops.RMS_NORM_MUL_ROPE.92e31eecab55`
- `llama.gfx1201.backend_ops.RMS_NORM_MUL_ROPE.a61604d8ebe1`
- `llama.gfx1201.backend_ops.RMS_NORM_MUL_ROPE.c9137b08fd08`
- `llama.gfx1201.backend_ops.SOLVE_TRI.4b2665c93c4f`
- `llama.gfx1201.backend_ops.SOLVE_TRI.efce0cf0e0d3`
- `llama.gfx1201.backend_ops.TOPK_MOE.0e270f70f362`
- `llama.gfx1201.backend_ops.TOPK_MOE.2a02b005d62f`
- `llama.gfx1201.backend_ops.TOPK_MOE.2e7ea640edeb`
- `llama.gfx1201.backend_ops.TOPK_MOE.5cb8fc0cd986`
- `llama.gfx1201.backend_ops.TOPK_MOE.6ec8af8002fa`
- `llama.gfx1201.backend_ops.TOPK_MOE.898147839f3e`
- `llama.gfx1201.backend_ops.TOPK_MOE.91b601cce199`
- `llama.gfx1201.backend_ops.TOPK_MOE.ba01fa21e49a`
- `llama.gfx1201.backend_ops.TOPK_MOE.fa7a6e1ee615`
- `llama_mul_mat_vec_q::default`
- `llama_rms_norm::default`
- `rocblas_sgemm::alpha_beta`
- `rocblas_sgemm::beta_zero`
- `rocblas_sgemm::fuzz_iter0`
- `rocblas_sgemm::fuzz_iter1`
- `rocblas_sgemm::fuzz_iter2`
- `rocblas_sgemm::fuzz_iter3`
- `rocblas_sgemm::fuzz_iter4`
- `rocblas_sgemm::large_2048x2048`
- `rocblas_sgemm::rect_16x32x8`
- `rocblas_sgemm::rect_1x64x1`
- `rocblas_sgemm::rect_64x1x64`
- `rocblas_sgemm::rect_7x11x13`
- `rocblas_sgemm::square_16x16`
- `rocblas_sgemm::square_32x32`
- `rocblas_sgemm::square_4x4`
- `rocblas_sgemm::square_8x8`
- `rocblas_sgemm::tiny_2x2x3`

### long-running test: 1

- `llama.gfx1201.backend_ops.MUL_MAT_ID.dbedbadd142a`

### possible hang or freeze: 6

- `llama.gfx1201.backend_ops.LIGHTNING_INDEXER.28d7def07117`
- `llama.gfx1201.backend_ops.MUL_MAT_ID.4c6f0c369254`
- `llama.gfx1201.backend_ops.MUL_MAT_ID.501e10751a9a`
- `llama.gfx1201.backend_ops.MUL_MAT_ID_FUSION.b3d0befaa714`
- `llama.gfx1201.backend_ops.MUL_MAT_ID_FUSION.dfb1ed5ca4e9`
- `llama.gfx1201.backend_ops.MUL_MAT_ID_FUSION.f6f7601c1c97`

## gfx1250

### functional failure: 14

- `cts.gfx1250.fpsan.fpsan_amdgcn_ldexp_test`
- `cts.gfx1250.fpsan.fpsan_amdgcn_math_extra_test`
- `cts.gfx1250.fpsan.fpsan_amdgcn_math_test`
- `cts.gfx1250.fpsan.fpsan_atomic_test`
- `cts.gfx1250.fpsan.fpsan_cvt_scalef32_pk_gfx1250_test`
- `cts.gfx1250.fpsan.fpsan_hip_device_test`
- `cts.gfx1250.fpsan.fpsan_swmmac_16x16x128_fp8_gfx1250_test`
- `cts.gfx1250.fpsan.fpsan_swmmac_16x16x64_16_gfx1250_test`
- `cts.gfx1250.fpsan.fpsan_wmma_16x16x128_f8f6f4_gfx1250_test`
- `cts.gfx1250.fpsan.fpsan_wmma_16x16x128_fp8_gfx1250_test`
- `cts.gfx1250.fpsan.fpsan_wmma_16x16x32_gfx1250_test`
- `cts.gfx1250.fpsan.fpsan_wmma_16x16x64_fp8_gfx1250_test`
- `cts.gfx1250.fpsan.fpsan_wmma_32x16x128_f4_gfx1250_test`
- `cts.gfx1250.fpsan.fpsan_wmma_scale_f8f6f4_gfx1250_test`

### flaky: 1

- `iree.gfx1250.e2e.pack_i8.static_pack_vnni_lhs_large`

### possible hang or freeze: 1

- `rocblas_sgemm::large_2048x2048`
