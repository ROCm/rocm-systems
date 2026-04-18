# gfx1250 Test Kernels

## Provenance

These `.hsaco` code objects were generated via **AOT (Ahead-Of-Time)
compilation** using Triton 3.7.0 targeting `gfx1250` (RDNA4), on a machine
**without** a gfx1250 GPU.

They are **standard `triton.jit` kernels**, not the gluon kernels from
`triton/third_party/amd/python/examples/gluon/` (which require TDM, warp
specialization, and full gluon runtime context for compilation).

### Generation script

```
triton/aot_compile_gfx1250.py
```

### How to regenerate

```bash
source triton/.venv/bin/activate
PYTHONPATH=triton/python:triton/third_party/amd/python \
  python3 triton/aot_compile_gfx1250.py
cp triton/gfx1250_kernels/*.hsaco \
  projects/rocr-runtime/runtime/hsa-runtime/hotswap/test_data/gfx1250/
```

### Build environment

| Component | Version |
|-----------|---------|
| Triton | 3.7.0 (git 12138f43, triton-lang/triton main) |
| LLVM | 23.0.0git (`~/shared-llvm`, AMDGPU-only build) |
| PyTorch | 2.9.1+rocm6.3 |
| Python | 3.12.12 |
| OS | Ubuntu 22.04.5 LTS |
| Host GPU | MI300X (gfx942) — no gfx1250 present |

## Kernels

| File | Kernel | Type | WMMA | Key instructions |
|------|--------|------|------|------------------|
| `vecadd_gfx1250.hsaco` | fp16 vector add | elementwise | No | `global_load_u16`, `v_add_f16`, `global_store_b16` |
| `matmul_f16_gfx1250.hsaco` | fp16 GEMM 64×64×32, 4 warps | matmul | **Yes** | `v_wmma_f32_16x16x32_f16`, `ds_load_tr16_b128` |
| `matmul_f16_large_gfx1250.hsaco` | fp16 GEMM 128×128×32, 8 warps | matmul | **Yes** | `v_wmma_f32_16x16x32_f16`, `v_bitop3_b32`, `s_set_vgpr_msb` |
| `softmax_gfx1250.hsaco` | fused row softmax 1024 cols | reduction | No | `v_pk_add_f32`, `v_permlanex16_b32`, `v_exp_f32` |
| `permlane16_swap_gfx1250.hsaco` | per-lane swap regression for CROSS_LANE_SURVEY P4 | unit test | No | `v_permlane16_swap_b32_e32` |

### Per-kernel build provenance

The Triton-built `.hsaco` files (vecadd, matmul, softmax) are
regenerated via `triton/aot_compile_gfx1250.py` per the recipe above.

The hipcc-built `.hsaco` (`permlane16_swap_gfx1250.hsaco`) is built
directly from the committed source `permlane16_swap_kernel.hip` —
the `.hip`'s comment block documents the exact `hipcc` +
`clang-offload-bundler` invocation. It exists to give the P4
end-to-end test (`Gfx1250Gpu.Permlane16Swap` in
`tests/gfx1250_gpu_test.cpp`) a reproducible source, so any future
maintainer can re-derive the binary from the .hip without consulting
external tooling state.

## What these exercise (and what they don't)

### Covered gfx1250 features
- `v_wmma_f32_16x16x32_f16` — Wave Matrix Multiply Accumulate
- `ds_load_tr16_b128` — LDS transpose load (used by WMMA)
- `s_barrier_signal` / `s_barrier_wait` — new barrier model
- `v_dual_*` — VOPD dual-issue instructions
- `v_bitop3_b16` / `v_bitop3_b32` — ternary bitwise operations
- `s_wait_dscnt` / `s_wait_loadcnt` / `s_wait_xcnt` — new wait counters
- `v_pk_add_f32` / `v_pk_mul_f32` — packed f32 operations
- `v_permlanex16_b32` — cross-lane permute
- `s_cvt_f32_u32` / `s_cvt_u32_f32` — scalar type conversions
- `s_mul_f32` / `s_mul_u64` — scalar FP and 64-bit multiply
- `s_setreg_imm32_b32` with `HW_REG_WAVE_MODE` — wave32 mode setting

### NOT covered (requires gluon kernels)
- **TDM** (Tensor Data Mover): `async_load`, `prefetch`, `async_wait`
- **Cluster operations**: `cluster_arrive`, `cluster_wait`
- **Warp specialization / warp pipeline**
- **MXFP data types** (e2m1, float8_e4m3 with scale)
- **v_wmma_scale_f32_16x16x128_f8f6f4** (scaled WMMA)
- **Multi-CTA / CGA layouts**
