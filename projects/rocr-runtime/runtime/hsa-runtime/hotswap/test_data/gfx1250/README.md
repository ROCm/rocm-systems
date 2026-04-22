# gfx1250 Test Kernels

## Provenance

The `.hsaco` files in this directory have **two different
provenances** depending on the kernel category:

### Triton AOT kernels (workload coverage)

`vecadd_gfx1250.hsaco`, `matmul_*_gfx1250.hsaco`,
`softmax_gfx1250.hsaco` are generated via **AOT (Ahead-Of-Time)
compilation** using Triton 3.7.0 targeting `gfx1250` (RDNA4), on a
machine **without** a gfx1250 GPU. They are **standard `triton.jit`
kernels**, not the gluon kernels from
`triton/third_party/amd/python/examples/gluon/` (which require TDM,
warp specialization, and full gluon runtime context for
compilation).

To regenerate:

```bash
source triton/.venv/bin/activate
PYTHONPATH=triton/python:triton/third_party/amd/python \
  python3 triton/aot_compile_gfx1250.py
cp triton/gfx1250_kernels/*.hsaco \
  projects/rocr-runtime/runtime/hsa-runtime/hotswap/test_data/gfx1250/
```

### Hand-crafted hipcc fixtures (per-handler regression coverage)

`permlane16_swap_gfx1250.hsaco`, `dpp_quad_perm_gfx1250.hsaco`,
`ds_swizzle_gfx1250.hsaco` are minimal HIP kernels using inline
asm to encode a specific cross-lane primitive. They give the
P4 / P5 / P6 handlers (see the cross-lane rewrite table at
hotswap/docs/wave-size-translation.md §5.3) CI-resident hardware
regression gates — see `tests/gfx1250_gpu_test.cpp::doTestPermlane16Swap
/ doTestDppQuadPerm / doTestDsSwizzle` for the GTest pattern.

Each kernel's `.hip` source is committed alongside its `.hsaco`
(e.g. `permlane16_swap_kernel.hip` next to
`permlane16_swap_gfx1250.hsaco`). The `.hip` file's header comment
documents the exact `hipcc` + `clang-offload-bundler` invocation
used to regenerate the binary.

To regenerate (example for permlane16_swap):

```bash
DD=projects/rocr-runtime/runtime/hsa-runtime/hotswap/test_data/gfx1250
hipcc --offload-arch=gfx1250 --genco \
  "$DD/permlane16_swap_kernel.hip" \
  -o "$DD/permlane16_swap_gfx1250.hsaco.bundle"
/opt/rocm-7.2.1/lib/llvm/bin/clang-offload-bundler --type=o \
  --targets=hipv4-amdgcn-amd-amdhsa--gfx1250 \
  --input="$DD/permlane16_swap_gfx1250.hsaco.bundle" \
  --output="$DD/permlane16_swap_gfx1250.hsaco" --unbundle
rm -f "$DD/permlane16_swap_gfx1250.hsaco.bundle"
```

Substitute `dpp_quad_perm_kernel` or `ds_swizzle_kernel` for the
other two fixtures. Verify with `llvm-objdump -d --mcpu=gfx1250
"$DD/<stem>.hsaco"` that the expected mnemonic
(`v_permlane16_swap_b32` / `v_mov_b32_dpp ... quad_perm:[1,0,3,2]` /
`ds_swizzle_b32 offset:swizzle(SWAP,2)`) appears.

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

| File | Kernel | Source | Type | WMMA | Key instructions | GTest |
|------|--------|--------|------|------|------------------|-------|
| `vecadd_gfx1250.hsaco` | fp16 vector add | Triton | elementwise | No | `global_load_u16`, `v_add_f16`, `global_store_b16` | `Gfx1250Gpu.Vecadd` |
| `matmul_f16_gfx1250.hsaco` | fp16 GEMM 64×64×32, 4 warps | Triton | matmul | **Yes** | `v_wmma_f32_16x16x32_f16`, `ds_load_tr16_b128` | `Gfx1250Gpu.Matmul64x64` |
| `matmul_f16_large_gfx1250.hsaco` | fp16 GEMM 128×128×32, 8 warps | Triton | matmul | **Yes** | `v_wmma_f32_16x16x32_f16`, `v_bitop3_b32`, `s_set_vgpr_msb` | `Gfx1250Gpu.Matmul128x128*` (XFAIL) |
| `softmax_gfx1250.hsaco` | fused row softmax 1024 cols | Triton | reduction | No | `v_pk_add_f32`, `v_permlanex16_b32`, `v_exp_f32` | `Gfx1250Gpu.Softmax` |
| `permlane16_swap_gfx1250.hsaco` | P4 hand-crafted regression (XOR-16 swap) | hipcc inline-asm | cross-lane unit | No | `v_permlane16_swap_b32_e32` | `Gfx1250Gpu.Permlane16Swap` |
| `dpp_quad_perm_gfx1250.hsaco` | P5 hand-crafted regression (XOR-1 quad swap) | hipcc inline-asm | cross-lane unit | No | `v_mov_b32_dpp ... quad_perm:[1,0,3,2]` | `Gfx1250Gpu.DppQuadPerm` |
| `ds_swizzle_gfx1250.hsaco` | P6 hand-crafted regression (XOR-2 BITMASK_PERM swizzle) | hipcc inline-asm | cross-lane unit | No | `ds_swizzle_b32 offset:swizzle(SWAP,2)` | `Gfx1250Gpu.DsSwizzle` |
| `rcp_sqrt_gfx1250.hsaco` | `1.0f / sqrtf(x)` literal-numer div-scale fixture | hipcc | elementwise | No | `v_sqrt_f32`, `v_div_scale_f32 .., 1.0`, `v_div_fixup_f32 .., 1.0` | `Gfx1250Gpu.RcpSqrt` |
| `tdm_smoke_gfx1250.hsaco` | 1-D tensor_load_to_lds copy (64 × u32) | hipcc builtin | TDM memory | No | `tensor_load_to_lds` (`_d4` form), `s_wait_tensorcnt` | `TdmGpu.CrossTargetCorpus` |
| `tdm_load_gfx1250.hsaco`  | Parameterised tensor_load_to_lds (up-to-5D D#) | hipcc builtin | TDM memory | No | `tensor_load_to_lds` (`_d4` form) | `TdmDescriptorCoverage.DispatchDenseContiguous/Load_*` |
| `tdm_store_gfx1250.hsaco` | Parameterised tensor_store_from_lds (up-to-5D D#) | hipcc builtin | TDM memory | No | `tensor_store_from_lds` (`_d4` form) | `TdmDescriptorCoverage.DispatchDenseContiguous/Store_*` |

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
- **Cluster operations**: `cluster_arrive`, `cluster_wait`
- **Warp specialization / warp pipeline**
- **MXFP data types** (e2m1, float8_e4m3 with scale)
- **v_wmma_scale_f32_16x16x128_f8f6f4** (scaled WMMA)
- **Multi-CTA / CGA layouts**
