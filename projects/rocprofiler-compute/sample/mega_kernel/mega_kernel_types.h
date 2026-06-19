// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
// Shared host/device definitions for the mega_kernel sample. Single definition
// avoids layout drift between main.cpp and mega_kernel.hip.

#pragma once

#include <hip/hip_runtime.h>

// MFMA implementation mode (host passes int to kernel)
enum MfmaMode
{
    MFMA_MODE_BOTH    = 0,
    MFMA_MODE_ASM     = 1,
    MFMA_MODE_BUILTIN = 2
};

struct TestResults
{
    int warp_shuffle_passed;
    int warp_ballot_passed;
    int warp_permute_passed;
    int warp_reduce_passed;

    int fp8_convert_passed;
    int bf8_convert_passed;
    int fp16_convert_passed;
    int bf16_convert_passed;

    int fp32_arith_passed;
    int fp64_arith_passed;
    int int32_arith_passed;
    int int64_arith_passed;

    int pk_f16_passed;
    int pk_bf16_passed;
    int pk_add_passed;

    int atomic_add_f32_passed;
    int atomic_add_f64_passed;
    int atomic_min_f32_passed;
    int atomic_max_f32_passed;
    int atomic_int_passed;
    int lds_atomic_passed;

    int trans_sin_passed;
    int trans_cos_passed;
    int trans_exp_passed;
    int trans_log_passed;
    int trans_sqrt_passed;
    int trans_rcp_passed;
    int trans_rsq_passed;

    int global_load_passed;
    int global_store_passed;
    int lds_load_passed;
    int lds_store_passed;

    int dot4_passed;
    int dot8_passed;

    int mfma_f32_passed;
    int mfma_f64_passed;
    int mfma_f16_passed;
    int mfma_bf16_passed;
    int mfma_i8_passed;
    int mfma_f8_passed;

    int dual_issue_passed;

    int lane_id_passed;
    int wavefront_passed;

    int async_lds_load_passed;
    int async_lds_store_passed;

    int wmma_f16_passed;
    int wmma_bf16_passed;
    int wmma_i8_passed;

    int vmem_flat_passed;
    int vmem_global_passed;
    int vmem_buffer_passed;
    int vmem_scratch_passed;
    int vmem_lds_passed;
    int vmem_tex_load_passed;
    int vmem_tex_store_passed;

    int atomic_global_int_passed;
    int atomic_global_f32_passed;
    int atomic_global_f64_passed;
    int atomic_flat_int_passed;
    int atomic_ds_int_passed;
    int atomic_ds_f32_passed;
    int atomic_ds_f64_passed;
    int atomic_pk_f16_passed;
    int atomic_pk_bf16_passed;
    int atomic_cas_passed;
};

extern "C" __global__ void mega_kernel(
    TestResults* results, float* global_float, double* global_double, int* global_int,
    float* input_buffer, float* output_buffer, float* async_lds_src, float* async_lds_dst,
    int buffer_size, int mfma_mode, hipTextureObject_t tex_obj,
    hipSurfaceObject_t surf_obj);
