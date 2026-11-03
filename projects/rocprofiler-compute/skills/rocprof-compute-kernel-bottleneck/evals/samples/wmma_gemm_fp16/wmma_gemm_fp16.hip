// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

// Sustained FP16 WMMA GEMM for gfx1250 metric health profiling.
// Adapted from hipNPIKernels WMMA_gemm fp16_16x16x32 (native16x16x32).

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "wmma_gemm_fp16_config.hpp"

using namespace wmma_gemm_fp16;

using Float16x4  = _Float16 __attribute__((ext_vector_type(4)));
using Float16x8  = _Float16 __attribute__((ext_vector_type(8)));
using Float16x16 = _Float16 __attribute__((ext_vector_type(16)));
using Floatx8    = float __attribute__((ext_vector_type(8)));

__device__ inline uint32_t lane_id() { return static_cast<uint32_t>(threadIdx.x); }

__device__ inline void load_row_major(Float16x16& regs,
                                      uint32_t row,
                                      uint32_t k_start,
                                      uint32_t row_len,
                                      const _Float16* __restrict__ buffer)
{
    const uint32_t low_high = lane_id() / 16;
    const uint32_t row_addr = row + (lane_id() % 16);

    const uint32_t col1 = k_start + low_high * 16;
    const uint32_t col2 = k_start + 4 + low_high * 16;
    const uint32_t col3 = k_start + 8 + low_high * 16;
    const uint32_t col4 = k_start + 12 + low_high * 16;

    auto* p_regs = reinterpret_cast<Float16x4*>(&regs);
    p_regs[0]    = *reinterpret_cast<const Float16x4*>(buffer + row_addr * row_len + col1);
    p_regs[1]    = *reinterpret_cast<const Float16x4*>(buffer + row_addr * row_len + col2);
    p_regs[2]    = *reinterpret_cast<const Float16x4*>(buffer + row_addr * row_len + col3);
    p_regs[3]    = *reinterpret_cast<const Float16x4*>(buffer + row_addr * row_len + col4);
}

__device__ inline void write_output(const Float16x8& frag,
                                    uint32_t wave_row,
                                    uint32_t wave_col,
                                    uint32_t row_len,
                                    _Float16* __restrict__ d)
{
    const uint32_t out_col = (lane_id() >= 16) ? 8 : 0;
    const uint32_t out_row = lane_id() % 16;
    const uint32_t row     = wave_row + out_row;
    const uint32_t col     = wave_col + out_col;
    *reinterpret_cast<Float16x8*>(d + row * row_len + col) = frag;
}

__device__ inline void inner_gemm(uint32_t sub_m_start,
                                  uint32_t sub_n_start,
                                  uint32_t m,
                                  uint32_t n,
                                  uint32_t k,
                                  const _Float16* __restrict__ a,
                                  const _Float16* __restrict__ b,
                                  _Float16* __restrict__ d)
{
#if !__has_builtin(__builtin_amdgcn_wmma_f32_16x16x32_f16)
    (void)sub_m_start;
    (void)sub_n_start;
    (void)m;
    (void)n;
    (void)k;
    (void)a;
    (void)b;
    if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        d[0] = _Float16(0);
    }
    return;
#else
    Floatx8 frag_acc = {};

    const uint32_t major_wave = __builtin_amdgcn_readfirstlane(threadIdx.x / kWaveSize);
    const uint32_t minor_wave = threadIdx.y;

    const uint32_t wave_row = major_wave * kWmmaM + sub_m_start;
    const uint32_t wave_col = minor_wave * kWmmaN + sub_n_start;

    const bool full_block =
        (sub_m_start + kBlockM <= m) && (sub_n_start + kBlockN <= n) && (k >= kBlockK);

    if (!full_block) {
        return;
    }

    const uint32_t k_tiles = k / kBlockK;
    uint32_t k_iter        = 0;

    Float16x16 reg_a = {};
    Float16x16 reg_b = {};

    load_row_major(reg_a, wave_row, 0, k, a);
    load_row_major(reg_b, wave_col, 0, k, b);

    if (k_tiles > 1) {
        do {
            const uint32_t next_k = (k_iter + 1) * kBlockK;

            Float16x16 next_a = {};
            Float16x16 next_b = {};

            load_row_major(next_a, wave_row, next_k, k, a);
            load_row_major(next_b, wave_col, next_k, k, b);

            __builtin_amdgcn_sched_barrier(0);
            frag_acc = __builtin_amdgcn_wmma_f32_16x16x32_f16(
                0, reg_b, 0, reg_a, 0, frag_acc, 0, 0);
            __builtin_amdgcn_sched_barrier(0);

            reg_a = next_a;
            reg_b = next_b;
            ++k_iter;
        } while (k_iter < (k_tiles - 1));
    }

    frag_acc = __builtin_amdgcn_wmma_f32_16x16x32_f16(0, reg_b, 0, reg_a, 0, frag_acc, 0, 0);

    Float16x8 frag_d = {};
    for (int i = 0; i < 8; ++i) {
        frag_d[i] = static_cast<_Float16>(frag_acc[i]);
    }

    write_output(frag_d, wave_row, wave_col, n, d);
#endif
}

__attribute__((amdgpu_flat_work_group_size(1, kFlatWorkgroupSize)))
extern "C" __global__ void gemm_wmma_fp16(uint32_t m,
                                          uint32_t n,
                                          uint32_t k,
                                          const _Float16* __restrict__ a,
                                          const _Float16* __restrict__ b,
                                          _Float16* __restrict__ d)
{
    const uint32_t block_count_m = DivUp(static_cast<int>(m), static_cast<int>(kBlockM));
    const uint32_t block_count_n = DivUp(static_cast<int>(n), static_cast<int>(kBlockN));

    for (uint32_t bm = blockIdx.x; bm < block_count_m; bm += gridDim.x) {
        for (uint32_t bn = blockIdx.y; bn < block_count_n; bn += gridDim.y) {
            const uint32_t block_m = bm * kBlockM;
            const uint32_t block_n = bn * kBlockN;
            if (block_m < m && block_n < n) {
                inner_gemm(block_m, block_n, m, n, k, a, b, d);
            }
        }
    }
}
