// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// undefine NDEBUG so asserts are implemented
#ifdef NDEBUG
#    undef NDEBUG
#endif

#include <cstdint>
#include "hip/hip_runtime.h"

#define SHM_SIZE 64

namespace
{
constexpr uint32_t SHADER_ENGINE_ID_SHIFT = 20;
constexpr uint32_t SHADER_DATA_MASK       = (uint32_t{1} << SHADER_ENGINE_ID_SHIFT) - 1;
constexpr uint32_t DEFAULT_SHADER_DATA    = 0xDEADBEEF;

__device__ uint32_t
get_shader_engine_id()
{
#if defined(__GFX12__)
    const auto value = __builtin_amdgcn_is_invocable(__builtin_amdgcn_s_sendmsg_rtn)
                           ? __builtin_amdgcn_s_sendmsg_rtn(RTN_GET_SE_HW_ID)
                           : 0u;
    const auto se_id = (value >> SE_HW_ID_SE_ID_OFFSET) & ((1u << SE_HW_ID_SE_ID_SIZE) - 1);
#    if defined(__gfx1250__)
    constexpr uint32_t XCC_ID_OFFSET = 16;
    constexpr uint32_t XCC_ID_SIZE   = 4;
    constexpr uint32_t SE_PER_XCC    = 2;
    const auto         xcc_id        = (value >> XCC_ID_OFFSET) & ((1u << XCC_ID_SIZE) - 1);
    return xcc_id * SE_PER_XCC + se_id;
#    else
    return se_id;
#    endif
#else
    const auto se_id  = __builtin_amdgcn_is_invocable(__builtin_amdgcn_s_getreg)
                            ? __builtin_amdgcn_s_getreg(
                                 GETREG_IMMED(HW_ID_SE_ID_SIZE - 1, HW_ID_SE_ID_OFFSET, HW_ID))
                            : 0u;

#    if defined(__gfx942__) || defined(__gfx950__)
    const auto xcc_id = __builtin_amdgcn_is_invocable(__builtin_amdgcn_s_getreg)
                            ? __builtin_amdgcn_s_getreg(GETREG_IMMED(
                                  XCC_ID_XCC_ID_SIZE - 1, XCC_ID_XCC_ID_OFFSET, XCC_ID))
                            : 0u;
    return (xcc_id << HW_ID_SE_ID_SIZE) | se_id;
#    else
    return se_id;
#    endif
#endif
}

}  // namespace

__global__ void
looping_lds_kernel(float* __restrict__ a,
                   const float* __restrict__ b,
                   const float* __restrict__ c,
                   size_t   size,
                   size_t   loopcount,
                   uint32_t ttracedata)
{
    __shared__ float interm[SHM_SIZE];

    size_t index = blockDim.x * blockIdx.x + threadIdx.x;

    for(size_t i = index; i < size; i += blockDim.x * gridDim.x)
        interm[threadIdx.x % SHM_SIZE] = b[index] + threadIdx.x;

    for(size_t it = 0; it < loopcount; it++)
    {
        __syncthreads();
        float value = interm[(it + threadIdx.x + SHM_SIZE / 2) % SHM_SIZE];
        __syncthreads();
        interm[threadIdx.x % SHM_SIZE] += value;
    }

    a[index] = interm[threadIdx.x % SHM_SIZE] + c[index];

    if(ttracedata != DEFAULT_SHADER_DATA)
        ttracedata =
            (get_shader_engine_id() << SHADER_ENGINE_ID_SHIFT) | (ttracedata & SHADER_DATA_MASK);

    asm volatile("s_mov_b32 m0, %0" : : "r"(ttracedata));
    asm volatile("s_nop 4");
    asm volatile("s_ttracedata");
}
