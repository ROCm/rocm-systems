/*
Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <iostream>
#include <mutex>

#include <hip/hip_runtime.h>

#define ITER_NUM   16 * 1024
#define BLOCK_SIZE 1024

#define HIP_API_CALL(CALL)                                                                         \
    {                                                                                              \
        hipError_t error_ = (CALL);                                                                \
        if(error_ != hipSuccess)                                                                   \
        {                                                                                          \
            auto _hip_api_print_lk = auto_lock_t{print_lock};                                      \
            fprintf(stderr,                                                                        \
                    "%s:%d :: HIP error : %s\n",                                                   \
                    __FILE__,                                                                      \
                    __LINE__,                                                                      \
                    hipGetErrorString(error_));                                                    \
            throw std::runtime_error("hip_api_call");                                              \
        }                                                                                          \
    }

namespace
{
using auto_lock_t = std::unique_lock<std::mutex>;
auto print_lock   = std::mutex{};

void
check_hip_error(void);
}  // namespace

// ======================================================
__global__ void
kernel1(const int c)
{
    int a = 0;
#pragma nounroll
    for(int i = 0; i < ITER_NUM; i++)
    {
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
        asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c));
    }
}

__global__ void
kernel2(const int c)
{
    int a = 0;
#pragma nounroll
    for(int i = 0; i < ITER_NUM; i++)
    {
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
        asm volatile("s_mov_b32 %0 %1\n" : "=s"(a) : "s"(c));
    }
}

__global__ void
kernel3(const float c)
{
    double a        = threadIdx.x;
    float  i        = 0;
    float  d        = threadIdx.x;
    float  e        = 0;
    int    tid_even = threadIdx.x % 2;
    for(int j = 0; j < ITER_NUM; j++)
    {
        if(tid_even == 0)
        {
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
            asm volatile("v_rcp_f64 %0, %0\n" : "+v"(a), "=s"(i) : "s"(c));
        }
        else
        {
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
            asm volatile("v_rcp_f32 %0, %0\n" : "+v"(d), "=s"(e) : "s"(c));
        }
    }
}

// ======================================================

void
run_kernel()
{
    for(int i = 1; i <= 64; i++)
    {
        if(i % 2 == 1)
            kernel1<<<BLOCK_SIZE, i>>>(i);
        else
            kernel2<<<BLOCK_SIZE, i>>>(i);

        check_hip_error();
        HIP_API_CALL(hipDeviceSynchronize());
    }

    float arg = 0;
    kernel3<<<BLOCK_SIZE, 4 * 64>>>(arg);
    check_hip_error();
    HIP_API_CALL(hipDeviceSynchronize());
}

int
main()
{
    run_kernel();
    return 0;
}

namespace
{
void
check_hip_error(void)
{
    hipError_t err = hipGetLastError();
    if(err != hipSuccess)
    {
        auto_lock_t _lk{print_lock};
        std::cerr << "Error: " << hipGetErrorString(err) << std::endl;
        throw std::runtime_error("hip_api_call");
    }
}
}  // namespace
