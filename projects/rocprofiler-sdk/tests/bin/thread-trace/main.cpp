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

#include <rocprofiler-sdk-roctx/roctx.h>

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include "hip/hip_runtime.h"

// 304 CUs * 64 threads * 4 simds * 8 slots on MI300
#define DATA_SIZE          (304 * 256 * 8)
#define HIP_API_CALL(CALL) assert((CALL) == hipSuccess)

#define SHM_SIZE  64
#define LOOPCOUNT 15

__global__ void
branching_kernel(float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ c);

__global__ void
looping_lds_kernel(float* __restrict__ a,
                   const float* __restrict__ b,
                   const float* __restrict__ c,
                   size_t   size,
                   size_t   loopcount,
                   uint32_t ttracedata);

class hipMemory
{
public:
    hipMemory(size_t size)
    {
        HIP_API_CALL(hipMalloc(&ptr, size * sizeof(float)));
        HIP_API_CALL(hipMemset(ptr, 0, size * sizeof(float)));
    }
    ~hipMemory()
    {
        if(ptr) HIP_API_CALL(hipFree(ptr));
    }
    float* ptr = nullptr;
};

int
main(int /*argc*/, char** /*argv*/)
{
    hipMemory src1(DATA_SIZE);
    hipMemory src2(DATA_SIZE);
    hipMemory dst(DATA_SIZE);

    HIP_API_CALL(hipDeviceSynchronize());
    roctxProfilerResume(0);

    bool is_triple_buffer = std::getenv("TRIPLEBUFFER") ? atoi(std::getenv("TRIPLEBUFFER")) : false;
    bool start_and_stop   = std::getenv("STARTSTOP") ? atoi(std::getenv("STARTSTOP")) : false;

    int loopcount = LOOPCOUNT;
    if(start_and_stop)
        loopcount = 5000;  // prevent multiple-cmds test taking too long / timeout
    else if(is_triple_buffer)
        loopcount = 30000;

    for(int i = 0; i < loopcount; i++)
    {
        if(start_and_stop && (i % 500) == 0)
        {
            roctxProfilerPause(0);
            roctxProfilerResume(0);
        }

        hipLaunchKernelGGL(
            branching_kernel, dim3(DATA_SIZE / 64), dim3(64), 0, 0, dst.ptr, src1.ptr, src2.ptr);
        HIP_API_CALL(hipGetLastError());

        uint32_t tracedata = is_triple_buffer ? i : 0xDEADBEEF;
        hipLaunchKernelGGL(looping_lds_kernel,
                           dim3(DATA_SIZE / 64),
                           dim3(64),
                           0,
                           0,
                           dst.ptr,
                           src1.ptr,
                           src2.ptr,
                           DATA_SIZE,
                           LOOPCOUNT,
                           tracedata);
        HIP_API_CALL(hipGetLastError());
        HIP_API_CALL(hipDeviceSynchronize());
    }

    // We use EXECUTE_PAUSE to test for running triple buffering after global destructor
    const char* do_pause = std::getenv("EXECUTE_PAUSE");
    if(do_pause == nullptr || atoi(do_pause) == 1) roctxProfilerPause(0);

    return 0;
}
