// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip/hip_runtime.h>

#include "client.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define HIP_CALL(call)                                                                             \
    do                                                                                             \
    {                                                                                              \
        hipError_t err = call;                                                                     \
        if(err != hipSuccess)                                                                      \
        {                                                                                          \
            fprintf(stderr, "%s\n", hipGetErrorString(err));                                       \
            abort();                                                                               \
        }                                                                                          \
    } while(0)

__global__ void
long_running_kernel(uint32_t nspin)
{
    for(uint32_t i = 0; i < nspin / 64; i++)
        asm volatile("s_sleep 1");
    if(nspin > 64)
        for(uint32_t i = 0; i < nspin % 64; i++)
            asm volatile("s_sleep 1");
}

__global__ void
short_kernel(int x, int y)
{
    x = x + y;
}

int
main()
{
    HIP_CALL(hipSetDevice(0));

    start();

    hipStream_t stream = {};
    HIP_CALL(hipStreamCreate(&stream));

    constexpr uint32_t nspin = 4 * 10000;
    long_running_kernel<<<dim3(128), dim3(32), 0, stream>>>(nspin);
    HIP_CALL(hipGetLastError());

    // Stop profiling while the kernel is still in flight.
    // queue_controller_sync in stop_context must drain the kernel without deadlocking.
    stop_profiling();

    // Launch post-stop kernels — these must not deadlock on serializer barriers
    for(int i = 0; i < 5; i++)
        hipLaunchKernelGGL(short_kernel, dim3(1), dim3(1), 0, stream, 1, 2);
    HIP_CALL(hipGetLastError());

    HIP_CALL(hipStreamSynchronize(stream));
    HIP_CALL(hipStreamDestroy(stream));

    fprintf(stderr, "Run complete\n");
    return 0;
}
