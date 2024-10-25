// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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
kernelA(int* wait_on, int value, int* no_opt)
{
    while(*wait_on != value)
    {
        (*no_opt)++;
    };
    (*wait_on)--;
}

int
main(int, char**)
{
    int ntotdevice = 0;
    HIP_CALL(hipGetDeviceCount(&ntotdevice));
    if(ntotdevice < 2) return 0;

    start();
    int* check_value = nullptr;
    int* no_opt      = nullptr;
    HIP_CALL(hipMallocManaged(&check_value, sizeof(*check_value)));
    HIP_CALL(hipMallocManaged(&no_opt, sizeof(*no_opt)));
    *no_opt      = 0;
    *check_value = 1;
    // Will hang if per-device serialization is not functional
    HIP_CALL(hipSetDevice(0));
    hipLaunchKernelGGL(kernelA, dim3(1), dim3(1), 0, 0, check_value, 0, no_opt);
    HIP_CALL(hipSetDevice(1));
    hipLaunchKernelGGL(kernelA, dim3(1), dim3(1), 0, 0, check_value, 1, no_opt);
    HIP_CALL(hipSetDevice(0));
    HIP_CALL(hipDeviceSynchronize());

    std::cerr << "Run complete\n";
}
