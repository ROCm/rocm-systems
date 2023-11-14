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
kernelA(int x, int y)
{
    x = x + y;
}

__global__ void
kernelB(int x, int y)
{
    x = x + y;
}

template <typename T>
__global__ void
kernelC(T* C_d, const T* A_d, size_t N)
{
    size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t stride = blockDim.x * gridDim.x;
    for(size_t i = offset; i < N; i += stride)
    {
        C_d[i] = A_d[i] * A_d[i];
    }
}

void
launchKernals()
{
    const int NUM_LAUNCH = 200000;
    // Normal HIP Calls
    int*                             gpuMem;
    [[maybe_unused]] hipDeviceProp_t devProp;
    HIP_CALL(hipGetDeviceProperties(&devProp, 0));
    HIP_CALL(hipMalloc((void**) &gpuMem, 1 * sizeof(int)));

    for(int i = 0; i < NUM_LAUNCH; i++)
    {
        // KernelA and KernelB to be profiled as part of the session
        hipLaunchKernelGGL(kernelA, dim3(1), dim3(1), 0, 0, 1, 2);
        hipLaunchKernelGGL(kernelB, dim3(1), dim3(1), 0, 0, 1, 2);
    }

    const int NElems = 512 * 512;
    const int Nbytes = NElems * 2;
    int *     A_d, *C_d;
    int       A_h[NElems], C_h[NElems];

    for(int i = 0; i < NElems; i++)
    {
        A_h[i] = i;
    }

    HIP_CALL(hipDeviceSynchronize());

    HIP_CALL(hipMalloc(&A_d, Nbytes));
    HIP_CALL(hipMalloc(&C_d, Nbytes));
    HIP_CALL(hipMemcpy(A_d, A_h, Nbytes, hipMemcpyHostToDevice));
    HIP_CALL(hipDeviceSynchronize());
    const unsigned blocks          = 512;
    const unsigned threadsPerBlock = 256;
    for(int i = 0; i < NUM_LAUNCH; i++)
    {
        hipLaunchKernelGGL(kernelC, dim3(blocks), dim3(threadsPerBlock), 0, 0, C_d, A_d, NElems);
    }
    HIP_CALL(hipMemcpy(C_h, C_d, Nbytes, hipMemcpyDeviceToHost));
    HIP_CALL(hipDeviceSynchronize());
    HIP_CALL(hipFree(gpuMem));
    HIP_CALL(hipFree(A_d));
    HIP_CALL(hipFree(C_d));
    std::cerr << "Run complete\n";
}

int
main()
{
    start();
    launchKernals();
}
