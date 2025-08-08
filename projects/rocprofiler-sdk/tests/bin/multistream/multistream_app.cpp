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

#include <hip/hip_runtime.h>
#include <vector>
#define HIP_ASSERT(call)                                                                           \
    do                                                                                             \
    {                                                                                              \
        hipError_t err = call;                                                                     \
        if(err != hipSuccess)                                                                      \
        {                                                                                          \
            fprintf(stderr, "%s\n", hipGetErrorString(err));                                       \
            abort();                                                                               \
        }                                                                                          \
    } while(0)

__device__ int counter = 0;
__global__ void
add(int n, float* x, float* y)
{
    if(__hip_atomic_load(&counter, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) != 0)
    {
        abort();
    }
    __hip_atomic_fetch_add(&counter, 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);

    int index  = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < n; i += stride)
        y[i] = x[i] + y[i];
    __hip_atomic_fetch_add(&counter, -1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

void
LaunchMultiStreamKernels()
{
    int    N = 1 << 4;
    float* x = new float[N];
    float* y = new float[N];
    float* d_x;
    float* d_y;
    //   Allocate Unified Memory -- accessible from CPU or GPU
    HIP_ASSERT(hipMallocManaged(&d_x, N * sizeof(float)));
    HIP_ASSERT(hipMallocManaged(&d_y, N * sizeof(float)));

    //   initialize x and y arrays on the host
    for(int i = 0; i < N; i++)
    {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
    std::vector<hipStream_t> hip_streams;
    for(int i = 0; i < 100; i++)
    {
        hipStream_t stream;
        HIP_ASSERT(hipStreamCreate(&stream));
        hip_streams.push_back(stream);
    }
    HIP_ASSERT(hipMemcpy(d_x, x, N * sizeof(float), hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(d_y, y, N * sizeof(float), hipMemcpyHostToDevice));

    // Launch kernel on one or two warps on the GPU
    int blockSize = 64;
    // This Kernel will always be launched with one wave
    int numBlocks = 1;
    for(int i = 0; i < 100; i++)
    {
        for(auto& hip_stream : hip_streams)
        {
            hipLaunchKernelGGL(add, numBlocks, blockSize, 0, hip_stream, N, d_x, d_y);
        }
    }

    // Wait for GPU to finish before accessing on host
    HIP_ASSERT(hipDeviceSynchronize());

    HIP_ASSERT(hipMemcpy(x, d_x, N * sizeof(float), hipMemcpyDeviceToHost));
    HIP_ASSERT(hipMemcpy(y, d_y, N * sizeof(float), hipMemcpyDeviceToHost));

    //   Free memory
    HIP_ASSERT(hipFree(d_x));
    HIP_ASSERT(hipFree(d_y));

    delete[] x;
    delete[] y;

    for(auto& hip_stream : hip_streams)
    {
        HIP_ASSERT(hipStreamDestroy(hip_stream));
    }
}

int
main()
{
    LaunchMultiStreamKernels();
}
