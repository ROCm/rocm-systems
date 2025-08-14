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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <thread>

#include "hip/hip_runtime.h"

#define BLOCKDIM 64

/* Macro for checking GPU API return values */
#define HIP_ASSERT(call)                                                                           \
    do                                                                                             \
    {                                                                                              \
        hipError_t gpuErr = call;                                                                  \
        if(hipSuccess != gpuErr)                                                                   \
        {                                                                                          \
            printf(                                                                                \
                "GPU API Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(gpuErr));   \
            exit(1);                                                                               \
        }                                                                                          \
    } while(0)

// HIP kernel. Each thread takes care of one element of input
__global__ void
cube(double* input, double* output, size_t offset, size_t elements_per_stream)
{
    size_t tid     = blockIdx.x * blockDim.x + threadIdx.x;
    size_t gstride = blockDim.x * gridDim.x;

    // Span all elements assigned to this stream
    for(size_t id = tid + offset; id < offset + elements_per_stream; id += gstride)
        for(size_t i = 0; i < 1000; ++i)
            output[id] = input[id] * input[id] * input[id];
}

static void
copy_to_dev()
{
    unsigned int n   = (32 * 1024);  // 32KB
    double*      A_h = nullptr;
    double*      A_d = nullptr;

    HIP_ASSERT(hipHostMalloc(&A_h, n * sizeof(double)));
    HIP_ASSERT(hipMalloc(&A_d, n * sizeof(double)));

    for(unsigned int i = 0; i < n; ++i)
    {
        A_h[i] = 123.5;
    }
    HIP_ASSERT(
        hipMemcpyAsync(A_d, A_h, n * sizeof(double), hipMemcpyHostToDevice, hipStreamPerThread));
    // Repeat to make sure streams remain the same
    HIP_ASSERT(
        hipMemcpyAsync(A_d, A_h, n * sizeof(double), hipMemcpyHostToDevice, hipStreamPerThread));

    // Release device memory
    HIP_ASSERT(hipFree(A_d));
    // Release host memory
    HIP_ASSERT(hipHostFree(A_h));
}

int
main()
{
    // number of streams
    const int num_streams = 8;

    // Number of threads in each thread block
    const int blockSize = 512;

    // Size of vectors
    int n                   = 100000000;
    int elements_per_stream = n / num_streams;
    int bytes_per_stream    = elements_per_stream * sizeof(double);

    // Host input vectors
    double* h_input1{nullptr};
    // Host output vector
    double* h_output1{nullptr};

    // Device input vectors
    double* d_input1{nullptr};
    // Device output vector
    double* d_output1{nullptr};

    // Creating events for timers
    hipEvent_t start{}, stop{};
    HIP_ASSERT(hipEventCreate(&start));
    HIP_ASSERT(hipEventCreate(&stop));

    // Creating streams
    hipStream_t streams[num_streams];
    for(int i = 0; i < num_streams; ++i)
    {
        HIP_ASSERT(hipStreamCreate(&streams[i]));
    }

    // Size, in bytes, of each vector
    size_t bytes = n * sizeof(double);

    // Allocate page locked memory for these vectors on host
    HIP_ASSERT(hipHostMalloc(&h_input1, bytes));
    HIP_ASSERT(hipHostMalloc(&h_output1, bytes));

    // Allocate memory for each vector on GPU
    HIP_ASSERT(hipMalloc(&d_input1, bytes));
    HIP_ASSERT(hipMalloc(&d_output1, bytes));

    // Initialize vectors on host
    for(int i = 0; i < n; i++)
    {
        h_input1[i] = sin(i);
    }

    // Number of thread blocks in grid
    const int gridSizePerStream = 104;  //(int)ceil((float)elements_per_stream/blockSize);

    HIP_ASSERT(hipEventRecord(start));
    // Extra copy with hipStreamPerThread to ensure special case is handled
    HIP_ASSERT(hipMemcpyAsync(
        &d_input1[0], &h_input1[0], bytes_per_stream, hipMemcpyHostToDevice, hipStreamLegacy));
    // split H2D copies and kernel calls into separate loops
    for(int i = 0; i < num_streams; i++)
    {
        int offset = i * elements_per_stream;
        HIP_ASSERT(hipMemcpyAsync(&d_input1[offset],
                                  &h_input1[offset],
                                  bytes_per_stream,
                                  hipMemcpyHostToDevice,
                                  streams[i]));
    }
    for(int i = 0; i < num_streams; i++)
    {
        int offset = i * elements_per_stream;
        cube<<<gridSizePerStream, blockSize, 0, streams[i]>>>(
            d_input1, d_output1, offset, elements_per_stream);
    }
    for(int i = 0; i < num_streams; i++)
    {
        int offset = i * elements_per_stream;
        HIP_ASSERT(hipMemcpyAsync(&h_output1[offset],
                                  &d_output1[offset],
                                  bytes_per_stream,
                                  hipMemcpyDeviceToHost,
                                  streams[i]));
    }

    HIP_ASSERT(hipEventRecord(stop));
    HIP_ASSERT(hipEventSynchronize(stop));

    float milliseconds = 0;
    HIP_ASSERT(hipEventElapsedTime(&milliseconds, start, stop));

    // Test hipStreamPerThread with multiple threads
    const size_t             thread_cnt = 8;
    std::vector<std::thread> threads(thread_cnt);
    for(auto& thread : threads)
    {
        thread = std::thread(copy_to_dev);
    }
    for(auto& thread : threads)
    {
        thread.detach();
    }

    // Release device memory
    HIP_ASSERT(hipFree(d_input1));
    HIP_ASSERT(hipFree(d_output1));

    // Release host memory
    HIP_ASSERT(hipHostFree(h_input1));
    HIP_ASSERT(hipHostFree(h_output1));

    // Destroy streams
    for(int i = 0; i < num_streams; ++i)
    {
        HIP_ASSERT(hipStreamDestroy(streams[i]));
    }

    return 0;
}
