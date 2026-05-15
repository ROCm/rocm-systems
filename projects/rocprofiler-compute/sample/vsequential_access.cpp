/*
##############################################################################bl
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
##############################################################################el



An example code to execute sequential access to explore cache hits/misses in L2 Cache.
*/


#include <hip/hip_runtime.h>
#include <iostream>
#include <assert.h>

#define HIP_ASSERT(x) (assert((x)==hipSuccess))

// Kernel: sequential access, each thread reads/writes an element in order
__global__ void sequentialAccessKernel(int *d_data, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
    {
        d_data[tid] += 1;
    }
}

int main()
{
    hipError_t hip_status;

    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(int);
    // Allocate host memory
    int *h_data = (int *)malloc(size);
    std::fill_n(h_data, N, 0);

    // Allocate device memory
    int *d_data;
    HIP_ASSERT(hipMalloc(&d_data, size));

    // Copy h_data to device
    HIP_ASSERT(hipMemcpy(d_data, h_data, size, hipMemcpyHostToDevice));

    // Configure kernel
    dim3 blockSize(64);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    // Launch kernel
    hipLaunchKernelGGL(sequentialAccessKernel, gridSize, blockSize, 0, 0, d_data, N);
    hip_status = hipDeviceSynchronize();

    // Copy back to host
    HIP_ASSERT(hipMemcpy(h_data, d_data, size, hipMemcpyDeviceToHost));

    // Cleanup
    HIP_ASSERT(hipFree(d_data));
    free(h_data);

    std::cout << "SequentialAccess HIP test completed.\n";
    return 0;
}
