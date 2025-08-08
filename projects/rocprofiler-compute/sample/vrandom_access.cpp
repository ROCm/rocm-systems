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



An example code to execute random access to explore cache hits/misses in L2 Cache.
*/


#include <hip/hip_runtime.h>
#include <assert.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <algorithm>

#define HIP_ASSERT(x) (assert((x) == hipSuccess))

// Kernel: random access, each thread picks a random index
__global__ void randomAccessKernel(int *d_data, int N, unsigned int *d_seeds)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
    {
        unsigned int seed = d_seeds[tid];
        // Simple XORShift
        seed ^= (seed << 13);
        seed ^= (seed >> 17);
        seed ^= (seed << 5);
        int idx = seed % N;
        d_data[idx] += 1;
    }
}

int main()
{
    hipError_t hip_status;

    const int N = 1 << 24; // Try 16M elements to exceed cache
    size_t size = N * sizeof(int);

    // Host memory
    std::vector<int> h_data(N, 0);
    std::vector<unsigned int> h_seeds(N);

    // Generate seeds
    srand(time(nullptr));
    for (int i = 0; i < N; ++i)
    {
        // Keep them diverse. Could be random or based on i
        h_seeds[i] = rand();
    }

    // Allocate device memory
    int *d_data;
    unsigned int *d_seeds;
    HIP_ASSERT(hipMalloc(&d_data, size));
    HIP_ASSERT(hipMalloc(&d_seeds, N * sizeof(unsigned int)));

    // Copy h_data to device
    HIP_ASSERT(hipMemcpy(d_data, h_data.data(), size, hipMemcpyHostToDevice));
    HIP_ASSERT(hipMemcpy(d_seeds, h_seeds.data(), N * sizeof(unsigned int), hipMemcpyHostToDevice));

    // Configure kernel
    dim3 blockSize(64);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    // Launch kernel
    hipLaunchKernelGGL(randomAccessKernel, gridSize, blockSize, 0, 0, d_data, N, d_seeds);
    hip_status = hipDeviceSynchronize();

    HIP_ASSERT(hipMemcpy(h_data.data(), d_data, size, hipMemcpyDeviceToHost));

    // Cleanup
    HIP_ASSERT(hipFree(d_data));
    HIP_ASSERT(hipFree(d_seeds));
    ;

    std::cout << "RandomAccess HIP test completed.\n";
    return 0;
}
