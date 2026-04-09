// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

/*
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
