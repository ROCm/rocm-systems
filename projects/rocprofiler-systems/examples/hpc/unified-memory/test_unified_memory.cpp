// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cstdlib>
#include <hip/hip_runtime.h>
#include <iostream>

#define hipCheck(call)                                                                   \
    do                                                                                   \
    {                                                                                    \
        hipError_t error = call;                                                         \
        if(error != hipSuccess)                                                          \
        {                                                                                \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << ": "          \
                      << hipGetErrorString(error) << '\n';                               \
            exit(1);                                                                     \
        }                                                                                \
    } while(0)

__global__ void
doubleKernel(int* data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n)
    {
        data[idx] *= 2;
    }
}

long long
compute_expected_sum(int n)
{
    long long expected = 0;
    for(int i = 0; i < n; i++)
    {
        expected += (i % 1000) * 2;
    }
    return expected;
}

int
main()
{
    const int    N    = 16 * 1024 * 1024;
    const size_t size = N * sizeof(int);

    std::cout << "Allocating " << size / (1024.0 * 1024.0) << " MB of unified memory..."
              << '\n';

    int* data;
    hipCheck(hipMallocManaged(&data, size));

    std::cout << "Initializing data on CPU..." << '\n';
    for(int i = 0; i < N; i++)
    {
        data[i] = i % 1000;
    }

    std::cout << "Launching GPU kernel (triggers Host->Device migration)..." << '\n';
    int blockSize = 256;
    int gridSize  = (N + blockSize - 1) / blockSize;

    hipLaunchKernelGGL(doubleKernel, dim3(gridSize), dim3(blockSize), 0, 0, data, N);

    hipCheck(hipDeviceSynchronize());
    std::cout << "GPU kernel complete." << '\n';

    std::cout << "Accessing data on CPU (triggers Device->Host migration)..." << '\n';
    long long sum = 0;
    for(int i = 0; i < N; i++)
    {
        sum += data[i];
    }

    long long expected = compute_expected_sum(N);
    std::cout << "Sum: " << sum << '\n';
    std::cout << "Expected: " << expected << '\n';

    hipCheck(hipFree(data));

    if(sum != expected)
    {
        std::cerr << "FAIL: Sum mismatch! Got " << sum << ", expected " << expected
                  << '\n';
        return 1;
    }

    std::cout << "PASS: Test complete!" << '\n';
    return 0;
}
