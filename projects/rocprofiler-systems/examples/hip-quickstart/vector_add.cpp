/*
MIT License

Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

/**
 * @file vector_add.cpp
 * @brief Simple vector addition example for profiling with ROCm Systems Profiler
 *
 * This example demonstrates:
 * - Basic HIP kernel launch
 * - Memory transfer patterns
 * - Kernel execution profiling
 *
 * PROFILING THIS EXAMPLE:
 *
 *   # Quick profiling with sampling
 *   rocprof-sys-sample --quick --hip-trace -- ./vector_add
 *
 *   # Detailed profiling with instrumentation
 *   rocprof-sys-instrument -o vector_add.inst -- ./vector_add
 *   rocprof-sys-run --hip-trace --trace -- ./vector_add.inst
 *
 * WHAT TO LOOK FOR IN PROFILING OUTPUT:
 * - hipMemcpy time (data transfer overhead)
 * - vector_add_kernel execution time
 * - Ratio of compute time to data transfer time
 */

#include <cmath>
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

// Error checking macro
#define HIP_CHECK(cmd)                                                                   \
    {                                                                                    \
        hipError_t error = (cmd);                                                        \
        if(error != hipSuccess)                                                          \
        {                                                                                \
            std::cerr << "HIP error: " << hipGetErrorString(error) << " at " << __FILE__ \
                      << ":" << __LINE__ << std::endl;                                   \
            exit(EXIT_FAILURE);                                                          \
        }                                                                                \
    }

/**
 * @brief GPU kernel for vector addition
 * @param a Input vector A
 * @param b Input vector B
 * @param c Output vector C = A + B
 * @param n Vector size
 *
 * PROFILING INSIGHT: This kernel performs a simple memory-bound operation.
 * Look for low arithmetic intensity (few FLOPs per byte transferred).
 */
__global__ void
vector_add_kernel(const float* a, const float* b, float* c, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < n)
    {
        // Simple addition - very low computational intensity
        c[idx] = a[idx] + b[idx];
    }
}

/**
 * @brief Verify the computation results
 */
bool
verify_results(const std::vector<float>& a, const std::vector<float>& b,
               const std::vector<float>& c, float tolerance = 1e-5f)
{
    for(size_t i = 0; i < a.size(); ++i)
    {
        float expected = a[i] + b[i];
        if(std::abs(c[i] - expected) > tolerance)
        {
            std::cerr << "Verification failed at index " << i << ": expected " << expected
                      << ", got " << c[i] << std::endl;
            return false;
        }
    }
    return true;
}

int
main(int argc, char** argv)
{
    // Parse command line for vector size
    size_t n = 1000000;  // Default: 1 million elements
    if(argc > 1)
    {
        n = std::atoll(argv[1]);
    }

    std::cout << "Vector Addition Example\n";
    std::cout << "======================\n";
    std::cout << "Vector size: " << n << " elements\n";
    std::cout << "Memory per vector: " << (n * sizeof(float)) / (1024.0 * 1024.0)
              << " MB\n\n";

    // Allocate and initialize host vectors
    std::vector<float> h_a(n);
    std::vector<float> h_b(n);
    std::vector<float> h_c(n);

    // Initialize input vectors
    for(size_t i = 0; i < n; ++i)
    {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }

    // Allocate device memory
    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    HIP_CHECK(hipMalloc(&d_a, n * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_b, n * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_c, n * sizeof(float)));

    // Copy data to device
    // PROFILING POINT: Look for hipMemcpy time in trace
    std::cout << "Copying data to device...\n";
    HIP_CHECK(hipMemcpy(d_a, h_a.data(), n * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b.data(), n * sizeof(float), hipMemcpyHostToDevice));

    // Configure kernel launch parameters
    const int threads_per_block = 256;
    const int blocks            = (n + threads_per_block - 1) / threads_per_block;

    std::cout << "Launching kernel with " << blocks << " blocks of " << threads_per_block
              << " threads\n";

    // Launch kernel
    // PROFILING POINT: Look for vector_add_kernel execution time
    hipLaunchKernelGGL(vector_add_kernel, dim3(blocks), dim3(threads_per_block),
                       0,  // Shared memory
                       0,  // Stream
                       d_a, d_b, d_c, n);

    // Check for kernel launch errors
    HIP_CHECK(hipGetLastError());

    // Wait for kernel to complete
    HIP_CHECK(hipDeviceSynchronize());

    // Copy result back to host
    // PROFILING POINT: Look for hipMemcpy time (device to host)
    std::cout << "Copying results back to host...\n";
    HIP_CHECK(hipMemcpy(h_c.data(), d_c, n * sizeof(float), hipMemcpyDeviceToHost));

    // Verify results
    std::cout << "Verifying results...\n";
    if(verify_results(h_a, h_b, h_c))
    {
        std::cout << "SUCCESS: Results are correct!\n";
    }
    else
    {
        std::cerr << "FAILURE: Results are incorrect!\n";
        return EXIT_FAILURE;
    }

    // Clean up
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));

    std::cout << "\nPROFILING TIPS:\n";
    std::cout << "1. Check the ratio of kernel time to memory transfer time\n";
    std::cout << "2. For memory-bound kernels like this, look at memory bandwidth "
                 "utilization\n";
    std::cout
        << "3. Compare actual bandwidth to theoretical peak (~1.6 TB/s for MI200)\n";
    std::cout << "4. Try varying vector size to see impact on performance\n";

    return EXIT_SUCCESS;
}
