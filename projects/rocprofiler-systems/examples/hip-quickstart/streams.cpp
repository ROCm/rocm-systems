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
 * @file streams.cpp
 * @brief HIP streams example showing concurrent kernel execution
 *
 * This example demonstrates:
 * - Sequential vs. concurrent kernel execution
 * - Using HIP streams for overlapping computation
 * - Async memory operations
 * - Improving GPU utilization through concurrency
 *
 * PROFILING THIS EXAMPLE:
 *
 *   # View timeline to see concurrent execution
 *   rocprof-sys-sample --quick --hip-trace -- ./streams
 *
 *   # Visualize in Perfetto (look for overlapping kernels)
 *   Open perfetto-trace.proto in ui.perfetto.dev
 *
 * WHAT TO LOOK FOR:
 * - In sequential mode: kernels execute one after another
 * - In concurrent mode: kernels overlap in timeline
 * - Overall execution time should be lower with streams
 * - GPU utilization should be higher with concurrent execution
 */

#include <chrono>
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

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
 * @brief Simple compute kernel that performs multiple operations
 * @param data Input/output array
 * @param n Array size
 * @param iterations Number of compute iterations (to increase kernel duration)
 *
 * PROFILING INSIGHT: This kernel is designed to run long enough to observe
 * concurrent execution. Look for multiple instances in the timeline.
 */
__global__ void
compute_kernel(float* data, size_t n, int iterations)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < n)
    {
        float value = data[idx];

        // Perform multiple operations to increase kernel time
        for(int i = 0; i < iterations; ++i)
        {
            value = value * 1.001f + 0.001f;
            value = sqrtf(value);
        }

        data[idx] = value;
    }
}

/**
 * @brief Run kernels sequentially (no streams)
 */
double
run_sequential(std::vector<float*>& d_arrays, size_t n, int num_streams, int iterations)
{
    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;

    auto start = std::chrono::high_resolution_clock::now();

    // Launch all kernels in default stream (sequential execution)
    for(int i = 0; i < num_streams; ++i)
    {
        hipLaunchKernelGGL(compute_kernel, dim3(blocks), dim3(threads), 0, 0, d_arrays[i],
                           n, iterations);
    }

    HIP_CHECK(hipDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

/**
 * @brief Run kernels concurrently using HIP streams
 */
double
run_concurrent(std::vector<float*>& d_arrays, std::vector<hipStream_t>& streams, size_t n,
               int num_streams, int iterations)
{
    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;

    auto start = std::chrono::high_resolution_clock::now();

    // Launch kernels in different streams (concurrent execution)
    for(int i = 0; i < num_streams; ++i)
    {
        hipLaunchKernelGGL(compute_kernel, dim3(blocks), dim3(threads), 0, streams[i],
                           d_arrays[i], n, iterations);
    }

    // Wait for all streams to complete
    for(int i = 0; i < num_streams; ++i)
    {
        HIP_CHECK(hipStreamSynchronize(streams[i]));
    }

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

/**
 * @brief Demonstrate async memory operations with streams
 */
double
run_with_async_memory(std::vector<float*>& h_arrays, std::vector<float*>& d_arrays,
                      std::vector<hipStream_t>& streams, size_t n, int num_streams,
                      int iterations)
{
    const int threads = 256;
    const int blocks  = (n + threads - 1) / threads;

    auto start = std::chrono::high_resolution_clock::now();

    // Launch async operations in each stream:
    // 1. Async memory copy H2D
    // 2. Kernel execution
    // 3. Async memory copy D2H
    for(int i = 0; i < num_streams; ++i)
    {
        HIP_CHECK(hipMemcpyAsync(d_arrays[i], h_arrays[i], n * sizeof(float),
                                 hipMemcpyHostToDevice, streams[i]));

        hipLaunchKernelGGL(compute_kernel, dim3(blocks), dim3(threads), 0, streams[i],
                           d_arrays[i], n, iterations);

        HIP_CHECK(hipMemcpyAsync(h_arrays[i], d_arrays[i], n * sizeof(float),
                                 hipMemcpyDeviceToHost, streams[i]));
    }

    // Wait for all streams
    for(int i = 0; i < num_streams; ++i)
    {
        HIP_CHECK(hipStreamSynchronize(streams[i]));
    }

    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int
main(int argc, char** argv)
{
    // Parse parameters
    size_t n           = 1000000;  // Array size per stream
    int    num_streams = 4;        // Number of concurrent streams
    int    iterations  = 1000;     // Iterations per kernel (to increase kernel time)

    if(argc > 1) n = std::atoll(argv[1]);
    if(argc > 2) num_streams = std::atoi(argv[2]);
    if(argc > 3) iterations = std::atoi(argv[3]);

    std::cout << "HIP Streams Example\n";
    std::cout << "===================\n";
    std::cout << "Array size per stream: " << n << " elements\n";
    std::cout << "Number of streams: " << num_streams << "\n";
    std::cout << "Kernel iterations: " << iterations << "\n\n";

    // Allocate host memory (pinned for async operations)
    std::vector<float*> h_arrays(num_streams);
    for(int i = 0; i < num_streams; ++i)
    {
        HIP_CHECK(hipHostMalloc(&h_arrays[i], n * sizeof(float)));

        // Initialize
        for(size_t j = 0; j < n; ++j)
        {
            h_arrays[i][j] = static_cast<float>(j) * 0.001f;
        }
    }

    // Allocate device memory
    std::vector<float*> d_arrays(num_streams);
    for(int i = 0; i < num_streams; ++i)
    {
        HIP_CHECK(hipMalloc(&d_arrays[i], n * sizeof(float)));
    }

    // Create streams
    std::vector<hipStream_t> streams(num_streams);
    for(int i = 0; i < num_streams; ++i)
    {
        HIP_CHECK(hipStreamCreate(&streams[i]));
    }

    // === SEQUENTIAL EXECUTION ===
    std::cout << "Running SEQUENTIAL execution...\n";
    double seq_time = run_sequential(d_arrays, n, num_streams, iterations);
    std::cout << "  Execution time: " << seq_time << " ms\n\n";

    // === CONCURRENT EXECUTION ===
    std::cout << "Running CONCURRENT execution (with streams)...\n";
    double conc_time = run_concurrent(d_arrays, streams, n, num_streams, iterations);
    std::cout << "  Execution time: " << conc_time << " ms\n\n";

    // === WITH ASYNC MEMORY OPERATIONS ===
    std::cout << "Running with ASYNC MEMORY operations...\n";
    double async_time =
        run_with_async_memory(h_arrays, d_arrays, streams, n, num_streams, iterations);
    std::cout << "  Execution time: " << async_time << " ms\n\n";

    // Calculate speedups
    double stream_speedup = seq_time / conc_time;
    double async_speedup  = seq_time / async_time;

    std::cout << "RESULTS:\n";
    std::cout << "  Sequential time:  " << seq_time << " ms\n";
    std::cout << "  Concurrent time:  " << conc_time << " ms (speedup: " << stream_speedup
              << "x)\n";
    std::cout << "  Async mem time:   " << async_time << " ms (speedup: " << async_speedup
              << "x)\n\n";

    // Clean up
    for(int i = 0; i < num_streams; ++i)
    {
        HIP_CHECK(hipStreamDestroy(streams[i]));
        HIP_CHECK(hipFree(d_arrays[i]));
        HIP_CHECK(hipHostFree(h_arrays[i]));
    }

    std::cout << "PROFILING TIPS:\n";
    std::cout << "1. View the Perfetto trace to see kernel overlap\n";
    std::cout << "2. Look for gaps between kernels in sequential mode\n";
    std::cout << "3. Check GPU utilization % (should be higher with streams)\n";
    std::cout << "4. Observe how async memory operations overlap with kernels\n";
    std::cout << "5. Try different numbers of streams to find optimal concurrency\n\n";

    std::cout << "TIP: If you don't see much speedup, try:\n";
    std::cout << "  - Increasing array size: ./streams 10000000\n";
    std::cout << "  - Increasing iterations: ./streams 1000000 4 5000\n";
    std::cout << "  - More streams: ./streams 1000000 8\n";

    return EXIT_SUCCESS;
}
