/*
Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <vector>

#define HIP_CHECK(cmd)                                                                             \
    {                                                                                              \
        hipError_t error = cmd;                                                                    \
        if(error != hipSuccess)                                                                    \
        {                                                                                          \
            std::cerr << "HIP error: '" << #cmd << "' returned " << hipGetErrorString(error)       \
                      << " (" << error << ") at " << __FILE__ << ":" << __LINE__ << std::endl;     \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    }

// Simple kernel that does minimal work
__global__ void
simpleKernel(int* data, int value, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size)
    {
        data[idx] = value + idx;
    }
}

namespace
{
struct config
{
    int num_kernels       = 2000;
    int num_iterations    = 200;
    int array_size        = 256;
    int progress_interval = 50;
};

void
print_usage(const char* argv0)
{
    std::cerr << "Usage: " << argv0
              << " [num_kernels] [num_iterations] [array_size] [progress_interval]" << std::endl;
}

int
parse_positive_integer(const char* arg_name, const char* value)
{
    char* end = nullptr;
    auto  ret = std::strtol(value, &end, 10);
    if(end == value || (end != nullptr && *end != '\0') || ret <= 0)
    {
        std::cerr << "Invalid " << arg_name << ": " << value << std::endl;
        exit(EXIT_FAILURE);
    }

    return static_cast<int>(ret);
}

config
parse_args(int argc, char** argv)
{
    config cfg{};

    if(argc > 5)
    {
        print_usage(argv[0]);
        exit(EXIT_FAILURE);
    }

    if(argc > 1) cfg.num_kernels = parse_positive_integer("num_kernels", argv[1]);
    if(argc > 2) cfg.num_iterations = parse_positive_integer("num_iterations", argv[2]);
    if(argc > 3) cfg.array_size = parse_positive_integer("array_size", argv[3]);
    if(argc > 4) cfg.progress_interval = parse_positive_integer("progress_interval", argv[4]);

    if(cfg.array_size < 256)
    {
        std::cerr << "array_size must be at least 256, got " << cfg.array_size << std::endl;
        exit(EXIT_FAILURE);
    }

    return cfg;
}
}  // namespace

int
main(int argc, char** argv)
{
    const auto cfg = parse_args(argc, argv);

    std::cout << "Creating HIP graph with " << cfg.num_kernels << " kernel launches" << std::endl;
    std::cout << "Will execute graph " << cfg.num_iterations << " times" << std::endl;
    std::cout << "Array size: " << cfg.array_size << std::endl;
    std::cout << "Progress interval: " << cfg.progress_interval << std::endl;

    // Allocate device memory
    int* d_data;
    HIP_CHECK(hipMalloc(&d_data, cfg.array_size * sizeof(int)));

    // Create graph
    hipGraph_t graph;
    HIP_CHECK(hipGraphCreate(&graph, 0));

    // Create stream for graph capture
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // Begin graph capture
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));

    // Launch many kernels
    // Use fixed block size and derive grid size from array_size to prevent out-of-bounds access
    constexpr int BLOCK_SIZE = 256;
    dim3          blockSize(BLOCK_SIZE);
    dim3          gridSize((cfg.array_size + BLOCK_SIZE - 1) / BLOCK_SIZE);

    for(int i = 0; i < cfg.num_kernels; i++)
    {
        hipLaunchKernelGGL(simpleKernel, gridSize, blockSize, 0, stream, d_data, i, cfg.array_size);
        HIP_CHECK(hipGetLastError());
    }

    // End graph capture
    HIP_CHECK(hipStreamEndCapture(stream, &graph));

    // Create executable graph
    hipGraphExec_t graphExec;
    HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    std::cout << "Graph created and instantiated successfully" << std::endl;
    std::cout << "Starting graph execution loop..." << std::endl;

    // Start timing
    auto start = std::chrono::high_resolution_clock::now();

    // Execute the graph multiple times
    for(int iter = 0; iter < cfg.num_iterations; iter++)
    {
        HIP_CHECK(hipGraphLaunch(graphExec, stream));

        if((iter + 1) % cfg.progress_interval == 0 || (iter + 1) == cfg.num_iterations)
        {
            std::cout << "Completed " << (iter + 1) << " iterations" << std::endl;
        }
    }

    // Wait for completion
    HIP_CHECK(hipStreamSynchronize(stream));

    // End timing
    auto                          end     = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "All iterations completed successfully!" << std::endl;
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n=== Timing Results ===" << std::endl;
    std::cout << "Total execution time: " << elapsed.count() << " seconds" << std::endl;
    std::cout << "Total kernel launches: " << (cfg.num_kernels * cfg.num_iterations) << std::endl;
    std::cout << "Average time per iteration: " << (elapsed.count() / cfg.num_iterations)
              << " seconds" << std::endl;
    std::cout << "======================" << std::endl;

    // Cleanup
    HIP_CHECK(hipGraphExecDestroy(graphExec));
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(d_data));

    std::cout << "Test completed successfully" << std::endl;

    return 0;
}
