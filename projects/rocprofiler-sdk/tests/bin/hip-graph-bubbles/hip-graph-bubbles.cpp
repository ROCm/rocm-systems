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
#include <rocprofiler-sdk-roctx/roctx.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

#define HIP_CHECK(cmd)                                                                             \
    {                                                                                              \
        hipError_t error = cmd;                                                                    \
        if(error != hipSuccess)                                                                    \
        {                                                                                          \
            std::cerr << "HIP error: " << hipGetErrorString(error) << " at " << __FILE__ << ":"    \
                      << __LINE__ << " :: " << #cmd << std::endl;                                  \
            exit(EXIT_FAILURE);                                                                    \
        }                                                                                          \
    }

// Simple kernel that does minimal work
__global__ void
simpleKernel(int* data, int value)
{
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = value + idx;
}

int
main(int argc, char** argv)
{
    int       NUM_KERNELS    = 2000;
    int       NUM_ITERATIONS = 200;
    const int ARRAY_SIZE     = 256;

    for(int i = 1; i < argc; i++)
    {
        if(std::string_view{argv[i]} == "--help" || std::string_view{argv[i]} == "-h" ||
           std::string_view{argv[i]} == "-?")
        {
            std::cout << "Usage: " << argv[0] << " [num_kernels (default=" << NUM_KERNELS
                      << ")] [num_iterations (default=" << NUM_ITERATIONS << ")]" << std::endl;
            return 0;
        }
    }

    if(argc > 1) NUM_KERNELS = std::stoi(argv[1]);
    if(argc > 2) NUM_ITERATIONS = std::stoi(argv[2]);

    std::cout << "Creating HIP graph with " << NUM_KERNELS << " kernel launches" << std::endl;
    std::cout << "Will execute graph " << NUM_ITERATIONS << " times" << std::endl;

    // Allocate device memory
    int* d_data;
    HIP_CHECK(hipMalloc(&d_data, ARRAY_SIZE * sizeof(int)));

    // Create graph
    hipGraph_t graph;
    HIP_CHECK(hipGraphCreate(&graph, 0));

    // Create stream for graph capture
    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // Begin graph capture
    HIP_CHECK(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));

    // Launch many kernels
    dim3 blockSize(ARRAY_SIZE, 1, 1);
    dim3 gridSize(1, 1, 1);

    for(int i = 0; i < NUM_KERNELS; i++)
    {
        hipLaunchKernelGGL(simpleKernel, gridSize, blockSize, 0, stream, d_data, i);
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
    for(int iter = 0; iter < NUM_ITERATIONS; iter++)
    {
        roctxRangePush("graph_launch");
        HIP_CHECK(hipGraphLaunch(graphExec, stream));
        roctxRangePop();

        if((iter + 1) % 50 == 0)
        {
            std::cout << "Completed " << (iter + 1) << " iterations" << std::endl;
        }

        // Wait for completion
        // if(iter % 5 == 4)
        HIP_CHECK(hipStreamSynchronize(stream));

        // std::cout << "Synchronized after iteration " << (iter + 1) << std::endl;
        // std::this_thread::sleep_for(std::chrono::milliseconds(100));  // Sleep to simulate work
        // between iterations
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
    std::cout << "Total kernel launches: " << (NUM_KERNELS * NUM_ITERATIONS) << std::endl;
    std::cout << "Average time per iteration: " << (elapsed.count() / NUM_ITERATIONS) << " seconds"
              << std::endl;
    std::cout << "======================" << std::endl;

    // Cleanup
    HIP_CHECK(hipGraphExecDestroy(graphExec));
    HIP_CHECK(hipGraphDestroy(graph));
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(d_data));

    std::cout << "Test completed successfully" << std::endl;

    return 0;
}
