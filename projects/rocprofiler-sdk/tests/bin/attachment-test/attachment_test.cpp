// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <hip/hip_runtime.h>
#include <rocprofiler-sdk-roctx/roctx.h>
#include <unistd.h>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

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

__global__ void
simple_kernel(float* data, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size)
    {
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

void
execute_kernels(const size_t      tid,
                const hipStream_t stream,
                const size_t      stream_id,
                const size_t      device_id,
                const bool        infinite_loop,
                const size_t      num_iterations)
{
    // Set device
    HIP_ASSERT(hipSetDevice(device_id));

    // Allocate memory
    const int    size  = 1024 * 1024;  // 1M elements
    const size_t bytes = size * sizeof(float);

    float* h_data = new float[size];
    float* d_data = nullptr;

    HIP_ASSERT(hipMalloc(&d_data, bytes));

    // Initialize data
    for(int i = 0; i < size; ++i)
    {
        h_data[i] = static_cast<float>(i);
    }

    // Run kernels - infinite loop if infinite_loop is true, otherwise run for specified iterations
    if(infinite_loop)
    {
        std::cout << "Starting infinite kernel execution loop for thread " << tid << " with stream "
                  << stream_id << " on device " << device_id << "...\n";
    }
    else
    {
        std::cout << "Starting kernel execution loop for " << num_iterations
                  << " iterations for thread " << tid << " with stream " << stream_id
                  << " on device " << device_id << "...\n";
    }

    size_t iter = 0;
    while(infinite_loop || iter < num_iterations)
    {
        ++iter;
        // Add ROCTX markers for better profiling
        std::string range_name = "Iteration_" + std::to_string(iter);
        roctxRangePush(range_name.c_str());  // Removed - ROCTx not linked

        // Copy data to device
        roctxMark("Start_H2D_Copy");
        auto err = hipMemcpyAsync(d_data, h_data, bytes, hipMemcpyHostToDevice, stream);
        if(err != hipSuccess)
        {
            std::cerr << "Failed to copy data for thread " << tid << " with stream " << stream_id
                      << " on device " << device_id << "...\n";
            roctxRangePop();  // Removed - ROCTx not linked
            break;
        }

        // Launch kernel
        roctxMark("Launch_Kernel");
        int threads_per_block = 256;
        int blocks_per_grid   = (size + threads_per_block - 1) / threads_per_block;

        hipLaunchKernelGGL(
            simple_kernel, dim3(blocks_per_grid), dim3(threads_per_block), 0, stream, d_data, size);

        // Copy data back
        roctxMark("Start_D2H_Copy");
        err = hipMemcpyAsync(h_data, d_data, bytes, hipMemcpyDeviceToHost, stream);
        if(err != hipSuccess)
        {
            std::cerr << "Failed to copy data for thread " << tid << " with stream " << stream_id
                      << " on device " << device_id << "...\n";
            roctxRangePop();  // Removed - ROCTx not linked
            break;
        }

        // Wait for completion
        roctxMark("Stream_Synchronize");
        err = hipStreamSynchronize(stream);
        if(err != hipSuccess)
        {
            std::cerr << "Failed to synchronize stream " << stream_id << " with thread " << tid
                      << " on device " << device_id << "...\n";
            roctxRangePop();  // Removed - ROCTx not linked
            break;
        }

        roctxRangePop();  // Removed - ROCTx not linked

        // Small delay between iterations
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    // Note: This will never be reached if infinite_loop is true (app will be killed externally)
    std::cout << "Kernel execution loop completed for thread " << tid << " with stream "
              << stream_id << " on device " << device_id << "...\n";

    // Cleanup
    HIP_ASSERT(hipFree(d_data));
    delete[] h_data;
}

int
main(int argc, char** argv)
{
    size_t nthreads{32};
    size_t nstreams{8};
    int    ndevices{0};
    bool   infinite_loop{true};  // Default to infinite loop
    size_t num_iterations{30};   // Default iterations when not in infinite mode
    for(int i = 1; i < argc; ++i)
    {
        auto _arg = std::string{argv[i]};
        if(_arg == "?" || _arg == "-h" || _arg == "--help")
        {
            fprintf(stderr,
                    "usage: attachment-test [NUM_THREADS (%zu)] [NUM_STREAMS (%zu)] "
                    "[NUM_DEVICES (%d)] [INFINITE_LOOP (1=true)] [NUM_ITERATIONS (%zu)]\n",
                    nthreads,
                    nstreams,
                    ndevices,
                    num_iterations);
            exit(EXIT_SUCCESS);
        }
    }
    if(argc > 1) nthreads = std::atoll(argv[1]);
    if(argc > 2) nstreams = std::atoll(argv[2]);
    if(argc > 3) ndevices = std::stoi(argv[3]);
    if(argc > 4) infinite_loop = std::stoi(argv[4]) != 0;
    if(argc > 5) num_iterations = std::atoll(argv[5]);

    std::cout << "Attachment test app started with PID: " << getpid() << std::endl;

    // Initialize HIP
    int device_count = 0;
    HIP_ASSERT(hipGetDeviceCount(&device_count));
    if(device_count == 0)
    {
        std::cerr << "No HIP devices found or error getting device count" << std::endl;
        return 1;
    }
    // Default ndecives to device_count. Ensure that we do not use more devices than are available
    ndevices = ndevices == 0 ? device_count : ndevices;
    if(ndevices > device_count)
    {
        std::cout << "Using " << device_count << " HIP devices instead of the requested "
                  << ndevices << "\n";
        ndevices = device_count;
    }

    std::cout << "After first call " << getpid() << std::endl;

    auto _threads = std::vector<std::thread>{};
    auto _streams = std::vector<hipStream_t>(nstreams);
    _threads.reserve(nthreads);

    for(auto& itr : _streams)
        HIP_ASSERT(hipStreamCreate(&itr));
    for(size_t i = 0; i < nthreads; ++i)
        _threads.emplace_back(execute_kernels,
                              i,
                              _streams.at(i % nstreams),
                              i % nstreams,
                              i % ndevices,
                              infinite_loop,
                              num_iterations);
    for(auto& itr : _threads)
        itr.join();

    // Destroy streams
    for(auto itr : _streams)
        HIP_ASSERT(hipStreamDestroy(itr));

    std::cout << "Attachment test app finished" << std::endl;

    return 0;
}
