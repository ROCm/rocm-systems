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
#include <iostream>
#include <thread>
#include <vector>

__global__ void
simple_kernel(float* data, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < size)
    {
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

int
main(int /*argc*/, char** /*argv*/)
{
    std::cout << "Attachment test app started with PID: " << getpid() << std::endl;

    // Initialize HIP
    int        device_count = 0;
    hipError_t err          = hipGetDeviceCount(&device_count);
    if(err != hipSuccess || device_count == 0)
    {
        std::cerr << "No HIP devices found or error getting device count" << std::endl;
        return 1;
    }

    std::cout << "After first call " << getpid() << std::endl;

    // Set device
    err = hipSetDevice(0);
    if(err != hipSuccess)
    {
        std::cerr << "Failed to set device 0" << std::endl;
        return 1;
    }

    // Allocate memory
    const int    size  = 1024 * 1024;  // 1M elements
    const size_t bytes = size * sizeof(float);

    float* h_data = new float[size];
    float* d_data;

    err = hipMalloc(&d_data, bytes);
    if(err != hipSuccess)
    {
        std::cerr << "Failed to allocate device memory" << std::endl;
        delete[] h_data;
        return 1;
    }

    // Initialize data
    for(int i = 0; i < size; ++i)
    {
        h_data[i] = static_cast<float>(i);
    }

    // Run kernels in a loop for a while
    std::cout << "Starting kernel execution loop..." << std::endl;
    const int num_iterations = 30;

    for(int iter = 0; iter < num_iterations; ++iter)
    {
        // Add ROCTX markers for better profiling
        std::string range_name = "Iteration_" + std::to_string(iter + 1);
        roctxRangePush(range_name.c_str());  // Removed - ROCTx not linked

        // Copy data to device
        roctxMark("Start_H2D_Copy");
        err = hipMemcpy(d_data, h_data, bytes, hipMemcpyHostToDevice);
        if(err != hipSuccess)
        {
            std::cerr << "Failed to copy data to device" << std::endl;
            roctxRangePop();  // Removed - ROCTx not linked
            break;
        }

        // Launch kernel
        roctxMark("Launch_Kernel");
        int threads_per_block = 256;
        int blocks_per_grid   = (size + threads_per_block - 1) / threads_per_block;

        hipLaunchKernelGGL(
            simple_kernel, dim3(blocks_per_grid), dim3(threads_per_block), 0, 0, d_data, size);

        // Copy data back
        roctxMark("Start_D2H_Copy");
        err = hipMemcpy(h_data, d_data, bytes, hipMemcpyDeviceToHost);
        if(err != hipSuccess)
        {
            std::cerr << "Failed to copy data from device" << std::endl;
            roctxRangePop();  // Removed - ROCTx not linked
            break;
        }

        // Wait for completion
        roctxMark("Device_Synchronize");
        err = hipDeviceSynchronize();
        if(err != hipSuccess)
        {
            std::cerr << "Failed to synchronize device" << std::endl;
            roctxRangePop();  // Removed - ROCTx not linked
            break;
        }

        roctxRangePop();  // Removed - ROCTx not linked

        std::cout << "Iteration " << (iter + 1) << "/" << num_iterations << " completed"
                  << std::endl;

        // Small delay between iterations
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    std::cout << "Kernel execution loop completed" << std::endl;

    // Cleanup
    err = hipFree(d_data);
    if(err != hipSuccess)
    {
        std::cerr << "Warning: Failed to free device memory" << std::endl;
    }
    delete[] h_data;

    std::cout << "Attachment test app finished" << std::endl;

    return 0;
}
