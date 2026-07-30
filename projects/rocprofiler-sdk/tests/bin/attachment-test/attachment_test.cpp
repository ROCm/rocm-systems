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
#include <sys/wait.h>
#include <unistd.h>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace
{
// required type for signal handlers
volatile std::sig_atomic_t sigint_received   = 0;
volatile std::sig_atomic_t sigwinch_received = 0;
}  // namespace

// signal handler - must have C linkage
extern "C" void
attachment_test_signal_handler(int signum)
{
    if(signum == SIGINT)
    {
        sigint_received = signum;
    }
    else if(signum == SIGWINCH)
    {
        sigwinch_received = signum;
    }
}

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
execute_kernels(const size_t tid, const size_t device_id)
{
    // Set device
    HIP_ASSERT(hipSetDevice(device_id));

    auto* stream = hipStream_t{nullptr};
    HIP_ASSERT(hipStreamCreate(&stream));

    // Allocate memory
    const int    size  = 512 * 512;  // 256K elements
    const size_t bytes = size * sizeof(float);

    float* h_data = nullptr;
    float* d_data = nullptr;

    HIP_ASSERT(hipHostMalloc(&h_data, bytes));
    HIP_ASSERT(hipMalloc(&d_data, bytes));

    // Initialize data
    for(int i = 0; i < size; ++i)
    {
        h_data[i] = static_cast<float>(i);
    }

    // Run kernels in a loop for a while
    {
        // compose string first to avoid multithreaded handling of cout << operator
        auto msg = std::stringstream{};
        msg << "Starting kernel execution loop for thread " << tid << " on device " << device_id
            << "...\n";
        std::cout << msg.str();
    }
    size_t iter = 0;

    while(!sigint_received)
    {
        // Add ROCTX markers for better profiling
        std::string range_name = "Iteration_" + std::to_string(iter + 1);
        roctxRangePush(range_name.c_str());

        // Copy data to device
        roctxMark("Start_H2D_Copy");
        auto err = hipMemcpyAsync(d_data, h_data, bytes, hipMemcpyHostToDevice, stream);
        if(err != hipSuccess)
        {
            std::cerr << "Failed to copy data for thread " << tid << " on device " << device_id
                      << "...\n";
            roctxRangePop();
            break;
        }

        // Launch kernel
        roctxMark("Launch_Kernel");
        int threads_per_block = 256;
        int blocks_per_grid   = size / threads_per_block;

        hipLaunchKernelGGL(
            simple_kernel, dim3(blocks_per_grid), dim3(threads_per_block), 0, stream, d_data, size);

        // Copy data back
        roctxMark("Start_D2H_Copy");
        err = hipMemcpyAsync(h_data, d_data, bytes, hipMemcpyDeviceToHost, stream);
        if(err != hipSuccess)
        {
            std::cerr << "Failed to copy data for thread " << tid << " on device " << device_id
                      << "...\n";
            roctxRangePop();
            break;
        }

        // Wait for completion
        roctxMark("Stream_Synchronize");
        err = hipStreamSynchronize(stream);
        if(err != hipSuccess)
        {
            std::cerr << "Failed to synchronize stream " << stream << " with thread " << tid
                      << " on device " << device_id << "...\n";
            roctxRangePop();
            break;
        }

        roctxRangePop();

        if(sigint_received)
        {
            break;
        }

        // Small delay between iterations
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        iter++;
    }

    {
        // compose string first to avoid multithreaded handling of cout << operator
        auto msg = std::stringstream{};
        msg << "Kernel execution loop completed " << iter << " iterations for thread " << tid
            << " on device " << device_id << "\n";
        std::cout << msg.str();
    }

    HIP_ASSERT(hipStreamDestroy(stream));
    // Cleanup
    HIP_ASSERT(hipHostFree(h_data));
    HIP_ASSERT(hipFree(d_data));
}

int
main(int argc, char** argv)
{
    // Install signal handler for SIGINT and SIGWINCH
    std::signal(SIGINT, attachment_test_signal_handler);
    std::signal(SIGWINCH, attachment_test_signal_handler);

    size_t nthreads{8};
    int    ndevices{0};
    bool   fork_child{false};
    int    positional{0};
    for(int i = 1; i < argc; ++i)
    {
        auto _arg = std::string{argv[i]};
        if(_arg == "--fork-child")
        {
            fork_child = true;
        }
        else if(_arg == "?" || _arg == "-h" || _arg == "--help")
        {
            fprintf(
                stderr,
                "usage: attachment-test [--fork-child] [NUM_THREADS (%zu)] [NUM_DEVICES (%d)]\n",
                nthreads,
                ndevices);
            exit(EXIT_SUCCESS);
        }
        else if(positional == 0)
        {
            nthreads = std::atoll(argv[i]);
            ++positional;
        }
        else if(positional == 1)
        {
            ndevices = std::stoi(argv[i]);
            ++positional;
        }
    }

    // Fork a child process before HIP initialization so each process gets a
    // clean HIP context. Used by the attach-tree test to exercise --attach-children.
    pid_t child_pid = -1;
    if(fork_child)
    {
        child_pid = fork();
        if(child_pid < 0)
        {
            std::cerr << "fork failed\n";
            return 1;
        }
    }

    std::cout << "Attachment test app started with PID: " << getpid() << std::endl;

    // Initialize HIP
    int device_count = 0;
    HIP_ASSERT(hipGetDeviceCount(&device_count));
    if(device_count == 0)
    {
        std::cerr << "No HIP devices found or error getting device count" << std::endl;
        return 1;
    }

    // Default ndevices to device_count. Ensure that we do not use more devices than are available
    ndevices = ndevices == 0 ? device_count : ndevices;
    if(ndevices > device_count)
    {
        std::cout << "Using " << device_count << " HIP devices instead of the requested "
                  << ndevices << "\n";
        ndevices = device_count;
    }

    auto _threads = std::vector<std::thread>{};
    _threads.reserve(nthreads);

    for(size_t i = 0; i < nthreads; ++i)
        _threads.emplace_back(execute_kernels, i, i % ndevices);
    for(auto& itr : _threads)
        itr.join();

    if(sigwinch_received)
    {
        std::cout << "Attachment test process " << getpid() << " received signal " << SIGWINCH
                  << "\n";
    }
    std::cout << "Attachment test app finished" << std::endl;

    if(child_pid > 0)
    {
        kill(child_pid, SIGINT);
        int child_status = 0;
        waitpid(child_pid, &child_status, 0);
        if(child_status != 0)
        {
            std::cout << "Child process " << child_pid
                      << " returned non-zero status: " << child_status << std::endl;
            return 1;
        }
    }

    return 0;
}
