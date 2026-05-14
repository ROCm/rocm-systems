// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/**
 * @file tests/bin/code-object-multi-threaded/main.cpp
 *
 * @brief Multi-threaded code object loading stress test for rocprofiler-sdk
 *
 * This test verifies thread-safety when multiple threads concurrently load
 * HIP modules on different GPUs, which triggers concurrent calls to
 * executable_freeze_internal and code object tracing callbacks.
 *
 * Data races tested:
 * - user_data map access (now protected with Synchronized)
 * - contexts vector assignment (now protected with wlock)
 * - is_shutdown flag (now atomic)
 * - end_notified/beg_notified flags (now atomic)
 * - executable_destroy serialization (now protected with mutex)
 */

#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>
#include <vector>

#define HIP_CHECK(expr)                                                                            \
    do                                                                                             \
    {                                                                                              \
        hipError_t _err = (expr);                                                                  \
        if(_err != hipSuccess)                                                                     \
        {                                                                                          \
            std::cerr << "HIP error " << hipGetErrorString(_err) << " at " << __FILE__ << ":"      \
                      << __LINE__ << "\n";                                                         \
            std::abort();                                                                          \
        }                                                                                          \
    } while(0)

// Simple kernel code as string for runtime compilation/loading
static const char* kernel_code = R"(
extern "C" __global__ void dynamic_kernel_a(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) data[idx] = idx * 2;
}

extern "C" __global__ void dynamic_kernel_b(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) data[idx] = idx * 3.14f;
}

extern "C" __global__ void dynamic_kernel_c(int* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) data[idx] = idx * idx;
}
)";

// Static kernels as fallback
__global__ void
test_kernel_a()
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx == 0) printf("Kernel A\n");
}

__global__ void
test_kernel_b(int* data)
{
    int idx   = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = idx * 2;
}

__global__ void
test_kernel_c(float* data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) data[idx] = idx * 3.14f;
}

void
gpu_worker(int gpu_id, int iterations)
{
    // Set device for this thread
    HIP_CHECK(hipSetDevice(gpu_id));

    const int n = 1024;

    // Each thread loads modules in parallel - this triggers concurrent
    // calls to executable_freeze_internal and code object callbacks
    for(int i = 0; i < iterations; ++i)
    {
        // Use hiprtc to compile kernel source at runtime, then load the compiled module
        // This triggers parallel module loading across threads
        hiprtcProgram prog{};
        hiprtcResult  rtc_result =
            hiprtcCreateProgram(&prog, kernel_code, "dynamic_kernels.cu", 0, nullptr, nullptr);

        bool use_dynamic_loading = (rtc_result == HIPRTC_SUCCESS);

        if(use_dynamic_loading)
        {
            // Compile the program
            rtc_result = hiprtcCompileProgram(prog, 0, nullptr);

            if(rtc_result != HIPRTC_SUCCESS)
            {
                size_t log_size{0};
                hiprtcGetProgramLogSize(prog, &log_size);
                if(log_size > 1)
                {
                    auto log = std::vector<char>(log_size);
                    hiprtcGetProgramLog(prog, log.data());
                    std::cerr << "Compilation failed:\n" << log.data() << "\n";
                }
                use_dynamic_loading = false;
            }
        }

        if(use_dynamic_loading)
        {
            // Get the compiled code
            size_t code_size{0};
            hiprtcGetCodeSize(prog, &code_size);

            std::vector<char> code(code_size);
            hiprtcGetCode(prog, code.data());
            hiprtcDestroyProgram(&prog);

            // Load the compiled module - THIS is what triggers the parallel code object loading
            hipModule_t module;
            hipError_t  err = hipModuleLoadData(&module, code.data());

            if(err == hipSuccess)
            {
                // Get functions from the module
                hipFunction_t func_a, func_b, func_c;
                HIP_CHECK(hipModuleGetFunction(&func_a, module, "dynamic_kernel_a"));
                HIP_CHECK(hipModuleGetFunction(&func_b, module, "dynamic_kernel_b"));
                HIP_CHECK(hipModuleGetFunction(&func_c, module, "dynamic_kernel_c"));

                // Allocate device memory
                int*   d_int_data{nullptr};
                float* d_float_data{nullptr};
                HIP_CHECK(hipMalloc(&d_int_data, n * sizeof(int)));
                HIP_CHECK(hipMalloc(&d_float_data, n * sizeof(float)));

                // Launch kernels using module functions
                int   n_arg    = n;
                void* args_a[] = {&d_int_data, &n_arg};
                HIP_CHECK(hipModuleLaunchKernel(
                    func_a, (n + 255) / 256, 1, 1, 256, 1, 1, 0, 0, args_a, nullptr));

                void* args_b[] = {&d_float_data, &n_arg};
                HIP_CHECK(hipModuleLaunchKernel(
                    func_b, (n + 255) / 256, 1, 1, 256, 1, 1, 0, 0, args_b, nullptr));

                void* args_c[] = {&d_int_data, &n_arg};
                HIP_CHECK(hipModuleLaunchKernel(
                    func_c, (n + 255) / 256, 1, 1, 256, 1, 1, 0, 0, args_c, nullptr));
            }
            else
            {
                use_dynamic_loading = false;
            }
        }

        if(!use_dynamic_loading)
        {
            // Fallback to regular kernel launches if module loading fails
            int*   d_int_data{nullptr};
            float* d_float_data{nullptr};
            HIP_CHECK(hipMalloc(&d_int_data, n * sizeof(int)));
            HIP_CHECK(hipMalloc(&d_float_data, n * sizeof(float)));

            test_kernel_a<<<1, 64>>>();
            HIP_CHECK(hipGetLastError());

            test_kernel_b<<<(n + 255) / 256, 256>>>(d_int_data);
            HIP_CHECK(hipGetLastError());

            test_kernel_c<<<(n + 255) / 256, 256>>>(d_float_data, n);
            HIP_CHECK(hipGetLastError());
        }
    }
}

int
main()
{
    std::cout << "Multi-Threaded Code Object Loading Test\n";

    int num_gpus = 0;
    HIP_CHECK(hipGetDeviceCount(&num_gpus));

    if(num_gpus == 0)
    {
        std::cerr << "No GPUs found. Test requires at least one GPU.\n";
        return 1;
    }

    std::cout << "Found " << num_gpus << " GPU(s)\n";

    // Cap at 64 threads to avoid overwhelming the system while still testing concurrency
    int num_threads     = std::min(std::thread::hardware_concurrency(), 64u);
    int threads_per_gpu = num_threads / num_gpus;
    std::cout << "Launching " << num_threads << " threads\n";

    constexpr int iterations = 3;

    // Create worker threads
    auto threads = std::vector<std::thread>{};
    threads.reserve(num_threads);

    for(int gpu_id = 0; gpu_id < num_gpus; ++gpu_id)
    {
        for(int thread_id = 0; thread_id < threads_per_gpu; ++thread_id)
        {
            threads.emplace_back(gpu_worker, gpu_id, iterations);
        }
    }

    // Wait for all threads to complete
    for(auto& t : threads)
    {
        t.join();
    }

    std::cout << "Test completed successfully\n";
    return 0;
}
