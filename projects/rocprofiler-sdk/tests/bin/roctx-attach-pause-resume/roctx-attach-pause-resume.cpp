// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <chrono>
#include <csignal>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

namespace
{
volatile std::sig_atomic_t sigint_received = 0;
}

extern "C" void
roctx_attach_pause_resume_signal_handler(int signum)
{
    if(signum == SIGINT) sigint_received = signum;
}

#define HIP_ASSERT(call)                                                                           \
    do                                                                                             \
    {                                                                                              \
        hipError_t gpu_err = call;                                                                 \
        if(hipSuccess != gpu_err)                                                                  \
        {                                                                                          \
            std::cerr << "HIP error at " << __FILE__ << ":" << __LINE__ << ": "                    \
                      << hipGetErrorString(gpu_err) << "\n";                                       \
            std::exit(EXIT_FAILURE);                                                               \
        }                                                                                          \
    } while(0)

__global__ void
roctx_attach_before_pause_kernel(float* data)
{
    data[threadIdx.x] += 1.0f;
}

__global__ void
roctx_attach_during_pause_kernel(float* data)
{
    data[threadIdx.x] += 2.0f;
}

__global__ void
roctx_attach_after_resume_kernel(float* data)
{
    data[threadIdx.x] += 3.0f;
}

__global__ void
roctx_attach_outside_before_kernel(float* data)
{
    data[threadIdx.x] += 4.0f;
}

__global__ void
roctx_attach_inside_region_kernel(float* data)
{
    data[threadIdx.x] += 5.0f;
}

__global__ void
roctx_attach_outside_after_kernel(float* data)
{
    data[threadIdx.x] += 6.0f;
}

void
wait_for_trigger(const std::string& trigger_file)
{
    constexpr auto max_wait = std::chrono::seconds{30};
    const auto     beg      = std::chrono::steady_clock::now();

    while(!sigint_received)
    {
        if(std::ifstream{trigger_file}.good()) return;
        if(std::chrono::steady_clock::now() - beg > max_wait)
        {
            std::cerr << "Timed out waiting for trigger file: " << trigger_file << "\n";
            std::exit(EXIT_FAILURE);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds{50});
    }
}

template <typename KernelT>
void
launch_kernel(const char* name, KernelT kernel, float* data)
{
    roctxMark(name);
    hipLaunchKernelGGL(kernel, dim3(1), dim3(64), 0, 0, data);
    HIP_ASSERT(hipGetLastError());
    HIP_ASSERT(hipDeviceSynchronize());
}

void
run_normal_mode(float* data)
{
    launch_kernel("before_pause", roctx_attach_before_pause_kernel, data);

    roctxProfilerPause(0);
    launch_kernel("during_pause", roctx_attach_during_pause_kernel, data);

    roctxProfilerResume(0);
    launch_kernel("after_resume", roctx_attach_after_resume_kernel, data);
}

void
run_selected_regions_mode(float* data)
{
    launch_kernel("outside_before", roctx_attach_outside_before_kernel, data);

    roctxProfilerResume(0);
    launch_kernel("inside", roctx_attach_inside_region_kernel, data);

    roctxProfilerPause(0);
    launch_kernel("outside_after", roctx_attach_outside_after_kernel, data);
}

int
main(int argc, char** argv)
{
    std::signal(SIGINT, roctx_attach_pause_resume_signal_handler);

    if(argc != 3)
    {
        std::cerr << "usage: roctx-attach-pause-resume <normal|selected> <trigger-file>\n";
        return EXIT_FAILURE;
    }

    const auto mode         = std::string{argv[1]};
    const auto trigger_file = std::string{argv[2]};

    HIP_ASSERT(hipFree(nullptr));

    float* data = nullptr;
    HIP_ASSERT(hipMalloc(&data, 64 * sizeof(float)));
    HIP_ASSERT(hipMemset(data, 0, 64 * sizeof(float)));

    if(mode == "normal")
    {
        roctxProfilerPause(0);
    }

    std::cout << "ROCTx attach pause/resume target ready in mode: " << mode << "\n";
    wait_for_trigger(trigger_file);

    if(mode == "normal")
    {
        run_normal_mode(data);
    }
    else if(mode == "selected")
    {
        run_selected_regions_mode(data);
    }
    else
    {
        std::cerr << "Unknown mode: " << mode << "\n";
        HIP_ASSERT(hipFree(data));
        return EXIT_FAILURE;
    }

    while(!sigint_received)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds{50});
    }

    HIP_ASSERT(hipFree(data));
    std::cout << "ROCTx attach pause/resume target finished\n";
    return EXIT_SUCCESS;
}
