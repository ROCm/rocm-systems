// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <random>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <vector>

// HIP and ROCm profiling headers
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <rocprofiler-sdk-roctx/roctx.h>

// Define HIP_API_CALL macro for error handling.
#define HIP_API_CALL(CALL)                                                               \
    {                                                                                    \
        hipError_t error_ = (CALL);                                                      \
        if(error_ != hipSuccess)                                                         \
        {                                                                                \
            auto _hip_api_print_lk = auto_lock_t{ print_lock };                          \
            fprintf(stderr, "%s:%d :: HIP error : %s\n", __FILE__, __LINE__,             \
                    hipGetErrorString(error_));                                          \
            throw std::runtime_error("hip_api_call");                                    \
        }                                                                                \
    }

namespace
{
using auto_lock_t = std::unique_lock<std::mutex>;
auto print_lock   = std::mutex{};
}  // namespace

// HIP Kernel Function
__global__ void
hipKernelLaunch(int* data)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    data[idx] += 1;
}

// Function to execute GPU workload with ROCTx profiling
void
gpu_workload()
{
    // Start a profiling range and push a sub-range for launching the kernel.
    uint64_t rangeId = roctxRangeStart("roctxRangeStart_GPU_Compute");
    roctxRangePush("roctxRangePush_HIP_Kernel");

    const int N      = 256;
    int*      d_data = nullptr;

    // Allocate device memory
    HIP_API_CALL(hipMalloc(&d_data, N * sizeof(int)));

    // Launch the kernel
    hipLaunchKernelGGL(HIP_KERNEL_NAME(hipKernelLaunch), dim3(1), dim3(N), 0, 0, d_data);

    // Wait for GPU to finish
    HIP_API_CALL(hipDeviceSynchronize());

    // Free device memory
    HIP_API_CALL(hipFree(d_data));

    // Pop the sub-range and stop the profiling range
    roctxRangePop();
    roctxRangeStop(rangeId);
}

// Function executed in a separate thread with ROCTx annotations.
void
roctxThreadFunc()
{
    roctxNameOsThread("roctxNameOsThread_New");
    roctxMark("roctxMark_Thread_Start");
    gpu_workload();
    roctxMark("roctxMark_End");
}

void
run_profiling()
{
    // Label HIP device and stream
    int deviceId{ 0 };
    HIP_API_CALL(hipGetDevice(&deviceId));
    roctxNameHipDevice("roctxNameHipDevice_device_id", deviceId);

    hipStream_t stream = {};
    HIP_API_CALL(hipStreamCreate(&stream));
    roctxNameHipStream("roctxNameHipStream_hip_stream", stream);

    // Insert a marker before the GPU workload
    roctxMark("roctxMark_GPU_workload");

    // Start a nested profiling range.
    roctxRangePush("roctxRangePush_run_profiling");

    // Execute GPU workload
    gpu_workload();

    // Pause profiling steps using ROCTx APIs.
    roctx_thread_id_t roctx_tid{};  // Thread identifier structure
    roctxGetThreadId(&roctx_tid);

    // Set names for OS thread, HSA agent, HIP device and stream.
    roctxNameOsThread(std::to_string(roctx_tid).c_str());
    // Prepare an hsa_agent_t with roctx thread id as a handle (example usage):
    hsa_agent_t hsa_agent = { .handle = roctx_tid };
    roctxNameHsaAgent("roctxNameHsaAgent_hsa_agent", &hsa_agent);
    roctxNameHipDevice("roctxNameHipDevice_hipdevice", 0);
    auto* hip_stream = hipStream_t{};
    roctxNameHipStream("roctxNameHipStream_hip_stream", hip_stream);

    // Pause ROCTx profiling for the current thread.
    roctxProfilerPause(roctx_tid);
    roctxMark("roctxMark_RoctxProfilerPause_End");

    // Start a separate thread executing additional profiling-annotated work.
    std::thread worker(roctxThreadFunc);
    worker.join();

    // Resume ROCTx profiling.
    roctxProfilerResume(roctx_tid);

    // End the nested profiling range.
    roctxRangePop();

    // Insert a marker after execution of workload.
    roctxMark("roctxMark_Finished_GPU");
    HIP_API_CALL(hipStreamDestroy(stream));
}

int
main()
{
    std::cout << "Roctx profiling started!" << std::endl;
    run_profiling();
    std::cout << "Roctx profiling Completed!" << std::endl;
    return 0;
}
