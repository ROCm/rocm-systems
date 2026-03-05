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

// Test for ROCPROFSYS_TRACE_REGION feature
// Uses roctxRangeStart/roctxRangeStop (process-wide markers) for region filtering
//
// Run without filter (traces everything):
//   rocprof-sys -- ./region_filter
//
// Run with filter (only traces inside TargetRegion):
//   ROCPROFSYS_TRACE_REGION=TargetRegion rocprof-sys -- ./region_filter

#include <cstdio>
#include <iostream>

#include <hip/hip_runtime.h>
#include <rocprofiler-sdk-roctx/roctx.h>

#define HIP_CHECK(call)                                                                  \
    do                                                                                   \
    {                                                                                    \
        hipError_t err = call;                                                           \
        if(err != hipSuccess)                                                            \
        {                                                                                \
            fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(err), __FILE__, \
                    __LINE__);                                                           \
            exit(1);                                                                     \
        }                                                                                \
    } while(0)

__global__ void
kernel_outside(float* data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) data[idx] = data[idx] * 2.0f;
}

__global__ void
kernel_inside_target(float* data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) data[idx] = data[idx] + 1.0f;
}

__global__ void
kernel_inside_other(float* data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) data[idx] = data[idx] - 1.0f;
}

int
main()
{
    const int    N    = 1024;
    const size_t size = N * sizeof(float);

    float* d_data;
    HIP_CHECK(hipMalloc(&d_data, size));
    HIP_CHECK(hipMemset(d_data, 0, size));

    std::cout << "=== Region Filter Test ===" << std::endl;
    std::cout << "Expected with ROCPROFSYS_TRACE_REGION=TargetRegion:" << std::endl;
    std::cout << "  - kernel_inside_target: TRACED" << std::endl;
    std::cout << "  - kernel_outside, kernel_inside_other: NOT traced" << std::endl;
    std::cout << std::endl;

    // Kernel OUTSIDE any region - should NOT be traced when filter is active
    std::cout << "Launching kernel_outside (no region)..." << std::endl;
    kernel_outside<<<(N + 255) / 256, 256>>>(d_data, N);
    HIP_CHECK(hipDeviceSynchronize());

    // Start a region that does NOT match the filter
    std::cout << "Starting OtherRegion..." << std::endl;
    roctx_range_id_t other_id = roctxRangeStartA("OtherRegion");

    // Kernel inside OtherRegion - should NOT be traced when filter is active
    std::cout << "Launching kernel_inside_other (OtherRegion)..." << std::endl;
    kernel_inside_other<<<(N + 255) / 256, 256>>>(d_data, N);
    HIP_CHECK(hipDeviceSynchronize());

    roctxRangeStop(other_id);
    std::cout << "Stopped OtherRegion." << std::endl;

    // Start the TARGET region
    std::cout << "Starting TargetRegion..." << std::endl;
    roctx_range_id_t target_id = roctxRangeStartA("TargetRegion");

    // Kernel inside TargetRegion - SHOULD be traced
    std::cout << "Launching kernel_inside_target (TargetRegion)..." << std::endl;
    kernel_inside_target<<<(N + 255) / 256, 256>>>(d_data, N);
    HIP_CHECK(hipDeviceSynchronize());

    // Launch another one to verify multiple kernels work
    std::cout << "Launching kernel_inside_target again (TargetRegion)..." << std::endl;
    kernel_inside_target<<<(N + 255) / 256, 256>>>(d_data, N);
    HIP_CHECK(hipDeviceSynchronize());

    roctxRangeStop(target_id);
    std::cout << "Stopped TargetRegion." << std::endl;

    // Another kernel OUTSIDE - should NOT be traced when filter is active
    std::cout << "Launching kernel_outside (no region)..." << std::endl;
    kernel_outside<<<(N + 255) / 256, 256>>>(d_data, N);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(d_data));

    std::cout << std::endl;
    std::cout << "=== Test complete ===" << std::endl;
    return 0;
}
