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

#include <cstdio>
#include <cstdlib>

#define HIP_API_CALL(CALL)                                                                         \
    {                                                                                              \
        hipError_t error_ = (CALL);                                                                \
        if(error_ != hipSuccess)                                                                   \
        {                                                                                          \
            fprintf(stderr,                                                                        \
                    "%s:%d :: HIP error %i: %s\n",                                                 \
                    __FILE__,                                                                      \
                    __LINE__,                                                                      \
                    (int) error_,                                                                  \
                    hipGetErrorString(error_));                                                    \
            return 1;                                                                              \
        }                                                                                          \
    }

// External init() function from the tool library
extern "C" void
late_start_init();

int
main(int argc, char** argv)
{
    (void) argc;
    (void) argv;

    printf("=== Late-Start Tracing Test ===\n\n");

    // Phase 1: Initialize HIP BEFORE calling init() (not traced)
    printf("Phase 1: Pre-init HIP calls (should NOT be traced)...\n");

    auto pre_init_range = roctxRangeStart("pre-init-phase");

    int device_count = 0;
    HIP_API_CALL(hipGetDeviceCount(&device_count));
    printf("  hipGetDeviceCount: found %d device(s)\n", device_count);

    if(device_count == 0)
    {
        printf("  No HIP devices found - test skipped\n");
        roctxRangeStop(pre_init_range);
        return 0;
    }

    HIP_API_CALL(hipSetDevice(0));
    printf("  hipSetDevice(0) completed\n");

    // Get device properties (additional HIP calls)
    hipDeviceProp_t prop;
    HIP_API_CALL(hipGetDeviceProperties(&prop, 0));
    printf("  Device: %s\n", prop.name);

    roctxRangeStop(pre_init_range);

    // Phase 2: Call init() to trigger late-start profiling
    printf("\nPhase 2: Calling init() to trigger late-start...\n");
    late_start_init();
    printf("  Late-start profiling activated\n");

    // Phase 3: Post-init HIP calls AFTER late-start (SHOULD be traced)
    printf("\nPhase 3: Post-init HIP calls (SHOULD be traced)...\n");

    auto post_init_range = roctxRangeStart("post-init-phase");

    // Make several HIP API calls that should be traced
    void* ptr = nullptr;
    HIP_API_CALL(hipMalloc(&ptr, 1024));
    printf("  hipMalloc completed\n");

    HIP_API_CALL(hipMemset(ptr, 0, 1024));
    printf("  hipMemset completed\n");

    HIP_API_CALL(hipDeviceSynchronize());
    printf("  hipDeviceSynchronize completed\n");

    HIP_API_CALL(hipFree(ptr));
    printf("  hipFree completed\n");

    // Additional ROCTx ranges to be traced
    auto nested_range = roctxRangeStart("nested-operation");
    HIP_API_CALL(hipGetLastError());
    roctxRangeStop(nested_range);

    roctxRangeStop(post_init_range);

    printf("\n=== Test Complete ===\n");
    printf("Check the JSON output file for validation\n");

    return 0;
}
