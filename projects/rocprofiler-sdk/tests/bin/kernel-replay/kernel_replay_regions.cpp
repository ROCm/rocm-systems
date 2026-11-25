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

// Standalone HIP application for kernel replay under --selected-regions. It launches saxpy
// <before> times while the profiler is paused, <inside> times between roctxProfilerResume and
// roctxProfilerPause, and <after> times once paused again. Only the launches inside the region are
// profiled, so only they count toward --kernel-iteration-range. Every launch adds alpha * x to y
// exactly once however many replay passes it gets, which the final check confirms.

#include <hip/hip_runtime.h>
#include <rocprofiler-sdk-roctx/roctx.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define HIP_CHECK(call)                                                                            \
    do                                                                                             \
    {                                                                                              \
        hipError_t _err = (call);                                                                  \
        if(_err != hipSuccess)                                                                     \
        {                                                                                          \
            fprintf(                                                                               \
                stderr, "HIP error '%s' at %s:%d\n", hipGetErrorString(_err), __FILE__, __LINE__); \
            return EXIT_FAILURE;                                                                   \
        }                                                                                          \
    } while(0)

namespace
{
__global__ void
saxpy(float alpha, const float* x, float* y, int n)
{
    const int stride = blockDim.x * gridDim.x;
    for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += stride)
        y[i] = alpha * x[i] + y[i];
}

int
launch(int count, float alpha, const float* x, float* y, int n)
{
    for(int i = 0; i < count; ++i)
    {
        saxpy<<<512, 512>>>(alpha, x, y, n);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());
    }
    return EXIT_SUCCESS;
}
}  // namespace

int
main(int argc, char** argv)
{
    const int n      = (argc > 1) ? atoi(argv[1]) : (1 << 16);
    const int before = (argc > 2) ? atoi(argv[2]) : 3;
    const int inside = (argc > 3) ? atoi(argv[3]) : 5;
    const int after  = (argc > 4) ? atoi(argv[4]) : 2;

    const float  alpha = 2.0f;
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    auto         h_x   = std::vector<float>(n, 1.0f);
    auto         h_y   = std::vector<float>(n, 0.0f);

    float* d_x = nullptr;
    float* d_y = nullptr;
    HIP_CHECK(hipMalloc(&d_x, bytes));
    HIP_CHECK(hipMalloc(&d_y, bytes));
    HIP_CHECK(hipMemcpy(d_x, h_x.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_y, h_y.data(), bytes, hipMemcpyHostToDevice));

    if(launch(before, alpha, d_x, d_y, n) != EXIT_SUCCESS) return EXIT_FAILURE;
    roctxProfilerResume(0);
    if(launch(inside, alpha, d_x, d_y, n) != EXIT_SUCCESS) return EXIT_FAILURE;
    roctxProfilerPause(0);
    if(launch(after, alpha, d_x, d_y, n) != EXIT_SUCCESS) return EXIT_FAILURE;

    HIP_CHECK(hipMemcpy(h_y.data(), d_y, bytes, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));

    const float expected   = alpha * static_cast<float>(before + inside + after);
    int         mismatches = 0;
    for(float value : h_y)
        if(std::fabs(value - expected) > 1e-3f) ++mismatches;
    if(mismatches != 0)
    {
        fprintf(stderr,
                "kernel-replay-regions: %d of %d elements differ from %.1f (y[0] = %.1f)\n",
                mismatches,
                n,
                expected,
                h_y[0]);
        return EXIT_FAILURE;
    }

    printf("kernel-replay-regions: %d launches before, %d inside and %d after the region OK\n",
           before,
           inside,
           after);
    return EXIT_SUCCESS;
}
