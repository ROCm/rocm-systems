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

// Standalone HIP application that launches the VecAdd and SAXPY kernels. Intended to be run under
// the profiler as an integration test for kernel replay: the profiler re-executes each dispatch
// and restores device memory between passes, so this application observes correct results exactly
// as if it ran without profiling.

#include <hip/hip_runtime.h>

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
// Each kernel grid-stride loops so all n elements are processed regardless of the (fixed) launch
// size; the fixed dims keep the dispatch's Grid_Size / Workgroup_Size deterministic for validation.

// VecAdd: simple element-wise add into a separate output buffer.
__global__ void
vecAdd(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n)
{
    const int stride = blockDim.x * gridDim.x;
    for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += stride)
        c[i] = a[i] + b[i];
}

// SAXPY: in-place read-write kernel (y is both read and written).
__global__ void
saxpy(float alpha, const float* __restrict__ x, float* __restrict__ y, int n)
{
    const int stride = blockDim.x * gridDim.x;
    for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += stride)
        y[i] = alpha * x[i] + y[i];
}

bool
approx_equal(float got, float want)
{
    return std::fabs(got - want) <= 1e-2f * std::fabs(want) + 1e-2f;
}

int
run_vecadd(int n, int iters)
{
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    std::vector<float> h_a(n), h_b(n), h_c(n);
    for(int i = 0; i < n; ++i)
    {
        h_a[i] = static_cast<float>(i % 1000);
        h_b[i] = static_cast<float>((i % 1000) * 2);
    }

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    HIP_CHECK(hipMalloc(&d_a, bytes));
    HIP_CHECK(hipMalloc(&d_b, bytes));
    HIP_CHECK(hipMalloc(&d_c, bytes));
    HIP_CHECK(hipMemcpy(d_a, h_a.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b.data(), bytes, hipMemcpyHostToDevice));

    for(int iter = 0; iter < iters; ++iter)
    {
        vecAdd<<<1024, 1024>>>(d_a, d_b, d_c, n);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        HIP_CHECK(hipMemcpy(h_c.data(), d_c, bytes, hipMemcpyDeviceToHost));
        for(int i = 0; i < n && i < 1024; ++i)
        {
            if(!approx_equal(h_c[i], h_a[i] + h_b[i]))
            {
                fprintf(stderr,
                        "vecAdd mismatch iter %d elem %d: %f != %f\n",
                        iter,
                        i,
                        h_c[i],
                        h_a[i] + h_b[i]);
                return EXIT_FAILURE;
            }
        }
    }

    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));
    printf("vecAdd: n=%d iters=%d OK\n", n, iters);
    return EXIT_SUCCESS;
}

int
run_saxpy(int n, int iters)
{
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    const float  alpha = 2.0f;

    std::vector<float> h_x(n), h_y(n), expected(n);
    for(int i = 0; i < n; ++i)
    {
        h_x[i]      = static_cast<float>(i % 1000) * 0.001f;
        h_y[i]      = static_cast<float>((n - i) % 1000) * 0.001f;
        expected[i] = h_y[i];
    }

    float *d_x = nullptr, *d_y = nullptr;
    HIP_CHECK(hipMalloc(&d_x, bytes));
    HIP_CHECK(hipMalloc(&d_y, bytes));
    HIP_CHECK(hipMemcpy(d_x, h_x.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_y, h_y.data(), bytes, hipMemcpyHostToDevice));

    for(int iter = 0; iter < iters; ++iter)
    {
        saxpy<<<512, 512>>>(alpha, d_x, d_y, n);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Each (app-observed) launch accumulates once into y.
        for(int i = 0; i < n; ++i)
            expected[i] = alpha * h_x[i] + expected[i];

        std::vector<float> h_result(n);
        HIP_CHECK(hipMemcpy(h_result.data(), d_y, bytes, hipMemcpyDeviceToHost));
        for(int i = 0; i < n && i < 1024; ++i)
        {
            if(!approx_equal(h_result[i], expected[i]))
            {
                fprintf(stderr,
                        "saxpy mismatch iter %d elem %d: %f != %f\n",
                        iter,
                        i,
                        h_result[i],
                        expected[i]);
                return EXIT_FAILURE;
            }
        }
    }

    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));
    printf("saxpy: n=%d iters=%d OK\n", n, iters);
    return EXIT_SUCCESS;
}
}  // namespace

int
main(int argc, char** argv)
{
    const int n     = (argc > 1) ? atoi(argv[1]) : (1 << 20);
    const int iters = (argc > 2) ? atoi(argv[2]) : 1;

    if(run_vecadd(n, iters) != EXIT_SUCCESS) return EXIT_FAILURE;
    if(run_saxpy(n, iters) != EXIT_SUCCESS) return EXIT_FAILURE;

    printf("kernel-replay: all kernels completed\n");
    return EXIT_SUCCESS;
}
