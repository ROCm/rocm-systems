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

// Standalone HIP application that launches three distinct kernels (VecAdd, SAXPY, VecScale).
// Intended to be run under the profiler as an integration test for kernel replay: the profiler
// re-executes each dispatch once per counter batch and restores device memory between passes, so
// this application observes correct results exactly as if it ran without profiling.

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

// Module-scope __device__ counter, bumped once per saxpy execution by a single thread. It lives in
// the executable's data segment (not a tracked hipMalloc allocation), so kernel replay must restore
// it between passes: a replayed dispatch then nets exactly one bump (as if run once), while a
// broken module-variable restore accumulates one bump per pass. Legit accumulation ACROSS host
// launches is preserved -- only per-pass accumulation is the bug. Exercised by run_inplace_saxpy().
__device__ int g_saxpy_calls = 0;

// SAXPY: in-place read-write kernel (y is both read and written).
__global__ void
saxpy(float alpha, const float* __restrict__ x, float* __restrict__ y, int n)
{
    const int stride = blockDim.x * gridDim.x;
    for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += stride)
        y[i] = alpha * x[i] + y[i];

    if(blockIdx.x == 0 && threadIdx.x == 0) g_saxpy_calls += 1;
}

// VecScale: in-place scalar multiply (a third distinct kernel).
__global__ void
vecScale(float* __restrict__ a, float s, int n)
{
    const int stride = blockDim.x * gridDim.x;
    for(int i = blockDim.x * blockIdx.x + threadIdx.x; i < n; i += stride)
        a[i] = a[i] * s;
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

int
run_vecscale(int n, int iters)
{
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);
    const float  scale = 0.5f;

    std::vector<float> h_a(n), expected(n);
    for(int i = 0; i < n; ++i)
    {
        h_a[i]      = static_cast<float>((i % 1000) + 1);
        expected[i] = h_a[i];
    }

    float* d_a = nullptr;
    HIP_CHECK(hipMalloc(&d_a, bytes));
    HIP_CHECK(hipMemcpy(d_a, h_a.data(), bytes, hipMemcpyHostToDevice));

    for(int iter = 0; iter < iters; ++iter)
    {
        vecScale<<<256, 256>>>(d_a, scale, n);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Each (app-observed) launch scales a once.
        for(int i = 0; i < n; ++i)
            expected[i] = expected[i] * scale;

        std::vector<float> h_result(n);
        HIP_CHECK(hipMemcpy(h_result.data(), d_a, bytes, hipMemcpyDeviceToHost));
        for(int i = 0; i < n && i < 1024; ++i)
        {
            if(!approx_equal(h_result[i], expected[i]))
            {
                fprintf(stderr,
                        "vecScale mismatch iter %d elem %d: %f != %f\n",
                        iter,
                        i,
                        h_result[i],
                        expected[i]);
                return EXIT_FAILURE;
            }
        }
    }

    HIP_CHECK(hipFree(d_a));
    printf("vecScale: n=%d iters=%d OK\n", n, iters);
    return EXIT_SUCCESS;
}

// Restore-correctness mode (buffer AND module variable). Launch the in-place saxpy dispatch
// (y = a*x + y with x=1, a=2, so each host-observed launch adds `step` to y) kLaunches times; the
// kernel also bumps the __device__ counter g_saxpy_calls once per execution. Under
// --replay-mode kernel --kernel-replay-beta-enabled each host-observed dispatch is re-executed N times (N counter
// groups), with device memory AND module variables restored between passes, so BOTH quantities must
// scale with the number of *launches*, never launches*passes:
//   restore WORKS  -> y == y0 + kLaunches*step   AND g_saxpy_calls == kLaunches      (e.g. 106 / 3)
//   restore BROKEN -> y == y0 + kLaunches*N*step AND g_saxpy_calls == kLaunches*N    (e.g. 130 /
//   15)
// Legit accumulation ACROSS launches is expected (kLaunches, not 1); only per-pass accumulation is
// a bug. Checking the tracked buffer AND the untracked __device__ global (snap()'s
// HSA_SYMBOL_KIND_VARIABLE path) makes this a stronger end-to-end proof than a single-launch check.
int
run_inplace_saxpy(int n)
{
    constexpr int kLaunches = 3;
    const size_t  bytes     = static_cast<size_t>(n) * sizeof(float);
    const float   alpha     = 2.0f;
    const float   x_val     = 1.0f;
    const float   y0        = 100.0f;
    const float   step      = alpha * x_val;  // per-application delta to y (2)

    std::vector<float> h_x(n, x_val);
    std::vector<float> h_y(n, y0);

    float *d_x = nullptr, *d_y = nullptr;
    HIP_CHECK(hipMalloc(&d_x, bytes));
    HIP_CHECK(hipMalloc(&d_y, bytes));
    HIP_CHECK(hipMemcpy(d_x, h_x.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_y, h_y.data(), bytes, hipMemcpyHostToDevice));

    // Reset the module-scope counter so earlier saxpy launches (run_saxpy) don't skew it.
    const int zero = 0;
    HIP_CHECK(hipMemcpyToSymbol(HIP_SYMBOL(g_saxpy_calls), &zero, sizeof(int)));

    bool ok = true;
    for(int i = 0; i < kLaunches; ++i)
    {
        saxpy<<<512, 512>>>(alpha, d_x, d_y, n);  // host-observed launch i
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Both the buffer and the __device__ counter grow by one application per launch,
        // never per replay pass, so a broken restore is caught at the first launch (values == N
        // instead of 1) rather than only in aggregate.
        const int   expected_calls = i + 1;
        const float expected_y     = y0 + static_cast<float>(expected_calls) * step;

        HIP_CHECK(hipMemcpy(h_y.data(), d_y, bytes, hipMemcpyDeviceToHost));
        int calls = -1;
        HIP_CHECK(hipMemcpyFromSymbol(&calls, HIP_SYMBOL(g_saxpy_calls), sizeof(int)));

        int nonuniform = 0;
        for(int e = 0; e < n; ++e)
            if(!approx_equal(h_y[e], h_y[0])) ++nonuniform;

        const float buf_apps = (h_y[0] - y0) / step;
        printf("[rstest] launch=%d/%d y[0]=%.3f expected=%.3f buffer_applications=%.3f "
               "device_counter=%d nonuniform=%d\n",
               expected_calls,
               kLaunches,
               h_y[0],
               expected_y,
               buf_apps,
               calls,
               nonuniform);

        if(!approx_equal(h_y[0], expected_y) || nonuniform != 0 || calls != expected_calls)
        {
            printf("[rstest] FAIL after launch %d: buffer_applications=%.3f device_counter=%d "
                   "(expected %d each) -> restore did NOT revert between replay passes\n",
                   expected_calls,
                   buf_apps,
                   calls,
                   expected_calls);
            ok = false;
            break;
        }
    }

    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));

    if(ok)
    {
        printf("[rstest] PASS: buffer and __device__ global each grew exactly once per launch over "
               "%d launches (restore reverted inputs between passes)\n",
               kLaunches);
        return EXIT_SUCCESS;
    }
    return EXIT_FAILURE;
}
}  // namespace

int
main(int argc, char** argv)
{
    const int n     = (argc > 1) ? atoi(argv[1]) : (1 << 20);
    const int iters = (argc > 2) ? atoi(argv[2]) : 1;

    if(run_vecadd(n, iters) != EXIT_SUCCESS) return EXIT_FAILURE;
    if(run_saxpy(n, iters) != EXIT_SUCCESS) return EXIT_FAILURE;
    if(run_vecscale(n, iters) != EXIT_SUCCESS) return EXIT_FAILURE;
    if(run_inplace_saxpy(n) != EXIT_SUCCESS) return EXIT_FAILURE;

    printf("kernel-replay: all kernels completed\n");
    return EXIT_SUCCESS;
}
