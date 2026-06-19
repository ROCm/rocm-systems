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

// Simple kernel-replay benchmark for the memory_tracker + memory_snapshot save/restore path.
//
// Scope is deliberately narrow: directly-allocated device memory only (hipMalloc), no unified /
// managed memory. Each test snapshots its device buffers, runs the kernel several times restoring
// memory between passes, checks correctness, and reports snap/restore timing.

#include "lib/rocprofiler-sdk/hsa/memory_snapshot.hpp"
#include "lib/rocprofiler-sdk/hsa/memory_tracker.hpp"

#include <hip/hip_runtime.h>
#include <gtest/gtest.h>

#include <chrono>
#include <cstdio>
#include <vector>

#include "kernels.hip"

#define HIP_CHECK(call)                                                                            \
    do                                                                                             \
    {                                                                                              \
        hipError_t err = (call);                                                                   \
        ASSERT_EQ(err, hipSuccess) << "HIP error: " << hipGetErrorString(err);                     \
    } while(0)

namespace
{
using hclock = std::chrono::high_resolution_clock;

double
elapsed_ms(hclock::time_point start, hclock::time_point end)
{
    return std::chrono::duration<double, std::milli>(end - start).count();
}
}  // namespace

// VecAdd: baseline replay. Output buffer is rewritten each pass; restore reverts inputs so every
// pass produces the same result.
TEST(ReplayBenchmark, VecAdd)
{
    constexpr int    N     = 1 << 20;
    constexpr size_t bytes = N * sizeof(float);

    std::vector<float> h_a(N), h_b(N), h_c(N);
    for(int i = 0; i < N; ++i)
    {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i * 2);
    }

    float *d_a, *d_b, *d_c;
    HIP_CHECK(hipMalloc(&d_a, bytes));
    HIP_CHECK(hipMalloc(&d_b, bytes));
    HIP_CHECK(hipMalloc(&d_c, bytes));

    namespace mt = rocprofiler::hsa::memory_tracker;
    mt::set_tracking_enabled(true);
    mt::record_alloc(d_a, bytes);
    mt::record_alloc(d_b, bytes);
    mt::record_alloc(d_c, bytes);

    HIP_CHECK(hipMemcpy(d_a, h_a.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b.data(), bytes, hipMemcpyHostToDevice));

    rocprofiler::hsa::memory_snapshot::Snapshot snapshot;
    auto   t0      = hclock::now();
    size_t snapped = snapshot.snap();
    auto   t1      = hclock::now();
    EXPECT_GE(snapped, 3u);
    fprintf(stderr, "  VecAdd snap: %.2f ms (%zu regions)\n", elapsed_ms(t0, t1), snapped);

    constexpr int kPasses = 3;
    for(int pass = 0; pass < kPasses; ++pass)
    {
        int blocks = (N + 255) / 256;
        vecAdd<<<blocks, 256>>>(d_a, d_b, d_c, N);
        HIP_CHECK(hipDeviceSynchronize());

        HIP_CHECK(hipMemcpy(h_c.data(), d_c, bytes, hipMemcpyDeviceToHost));
        for(int i = 0; i < 100; ++i)
            ASSERT_FLOAT_EQ(h_c[i], h_a[i] + h_b[i]) << "pass " << pass << " elem " << i;

        if(pass < kPasses - 1)
        {
            auto   r0    = hclock::now();
            size_t regns = snapshot.restore();
            auto   r1    = hclock::now();
            fprintf(stderr,
                    "  VecAdd restore pass %d: %.2f ms (%zu regions)\n",
                    pass,
                    elapsed_ms(r0, r1),
                    regns);
        }
    }

    mt::record_free(d_a);
    mt::record_free(d_b);
    mt::record_free(d_c);
    mt::set_tracking_enabled(false);

    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));
}

// SAXPY: in-place overwrite of y. Without a correct restore between passes the result would drift;
// this validates that snapshot save/restore fully reverts a read-write buffer.
TEST(ReplayBenchmark, Saxpy)
{
    constexpr int    N     = 1 << 20;
    constexpr size_t bytes = N * sizeof(float);
    constexpr float  alpha = 2.0f;

    std::vector<float> h_x(N), h_y(N), h_y_orig(N);
    for(int i = 0; i < N; ++i)
    {
        h_x[i] = static_cast<float>(i) * 0.001f;
        h_y[i] = static_cast<float>(N - i) * 0.001f;
    }
    h_y_orig = h_y;

    float *d_x, *d_y;
    HIP_CHECK(hipMalloc(&d_x, bytes));
    HIP_CHECK(hipMalloc(&d_y, bytes));

    namespace mt = rocprofiler::hsa::memory_tracker;
    mt::set_tracking_enabled(true);
    mt::record_alloc(d_x, bytes);
    mt::record_alloc(d_y, bytes);

    HIP_CHECK(hipMemcpy(d_x, h_x.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_y, h_y.data(), bytes, hipMemcpyHostToDevice));

    rocprofiler::hsa::memory_snapshot::Snapshot snapshot;
    auto   t0      = hclock::now();
    size_t snapped = snapshot.snap();
    auto   t1      = hclock::now();
    fprintf(stderr, "  SAXPY snap: %.2f ms (%zu regions)\n", elapsed_ms(t0, t1), snapped);

    constexpr int kPasses = 3;
    for(int pass = 0; pass < kPasses; ++pass)
    {
        int blocks = (N + 255) / 256;
        saxpy<<<blocks, 256>>>(alpha, d_x, d_y, N);
        HIP_CHECK(hipDeviceSynchronize());

        std::vector<float> h_result(N);
        HIP_CHECK(hipMemcpy(h_result.data(), d_y, bytes, hipMemcpyDeviceToHost));
        for(int i = 0; i < 100; ++i)
            ASSERT_NEAR(h_result[i], alpha * h_x[i] + h_y_orig[i], 1e-3f)
                << "pass " << pass << " elem " << i;

        if(pass < kPasses - 1)
        {
            auto   r0    = hclock::now();
            size_t regns = snapshot.restore();
            auto   r1    = hclock::now();
            fprintf(stderr,
                    "  SAXPY restore pass %d: %.2f ms (%zu regions)\n",
                    pass,
                    elapsed_ms(r0, r1),
                    regns);
        }
    }

    mt::record_free(d_x);
    mt::record_free(d_y);
    mt::set_tracking_enabled(false);

    HIP_CHECK(hipFree(d_x));
    HIP_CHECK(hipFree(d_y));
}
