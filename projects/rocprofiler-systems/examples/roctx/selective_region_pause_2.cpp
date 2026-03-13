// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Demonstrates interleaving of Pause/Resume with Selective Region Tracing.
// Scenario: Pause occurs OUTSIDE the target region (before it starts).
//
// The pause happens outside any target region, so it is not valid in the
// context of region filtering. When the target region starts, profiling
// begins normally. The resume inside the region is a no-op since there
// was no valid pause to undo.
//
// Code flow:
//   roctxProfilerPause                          (outside region — pause not valid)
//   CodeBlock_Z                                 (outside region — not profiled)
//   roctxRangeStartA("Region 1")               (enter target region — profiling starts)
//     CodeBlock_A                               (profiled)
//     CodeBlock_B                               (profiled)
//     roctxProfilerResume                       (no valid pause to undo — no-op)
//     as                               (profiled)
//   roctxRangeStop("Region 1")                  (region ends — profiling stops)
//   CodeBlock_D                                 (outside region — not profiled)
//
// Run with filter:
//   ROCPROFSYS_TRACE_REGION="Region 1" rocprof-sys -- ./selective_region_pause_2
//
// Expected: profiling data recorded for {CodeBlock_A, CodeBlock_B, CodeBlock_C}

#include <cstdio>

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
CodeBlock_Z(float* data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) data[idx] += 10.0f;
}

__global__ void
CodeBlock_A(float* data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) data[idx] += 20.0f;
}

__global__ void
CodeBlock_B(float* data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) data[idx] += 30.0f;
}

__global__ void
CodeBlock_C(float* data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) data[idx] += 40.0f;
}

__global__ void
CodeBlock_D(float* data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) data[idx] += 50.0f;
}

int
main()
{
    const int    N    = 256;
    const size_t size = N * sizeof(float);

    float* d_data;
    HIP_CHECK(hipMalloc(&d_data, size));
    HIP_CHECK(hipMemset(d_data, 0, size));

    roctx_thread_id_t tid{};
    roctxGetThreadId(&tid);

    roctxProfilerPause(tid);

    // Outside region
    CodeBlock_Z<<<(N + 255) / 256, 256>>>(d_data, N);
    HIP_CHECK(hipDeviceSynchronize());

    // Region 1 — profiling starts despite prior pause (pause was outside region)
    roctx_range_id_t region1_id = roctxRangeStartA("Region 1");

    CodeBlock_A<<<(N + 255) / 256, 256>>>(d_data, N);
    HIP_CHECK(hipDeviceSynchronize());

    CodeBlock_B<<<(N + 255) / 256, 256>>>(d_data, N);
    HIP_CHECK(hipDeviceSynchronize());

    // Resume (no valid pause to undo — no-op)
    roctxProfilerResume(tid);

    CodeBlock_C<<<(N + 255) / 256, 256>>>(d_data, N);
    HIP_CHECK(hipDeviceSynchronize());

    roctxRangeStop(region1_id);

    // Outside region
    CodeBlock_D<<<(N + 255) / 256, 256>>>(d_data, N);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(d_data));
    return 0;
}
