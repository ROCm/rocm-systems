// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Demonstrates interleaving of Pause/Resume with Selective Region Tracing.
// Scenario: Region ends while profiling is paused, resume occurs outside region.
//
// The pause happens inside the target region and is valid. Then the region
// ends while still paused (a warning is logged). After the region ends,
// the resume occurs outside the region and is ignored.
//
// Code flow:
//   roctxRangeStartA("Region 1")               (enter target region — profiling starts)
//     CodeBlock_A                               (profiled)
//     roctxProfilerPause                        (valid pause inside region — profiling
//     stops) CodeBlock_C                               (paused — not profiled)
//   roctxRangeStop("Region 1")                  (region ends while paused — warning
//   logged) CodeBlock_D                                 (outside region — not profiled)
//   roctxProfilerResume                         (outside region — ignored)
//
// Run with filter:
//   ROCPROFSYS_TRACE_REGION="Region 1" rocprof-sys -- ./selective_region_pause_3
//
// Expected: profiling data recorded for {CodeBlock_A}

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
CodeBlock_A(float* data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) data[idx] += 10.0f;
}

__global__ void
CodeBlock_C(float* data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) data[idx] += 30.0f;
}

__global__ void
CodeBlock_D(float* data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) data[idx] += 40.0f;
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

    // Region 1
    roctx_range_id_t region1_id = roctxRangeStartA("Region 1");

    CodeBlock_A<<<(N + 255) / 256, 256>>>(d_data, N);
    HIP_CHECK(hipDeviceSynchronize());

    // Pause inside region (valid)
    roctxProfilerPause(tid);

    CodeBlock_C<<<(N + 255) / 256, 256>>>(d_data, N);
    HIP_CHECK(hipDeviceSynchronize());

    // Region ends while paused — warning logged
    roctxRangeStop(region1_id);

    // Outside region
    CodeBlock_D<<<(N + 255) / 256, 256>>>(d_data, N);
    HIP_CHECK(hipDeviceSynchronize());

    // Resume outside region — ignored
    roctxProfilerResume(tid);

    HIP_CHECK(hipFree(d_data));
    return 0;
}
