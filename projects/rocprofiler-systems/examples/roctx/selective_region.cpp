// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Demonstrates the ROCPROFSYS_TRACE_REGION selective region tracing feature.
// Uses roctxRangeStartA/roctxRangeStop (process-wide markers) for region filtering.
//
// Code flow:
//   CodeBlock_A                                 (outside any region)
//   roctxRangeStartA("Region 1")
//     CodeBlock_B                               (inside Region 1)
//     roctxRangeStartA("Region 2")
//       CodeBlock_C                             (inside Region 1 + Region 2)
//     roctxRangeStop("Region 2")
//     CodeBlock_D                               (inside Region 1)
//   roctxRangeStop("Region 1")
//   roctxRangeStartA("Region 3")
//     CodeBlock_E                               (inside Region 3)
//   roctxRangeStop("Region 3")
//   roctxRangeStartA("Region 1")
//     CodeBlock_F                               (inside Region 1 again)
//   roctxRangeStop("Region 1")
//   CodeBlock_G                                 (outside any region)
//
// Run without filter (traces everything):
//   rocprof-sys -- ./selective_region
//
// Run with filter (only traces inside "Region 1"):
//   ROCPROFSYS_TRACE_REGION="Region 1" rocprof-sys -- ./selective_region
//
// Expected with filter: profiling data recorded for {CodeBlock_B, CodeBlock_C,
//                        CodeBlock_D, CodeBlock_F}

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
CodeBlock_B(float* data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) data[idx] += 20.0f;
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

__global__ void
CodeBlock_E(float* data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) data[idx] += 50.0f;
}

__global__ void
CodeBlock_F(float* data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) data[idx] += 60.0f;
}

__global__ void
CodeBlock_G(float* data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < n) data[idx] += 70.0f;
}

int
main()
{
    const int    N    = 256;
    const size_t size = N * sizeof(float);

    float* d_data;
    HIP_CHECK(hipMalloc(&d_data, size));
    HIP_CHECK(hipMemset(d_data, 0, size));

    // Outside any region
    CodeBlock_A<<<(N + 255) / 256, 256>>>(d_data, N);
    HIP_CHECK(hipDeviceSynchronize());

    // Region 1
    roctx_range_id_t region1_id = roctxRangeStartA("Region 1");

    CodeBlock_B<<<(N + 255) / 256, 256>>>(d_data, N);
    HIP_CHECK(hipDeviceSynchronize());

    // Nested Region 2 (does not match filter)
    roctx_range_id_t region2_id = roctxRangeStartA("Region 2");

    CodeBlock_C<<<(N + 255) / 256, 256>>>(d_data, N);
    HIP_CHECK(hipDeviceSynchronize());

    roctxRangeStop(region2_id);

    CodeBlock_D<<<(N + 255) / 256, 256>>>(d_data, N);
    HIP_CHECK(hipDeviceSynchronize());

    roctxRangeStop(region1_id);

    // Region 3 (does not match filter)
    roctx_range_id_t region3_id = roctxRangeStartA("Region 3");

    CodeBlock_E<<<(N + 255) / 256, 256>>>(d_data, N);
    HIP_CHECK(hipDeviceSynchronize());

    roctxRangeStop(region3_id);

    // Region 1 again
    roctx_range_id_t region1b_id = roctxRangeStartA("Region 1");

    CodeBlock_F<<<(N + 255) / 256, 256>>>(d_data, N);
    HIP_CHECK(hipDeviceSynchronize());

    roctxRangeStop(region1b_id);

    // Outside any region
    CodeBlock_G<<<(N + 255) / 256, 256>>>(d_data, N);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(d_data));
    return 0;
}
