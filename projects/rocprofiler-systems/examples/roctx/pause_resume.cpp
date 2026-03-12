// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Demonstrates the roctx Pause/Resume feature for selective profiling.
//
// Code flow:
//   CodeBlock_Z          (profiled)
//   CodeBlock_A          (profiled)
//   roctxProfilerPause
//   CodeBlock_B          (NOT profiled — paused)
//   roctxProfilerResume
//   CodeBlock_C          (profiled)
//   CodeBlock_D          (profiled)
//
// Run with profiling:
//   rocprof-sys -- ./pause_resume
//
// Expected: profiling data recorded for {CodeBlock_Z, CodeBlock_A, CodeBlock_C,
// CodeBlock_D}

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

    CodeBlock_Z<<<(N + 255) / 256, 256>>>(d_data, N);
    HIP_CHECK(hipDeviceSynchronize());

    CodeBlock_A<<<(N + 255) / 256, 256>>>(d_data, N);
    HIP_CHECK(hipDeviceSynchronize());

    roctxProfilerPause(tid);

    CodeBlock_B<<<(N + 255) / 256, 256>>>(d_data, N);
    HIP_CHECK(hipDeviceSynchronize());

    roctxProfilerResume(tid);

    CodeBlock_C<<<(N + 255) / 256, 256>>>(d_data, N);
    HIP_CHECK(hipDeviceSynchronize());

    CodeBlock_D<<<(N + 255) / 256, 256>>>(d_data, N);
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipFree(d_data));
    return 0;
}
