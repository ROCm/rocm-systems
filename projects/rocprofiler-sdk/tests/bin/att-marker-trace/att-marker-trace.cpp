// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rocprofiler-sdk-roctx/roctx.h>

#include <hip/hip_runtime.h>

#include <stdio.h>
#include <stdlib.h>

#define WIDTH 1024
#define NUM   (WIDTH * WIDTH)

#define THREADS_PER_BLOCK_X 4
#define THREADS_PER_BLOCK_Y 4

template <typename T>
void
check(T result, char const* const func, const char* const file, int const line)
{
    if(result)
    {
        fprintf(stderr,
                "Hip error at %s:%d code=%d(%s) \"%s\" \n",
                file,
                line,
                static_cast<unsigned int>(result),
                hipGetErrorName(result),
                func);
        exit(EXIT_FAILURE);
    }
}
#define checkHipErrors(val) check((val), #val, __FILE__, __LINE__)

// Kernel A: launched BEFORE roctxProfilerResume — should NOT appear in ATT output
__global__ void
before_trace_kernel(float* out, const float* in, const int width)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    out[y * width + x] = in[x * width + y] + 1.0f;
}

// Kernel B: launched AFTER roctxProfilerResume — should appear in ATT output
__global__ void
traced_kernel_first(float* out, const float* in, const int width)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    out[y * width + x] = in[x * width + y] * 2.0f;
}

// Kernel C: launched AFTER roctxProfilerResume — should appear in ATT output
__global__ void
traced_kernel_second(float* out, const float* in, const int width)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    out[y * width + x] = in[x * width + y] * 3.0f;
}

// Kernel D: launched AFTER roctxProfilerPause — should NOT appear in ATT output
__global__ void
after_trace_kernel(float* out, const float* in, const int width)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    out[y * width + x] = in[x * width + y] - 1.0f;
}

int
main()
{
    float* gpuIn  = nullptr;
    float* gpuOut = nullptr;
    float* hostIn = nullptr;

    hostIn = (float*) malloc(NUM * sizeof(float));
    for(int i = 0; i < NUM; i++)
        hostIn[i] = (float) i * 10.0f;

    checkHipErrors(hipMalloc(&gpuIn, NUM * sizeof(float)));
    checkHipErrors(hipMalloc(&gpuOut, NUM * sizeof(float)));
    checkHipErrors(hipMemcpy(gpuIn, hostIn, NUM * sizeof(float), hipMemcpyHostToDevice));

    dim3 grid(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y);
    dim3 block(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);

    // Kernel A: before tracing is enabled (all contexts start stopped
    // because --att-marker-trace implies --selected-regions)
    hipLaunchKernelGGL(before_trace_kernel, grid, block, 0, 0, gpuOut, gpuIn, WIDTH);
    checkHipErrors(hipDeviceSynchronize());

    // Start thread trace collection
    roctxProfilerResume(0);

    // Kernel B: traced
    hipLaunchKernelGGL(traced_kernel_first, grid, block, 0, 0, gpuOut, gpuIn, WIDTH);
    // Kernel C: traced
    hipLaunchKernelGGL(traced_kernel_second, grid, block, 0, 0, gpuOut, gpuIn, WIDTH);

    // Synchronize before stopping to ensure GPU work completes
    checkHipErrors(hipDeviceSynchronize());

    // Stop thread trace collection
    roctxProfilerPause(0);

    // Kernel D: after tracing is disabled
    hipLaunchKernelGGL(after_trace_kernel, grid, block, 0, 0, gpuOut, gpuIn, WIDTH);
    checkHipErrors(hipDeviceSynchronize());

    checkHipErrors(hipFree(gpuIn));
    checkHipErrors(hipFree(gpuOut));
    free(hostIn);

    return 0;
}
