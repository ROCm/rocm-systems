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

#include <rocprofiler-sdk-roctx/roctx.h>

// hip header file
#include <hip/hip_runtime.h>

#include <stdio.h>

#define WIDTH 1024

#define NUM (WIDTH * WIDTH)

#define THREADS_PER_BLOCK_X 4
#define THREADS_PER_BLOCK_Y 4

#define ITER_NUM   16 * 1024
#define BLOCK_SIZE 1024

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

__global__ void
kernel_mult(const float varA, const float varB, float* result)
{
    *result = varA * varB;
}

__global__ void
kernel_add(const float varA, const float varB, float* result)
{
    *result = varA + varB;
}

__global__ void
target_kernel(float* out, float* in, const int width)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    out[y * width + x] = in[x * width + y];
}

__global__ void
nested_kernel(float* out, float* in, const int width)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    out[y * width + x] = in[x * width + y];
}

#define ASM_LINE asm volatile("v_mov_b32 %0 %1\n" : "=v"(a) : "s"(c))

#define REPEAT_2(X)                                                                                \
    X;                                                                                             \
    X
#define REPEAT_4(X)                                                                                \
    REPEAT_2(X);                                                                                   \
    REPEAT_2(X)
#define REPEAT_8(X)                                                                                \
    REPEAT_4(X);                                                                                   \
    REPEAT_4(X)
#define REPEAT_16(X)                                                                               \
    REPEAT_8(X);                                                                                   \
    REPEAT_8(X)
#define REPEAT_32(X)                                                                               \
    REPEAT_16(X);                                                                                  \
    REPEAT_16(X)
#define REPEAT_64(X)                                                                               \
    REPEAT_32(X);                                                                                  \
    REPEAT_32(X)
#define REPEAT_100(X)                                                                              \
    REPEAT_64(X);                                                                                  \
    REPEAT_32(X);                                                                                  \
    REPEAT_4(X)  // 64+32+4=100

__global__ void
pc_sampling_kernel(const int c)
{
    int a = 0;
#pragma nounroll
    for(int i = 0; i < ITER_NUM; i++)
    {
        REPEAT_100(ASM_LINE);
    }
}

int
main()
{
    auto tid = roctx_thread_id_t{};
    // get the thread id recognized by rocprofiler-sdk from roctx
    roctxGetThreadId(&tid);

    auto stream = hipStream_t{};
    checkHipErrors(hipStreamCreate(&stream));

    // pause API tracing
    roctxProfilerPause(tid);

    // Set up matrices
    float* gpuMatrix          = nullptr;
    float* gpuTransposeMatrix = nullptr;
    float* Matrix             = nullptr;

    Matrix = (float*) malloc(NUM * sizeof(float));
    // initialize the input data
    for(auto i = 0; i < NUM; i++)
    {
        Matrix[i] = (float) i * 10.0f;
    }

    // allocate the memory on the device side
    checkHipErrors(hipMalloc((void**) &gpuMatrix, NUM * sizeof(float)));
    checkHipErrors(hipMalloc((void**) &gpuTransposeMatrix, NUM * sizeof(float)));

    // Memory transfer from host to device
    checkHipErrors(hipMemcpy(gpuMatrix, Matrix, NUM * sizeof(float), hipMemcpyHostToDevice));
    checkHipErrors(
        hipMemcpy(gpuTransposeMatrix, gpuMatrix, NUM * sizeof(float), hipMemcpyDeviceToDevice));

    // Set up simple kernels
    constexpr auto   NUM_KERNELS              = 64;
    constexpr size_t TARGET_KERNEL_ITERATIONS = NUM_KERNELS / 4;
    float*           result                   = nullptr;
    float            varA                     = 5.5;
    float            varB                     = 11.7;
    uint32_t         num_blocks               = BLOCK_SIZE;
    checkHipErrors(hipMallocAsync(&result, sizeof(float), stream));
    for(auto i = 0; i < NUM_KERNELS; ++i)
    {
        kernel_add<<<dim3(1), dim3(1), 0, stream>>>(varA++, varB++, result);
        kernel_mult<<<dim3(1), dim3(1), 0, stream>>>(varA++, varB++, result);
        // Run target kernel
        if(i % TARGET_KERNEL_ITERATIONS == 0)
        {
            roctxProfilerResume(tid);
            hipLaunchKernelGGL(target_kernel,
                               dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
                               dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                               0,
                               stream,
                               gpuTransposeMatrix,
                               gpuMatrix,
                               WIDTH);
            pc_sampling_kernel<<<num_blocks, i>>>(i);
            roctxProfilerPause(tid);
        }
    }

    // Nested test here
    roctxProfilerResume(tid);
    roctxProfilerResume(tid);
    roctxProfilerPause(tid);
    // Kernel should appear only with --selected-regions-ref-count
    hipLaunchKernelGGL(nested_kernel,
                       dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
                       dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                       0,
                       stream,
                       gpuTransposeMatrix,
                       gpuMatrix,
                       WIDTH);
    roctxProfilerPause(tid);
    // Should not appear in trace
    hipLaunchKernelGGL(nested_kernel,
                       dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
                       dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                       0,
                       stream,
                       gpuTransposeMatrix,
                       gpuMatrix,
                       WIDTH);

    // Wait for all kernels to complete before freeing memory
    checkHipErrors(hipStreamSynchronize(stream));

    // free the resources on device side
    checkHipErrors(hipFree(result));
    checkHipErrors(hipFree(gpuMatrix));
    checkHipErrors(hipFree(gpuTransposeMatrix));

    // destroy the stream
    checkHipErrors(hipStreamDestroy(stream));

    // free the resources on host side
    free(Matrix);
}
