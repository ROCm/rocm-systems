/*
Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <iostream>
#include <mutex>

// hip header file
#include <hip/hip_runtime.h>

// ROCTx header file
#include <rocprofiler-sdk-roctx/roctx.h>

#define WIDTH 1024

#define NUM (WIDTH * WIDTH)

#define THREADS_PER_BLOCK_X 4
#define THREADS_PER_BLOCK_Y 4
#define THREADS_PER_BLOCK_Z 1

#define HIP_API_CALL(CALL)                                                                         \
    {                                                                                              \
        hipError_t error_ = (CALL);                                                                \
        if(error_ != hipSuccess)                                                                   \
        {                                                                                          \
            auto _hip_api_print_lk = auto_lock_t{print_lock};                                      \
            fprintf(stderr,                                                                        \
                    "%s:%d :: HIP error : %s\n",                                                   \
                    __FILE__,                                                                      \
                    __LINE__,                                                                      \
                    hipGetErrorString(error_));                                                    \
            throw std::runtime_error("hip_api_call");                                              \
        }                                                                                          \
    }

namespace
{
using auto_lock_t = std::unique_lock<std::mutex>;
auto print_lock   = std::mutex{};
}  // namespace

// Device (Kernel) function, it must be void
__global__ void
matrixTranspose(float* out, float* in, const int width)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    out[y * width + x] = in[x * width + y];
}

// CPU implementation of matrix transpose
void
matrixTransposeCPUReference(float* output, float* input, const unsigned int width)
{
    for(unsigned int j = 0; j < width; j++)
    {
        for(unsigned int i = 0; i < width; i++)
        {
            output[i * width + j] = input[j * width + i];
        }
    }
}

int
main()
{
    roctxRangePush("main");

    float* Matrix;
    float* TransposeMatrix;
    float* cpuTransposeMatrix;

    float* gpuMatrix;
    float* gpuTransposeMatrix;

    hipDeviceProp_t devProp;
    HIP_API_CALL(hipGetDeviceProperties(&devProp, 0));

    std::cout << "Device name " << devProp.name << std::endl;

    int i;
    int errors;

    Matrix             = (float*) malloc(NUM * sizeof(float));
    TransposeMatrix    = (float*) malloc(NUM * sizeof(float));
    cpuTransposeMatrix = (float*) malloc(NUM * sizeof(float));

    // initialize the input data
    for(i = 0; i < NUM; i++)
    {
        Matrix[i] = (float) i * 10.0f;
    }

    // allocate the memory on the device side
    HIP_API_CALL(hipMalloc((void**) &gpuMatrix, NUM * sizeof(float)));
    HIP_API_CALL(hipMalloc((void**) &gpuTransposeMatrix, NUM * sizeof(float)));

    // Memory transfer from host to device
    HIP_API_CALL(hipMemcpy(gpuMatrix, Matrix, NUM * sizeof(float), hipMemcpyHostToDevice));

    auto tid = roctx_thread_id_t{};
    roctxGetThreadId(&tid);
    roctxProfilerPause(tid);
    // Memory transfer that should be hidden by profiling tool
    HIP_API_CALL(
        hipMemcpy(gpuTransposeMatrix, gpuMatrix, NUM * sizeof(float), hipMemcpyDeviceToDevice));
    roctxProfilerResume(tid);

    roctxMark("pre-kernel-launch");
    // Lauching kernel from host
    hipLaunchKernelGGL(matrixTranspose,
                       dim3(WIDTH / THREADS_PER_BLOCK_X, WIDTH / THREADS_PER_BLOCK_Y),
                       dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                       0,
                       0,
                       gpuTransposeMatrix,
                       gpuMatrix,
                       WIDTH);
    roctxMark("post-kernel-launch");

    // Memory transfer from device to host
    HIP_API_CALL(
        hipMemcpy(TransposeMatrix, gpuTransposeMatrix, NUM * sizeof(float), hipMemcpyDeviceToHost));

    // CPU MatrixTranspose computation
    matrixTransposeCPUReference(cpuTransposeMatrix, Matrix, WIDTH);

    // verify the results
    errors     = 0;
    double eps = 1.0E-6;
    for(i = 0; i < NUM; i++)
    {
        if(std::abs(TransposeMatrix[i] - cpuTransposeMatrix[i]) > eps)
        {
            errors++;
        }
    }
    if(errors != 0)
    {
        printf("FAILED: %d errors\n", errors);
    }
    else
    {
        printf("PASSED!\n");
    }

    // free the resources on device side
    HIP_API_CALL(hipFree(gpuMatrix));
    HIP_API_CALL(hipFree(gpuTransposeMatrix));

    // free the resources on host side
    free(Matrix);
    free(TransposeMatrix);
    free(cpuTransposeMatrix);

    roctxRangePop();

    return errors;
}
