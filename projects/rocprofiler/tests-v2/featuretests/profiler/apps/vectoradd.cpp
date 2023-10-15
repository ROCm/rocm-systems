/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

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
#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <iostream>
#include "hip/hip_runtime.h"
#include <unistd.h>
#include <vector>

#ifdef NDEBUG
#define HIP_ASSERT(x) x
#else
#define HIP_ASSERT(x) (assert((x)==hipSuccess))
#endif


#define WIDTH     (1024)
#define HEIGHT    (1024)

#define NUM       (WIDTH*HEIGHT)

#define THREADS_PER_BLOCK_X  64
#define THREADS_PER_BLOCK_Y  1
#define THREADS_PER_BLOCK_Z  1

// Computes vectorAdd with matrix-multiply
template<typename T>
__global__ void addition_kernel(
    T* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ c,
    int width,
    int height
) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= WIDTH || y >= HEIGHT) return;
    int index = y * width + x;

    a[index] = b[index]+c[index];
}


__global__ void subtract_kernel(
    float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ c,
    int width,
    int height
) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= WIDTH || y >= HEIGHT) return;
    int index = y * width + x;

    a[index] = abs(b[index]-c[index]);
}

__global__ void multiply_kernel(
    float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ c,
    int width,
    int height
) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= WIDTH || y >= HEIGHT) return;
    int index = y * width + x;

    a[index] = (b[index]-1)*(c[index]-1)+1;
}

__global__ void divide_kernel(
    float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ c,
    int width,
    int height
) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    if (x >= WIDTH || y >= HEIGHT) return;
    int index = y * width + x;

    a[index] = (b[index]-c[index]) / abs(c[index]+b[index]) + 1;
}

using namespace std;

void run(int NUM_QUEUE) {
    std::vector<float*> hostA(NUM_QUEUE);
    std::vector<float*> hostB(NUM_QUEUE);
    std::vector<float*> hostC(NUM_QUEUE);

    std::vector<float*> deviceA(NUM_QUEUE);
    std::vector<float*> deviceB(NUM_QUEUE);
    std::vector<float*> deviceC(NUM_QUEUE);

    std::vector<hipStream_t> streams(NUM_QUEUE);

    hipDeviceProp_t devProp;
    hipGetDeviceProperties(&devProp, 0);
    cout << " System minor " << devProp.minor << endl;
    cout << " System major " << devProp.major << endl;
    cout << " agent prop name " << devProp.name << endl;
    cout << "hip Device prop succeeded " << endl ;

    int i;
    int errors;

    for (int q=0; q<NUM_QUEUE; q++) {
        hipStreamCreateWithFlags(&streams[q], hipStreamNonBlocking);

        HIP_ASSERT(hipHostMalloc(&hostA[q], NUM * sizeof(float), 0));
        HIP_ASSERT(hipHostMalloc(&hostB[q], NUM * sizeof(float), 0));
        HIP_ASSERT(hipHostMalloc(&hostC[q], NUM * sizeof(float), 0));
        
        // initialize the input data
        for (i = 0; i < NUM; i++) {
            hostB[q][i] = (float)i;
            hostC[q][i] = (float)i*100.0f;
        }

        HIP_ASSERT(hipMalloc((void**)(&deviceA[q]), NUM * sizeof(float)));
        HIP_ASSERT(hipMalloc((void**)(&deviceB[q]), NUM * sizeof(float)));
        HIP_ASSERT(hipMalloc((void**)(&deviceC[q]), NUM * sizeof(float)));
        
        HIP_ASSERT(hipMemcpyAsync(deviceB[q], hostB[q], NUM*sizeof(float), hipMemcpyHostToDevice, streams[q]));
        HIP_ASSERT(hipMemcpyAsync(deviceC[q], hostC[q], NUM*sizeof(float), hipMemcpyHostToDevice, streams[q]));
    }
hipDeviceSynchronize();

    for (int RUN_I=0; RUN_I<2; RUN_I++)
    {
        int q = (4*RUN_I+0)%NUM_QUEUE;
        hipLaunchKernelGGL(addition_kernel,
                        dim3(WIDTH/THREADS_PER_BLOCK_X, HEIGHT/THREADS_PER_BLOCK_Y),
                        dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                        0, streams[q],
                        deviceA[q], deviceB[q], deviceC[q] ,WIDTH ,HEIGHT);

hipDeviceSynchronize();
        q = (4*RUN_I+1)%NUM_QUEUE;
        hipLaunchKernelGGL(subtract_kernel,
                        dim3(WIDTH/THREADS_PER_BLOCK_X, HEIGHT/THREADS_PER_BLOCK_Y),
                        dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                        0, streams[q],
                        deviceA[q], deviceB[q], deviceC[q] ,WIDTH ,HEIGHT);

hipDeviceSynchronize();
        q = (4*RUN_I+2)%NUM_QUEUE;
        hipLaunchKernelGGL(multiply_kernel,
                        dim3(WIDTH/THREADS_PER_BLOCK_X, HEIGHT/THREADS_PER_BLOCK_Y),
                        dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                        0, streams[q],
                        deviceA[q], deviceB[q], deviceC[q] ,WIDTH ,HEIGHT);

hipDeviceSynchronize();
        q = (4*RUN_I+3)%NUM_QUEUE;
        hipLaunchKernelGGL(divide_kernel,
                        dim3(WIDTH/THREADS_PER_BLOCK_X, HEIGHT/THREADS_PER_BLOCK_Y),
                        dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                        0, streams[q],
                        deviceB[q], deviceA[q], deviceC[q] ,WIDTH ,HEIGHT);
hipDeviceSynchronize();
    }

    for (int q=0; q<NUM_QUEUE; q++)
        HIP_ASSERT(hipMemcpyAsync(hostA[q], deviceA[q], NUM*sizeof(float), hipMemcpyDeviceToHost, streams[q]));


    for (int q=0; q<NUM_QUEUE; q++) {
        HIP_ASSERT(hipMemcpy(hostA[q], deviceA[q], NUM*sizeof(float), hipMemcpyDeviceToHost));
        hipDeviceSynchronize();

        HIP_ASSERT(hipFree(deviceA[q]));
        HIP_ASSERT(hipFree(deviceB[q]));
        HIP_ASSERT(hipFree(deviceC[q]));

        hipHostFree(hostA[q]);
        hipHostFree(hostB[q]);
        hipHostFree(hostC[q]);
        hipStreamDestroy(streams[q]);
    }
}

int main() {
    run(1);
    return 0;
}
