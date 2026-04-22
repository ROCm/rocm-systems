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

#include <hip/hip_runtime.h>
#include <atomic>
#include <chrono>
#include <thread>
#include <iostream>

#define HIP_CALL(call)                                                                             \
    do                                                                                             \
    {                                                                                              \
        hipError_t err = call;                                                                     \
        if(err != hipSuccess)                                                                      \
        {                                                                                          \
            fprintf(stderr, "%s\n", hipGetErrorString(err));                                       \
            abort();                                                                               \
        }                                                                                          \
    } while(0)

__device__ void sync()
{
    __syncthreads();
}

// A small device function that should NOT be instrumented
__device__ float add_one(float x)
{
    asm volatile("v_add_f32 %0, %1, 1" : "=v"(x) : "v"(x));
    asm volatile("v_add_f32 %0, %1, 1" : "=v"(x) : "v"(x));
    asm volatile("v_add_f32 %0, %1, 1" : "=v"(x) : "v"(x));
    asm volatile("v_add_f32 %0, %1, 1" : "=v"(x) : "v"(x));
    asm volatile("s_nop 3");
    return x;
}

__device__ float heavy_compute(int iters, float* out)
{
    float result = out[threadIdx.x];
    for (int i = 0; i < iters; i++) {
        result = add_one(result);
        result = result / (result + 1.0f);
        result = add_one(result);
        result = result * 2.0f - 0.5f;
        if (i%8 == 0)
        {
            sync();
            out[(threadIdx.x+64)%256] += result;
            sync();
        }
        else
            result += out[threadIdx.x]*0.01f;
    }
    return result;
}

__global__ void compute_kernel(float *out, const float *in, int size, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    __shared__ float shm[256];
    shm[threadIdx.x] = in[idx];

    float val = heavy_compute(iters, shm);
    out[idx] = add_one(val);
}

int main() {
    const int N = 304 * 256;
    float *d_in, *d_out;
    hipMalloc(&d_in, N * sizeof(float));
    hipMalloc(&d_out, N * sizeof(float));

    std::vector<float> h_in(N, 1.0f);
    hipMemcpy(d_in, h_in.data(), N * sizeof(float), hipMemcpyHostToDevice);

    compute_kernel<<<(N + 255) / 256, 256>>>(d_out, d_in, N, 10);
    hipDeviceSynchronize();

    hipFree(d_in);
    hipFree(d_out);
    return 0;
}
