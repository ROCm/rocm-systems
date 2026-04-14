/*
##############################################################################bl
# MIT License
#
# Copyright (c) 2021 - 2023 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
##############################################################################el



An example to explore achieved occupancy, and various occupancy limiters.
Written by Nicholas Curtis [AMD].
*/


#include "common.h"

__global__ void empty(int N, double* ptr) {

}

constexpr int bound = 16;
__launch_bounds__(256)
__global__ void vgprbound(int N, double* ptr) {
    double intermediates[bound];
    for (int i = 0 ; i < bound; ++i) intermediates[i] = N * threadIdx.x;
    double x = ptr[threadIdx.x];
    for (int i = 0; i < 100; ++i) {
        x += sin(pow(__shfl(x, i % warpSize) * intermediates[(i - 1) % bound], intermediates[i % bound]));
        intermediates[i % bound] = x;
    }
    if (x == N) ptr[threadIdx.x] = x;
}

constexpr size_t fully_allocate_lds = 64ul * 1024ul / sizeof(double);
__launch_bounds__(256)
__global__ void ldsbound(int N, double* ptr) {
    __shared__ double intermediates[fully_allocate_lds];
    for (int i = threadIdx.x ; i < fully_allocate_lds; i += blockDim.x) intermediates[i] = N * threadIdx.x;
    __syncthreads();
    double x = ptr[threadIdx.x];
    for (int i = threadIdx.x; i < fully_allocate_lds; i += blockDim.x) {
        x += sin(pow(__shfl(x, i % warpSize) * intermediates[(i - 1) % fully_allocate_lds], intermediates[i % fully_allocate_lds]));
        __syncthreads();
        intermediates[i % fully_allocate_lds] = x;
    }
    if (x == N) ptr[threadIdx.x] = x;
}

constexpr int sgprlim = 1;
__launch_bounds__(1024, 8)
__global__ void sgprbound(int N, double* ptr) {
    double intermediates[sgprlim];
    for (int i = 0 ; i < sgprlim; ++i) intermediates[i] = i;
    double x = ptr[0];
    #pragma unroll 1
    for (int i = 0; i < 100; ++i) {
        x += sin(pow(intermediates[(i - 1) % sgprlim], intermediates[i % sgprlim]));
        intermediates[i % sgprlim] = x;
    }
    if (x == N) ptr[0] = x;
}

int main() {
    double* ptr;
    hipCheck(hipMalloc(&ptr, 1024 * sizeof(double)));
    vgprbound<<<1024 * 1024, 256>>>(0, ptr);
    hipCheck(hipGetLastError());
    hipCheck(hipDeviceSynchronize());
    vgprbound<<<1024 * 1024, 256>>>(0, ptr);
    hipCheck(hipGetLastError());
    hipCheck(hipDeviceSynchronize());
    ldsbound<<<1024 * 1024, 256>>>(0, ptr);
    hipCheck(hipGetLastError());
    hipCheck(hipDeviceSynchronize());
    ldsbound<<<1024 * 1024, 256>>>(0, ptr);
    hipCheck(hipGetLastError());
    hipCheck(hipDeviceSynchronize());
    sgprbound<<<1024 * 1024, 256>>>(0, ptr);
    hipCheck(hipGetLastError());
    hipCheck(hipDeviceSynchronize());
    sgprbound<<<1024 * 1024, 256>>>(0, ptr);
    hipCheck(hipGetLastError());
    hipCheck(hipDeviceSynchronize());
    empty<<<1024 * 1024, 256>>>(0, ptr);
    hipCheck(hipGetLastError());
    hipCheck(hipDeviceSynchronize());
    empty<<<1024 * 1024, 256>>>(0, ptr);
    hipCheck(hipGetLastError());
    hipCheck(hipDeviceSynchronize());
    hipCheck(hipFree(ptr));
}
