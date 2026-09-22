// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

/*
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
