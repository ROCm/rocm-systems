/*
##############################################################################
# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

##############################################################################
*/

#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib>
#include <chrono>

#define TILE_SIZE 32     // Maximum block size: 32 x 32 = 1024 threads/block
#define N 4096           // Matrix size: 4096 x 4096 (~67M elements)

// Helper macro for HIP error checking
#define HIP_CHECK(call)                                              \
    do {                                                            \
        hipError_t err = call;                                      \
        if (err != hipSuccess) {                                    \
            std::cerr << "HIP error: " << hipGetErrorString(err)    \
                      << " at " << __FILE__ << ":" << __LINE__      \
                      << std::endl;                                 \
            std::exit(EXIT_FAILURE);                                \
        }                                                           \
    } while(0)

__global__ void matMulKernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int width) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // Loop over tiles of input matrices
    for (int t = 0; t < width / TILE_SIZE; ++t) {
        tileA[threadIdx.y][threadIdx.x] = A[row * width + t * TILE_SIZE + threadIdx.x];
        tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * width + col];
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < TILE_SIZE; ++i)
            sum += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];

        __syncthreads();
    }

    C[row * width + col] = sum;
}

int main() {
    size_t size = N * N * sizeof(float);

    float *h_A = new float[N * N];
    float *h_B = new float[N * N];
    float *d_A, *d_B, *d_C;

    // Initialize matrices with dummy values
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = static_cast<float>(i % 100) * 0.01f;
        h_B[i] = static_cast<float>((i + 1) % 100) * 0.01f;
    }

    HIP_CHECK(hipMalloc(&d_A, size));
    HIP_CHECK(hipMalloc(&d_B, size));
    HIP_CHECK(hipMalloc(&d_C, size));

    HIP_CHECK(hipMemcpy(d_A, h_A, size, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B, size, hipMemcpyHostToDevice));

    dim3 blockDim(TILE_SIZE, TILE_SIZE);  // 32 x 32 = 1024 threads
    dim3 gridDim(N / TILE_SIZE, N / TILE_SIZE);  // 128 x 128 = 16,384 thread blocks

    std::cout << "Launching kernel with grid: (" << gridDim.x << ", " << gridDim.y
              << ") and block: (" << blockDim.x << ", " << blockDim.y << ")\n";

    auto start = std::chrono::high_resolution_clock::now();

    matMulKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Execution time: " << elapsed.count() << " seconds\n";

    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
    delete[] h_A;
    delete[] h_B;

    return 0;
}
