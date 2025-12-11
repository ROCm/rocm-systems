// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Reference:  Finite Difference Methods for Ordinary and Partial Differential
// Equations by Randall J. LeVeque.

#include <algorithm>
#include <hip/hip_runtime.h>
#include <iostream>

// Helper macro for HIP error checking
#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    hipError_t err = call;                                                     \
    if (err != hipSuccess) {                                                   \
      std::cerr << "HIP error: " << hipGetErrorString(err) << " at "           \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

void initialize(float *U, int N) {
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      float x = 2.0f * (float)i / (float)N - 1.0f;
      float y = 2.0f * (float)j / (float)N - 1.0f;
      U[i * N + j] = 1.0f / (1.0f + x * x + y * y); // Initial guess
    }
  }
}

__global__ void jacobi_iteration(float *__restrict__ U_new,
                                 const float *__restrict__ U_old, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  int idx = j + i * N; // Flattened index for 2D grid

  extern __shared__ float U[];
  int s_i = threadIdx.x + 1;
  int s_j = threadIdx.y + 1;

  int tile_height = blockDim.y + 2;
  int s_idx = s_j + s_i * tile_height;

  // Load data into shared memory with halo regions
  if (i < N && j < N) {
    U[s_idx] = U_old[idx];

    // Load halo regions
    if (threadIdx.x == 0) {
      U[s_j] = (i > 0) ? U_old[(i - 1) * N + j] : U_old[(N - 1) * N + j];
    }

    if (threadIdx.x == blockDim.x - 1) {
      U[s_j + (blockDim.x + 1) * tile_height] =
          (i < N - 1) ? U_old[(i + 1) * N + j] : U_old[j];
    }

    if (threadIdx.y == 0) {
      U[s_i * tile_height] =
          (j > 0) ? U_old[i * N + (j - 1)] : U_old[i * N + (N - 1)];
    }

    if (threadIdx.y == blockDim.y - 1) {
      U[s_i * tile_height + blockDim.y + 1] =
          (j < N - 1) ? U_old[i * N + (j + 1)] : U_old[i * N];
    }
  }

  if (i < N && j < N) {
    int east = s_idx - tile_height;
    int west = s_idx + tile_height;
    int north = s_idx + 1;
    int south = s_idx - 1;
    U_new[idx] = 0.25f * (U[east] + U[west] + U[north] + U[south]);
  }
}

int main(int argc, char *argv[]) {
  int iter = 1000;

  for (int i = 0; i < argc; i++) {
    if (std::string(argv[i]) == "-i" && i + 1 < argc) {
      iter = std::atoi(argv[i + 1]);
    }
  }

  // Problem size
  const int n_small = 1024;
  const int n_mid = 2048;
  const int n_large = 4096;

  // Define block and grid sizes
  dim3 blockSize_small(16, 16);
  dim3 gridSize_small((n_small + blockSize_small.x - 1) / blockSize_small.x,
                      (n_small + blockSize_small.y - 1) / blockSize_small.y);
  dim3 blockSize_mid(24, 24);
  dim3 gridSize_mid((n_mid + blockSize_mid.x - 1) / blockSize_mid.x,
                    (n_mid + blockSize_mid.y - 1) / blockSize_mid.y);
  dim3 blockSize_large(32, 32);
  dim3 gridSize_large((n_large + blockSize_large.x - 1) / blockSize_large.x,
                      (n_large + blockSize_large.y - 1) / blockSize_large.y);

  // Host memory pointers
  float *h_U_small = new float[n_small * n_small];
  float *h_U_mid = new float[n_mid * n_mid];
  float *h_U_large = new float[n_large * n_large];

  // Initialize host arrays
  initialize(h_U_small, n_small);
  initialize(h_U_mid, n_mid);
  initialize(h_U_large, n_large);

  // Device memory pointers
  float *d_U_small_new, *d_U_small_old;
  float *d_U_mid_new, *d_U_mid_old;
  float *d_U_large_new, *d_U_large_old;

  // Allocate device memory
  HIP_CHECK(hipMalloc(&d_U_small_new, n_small * n_small * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_U_small_old, n_small * n_small * sizeof(float)));

  HIP_CHECK(hipMalloc(&d_U_mid_new, n_mid * n_mid * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_U_mid_old, n_mid * n_mid * sizeof(float)));

  HIP_CHECK(hipMalloc(&d_U_large_new, n_large * n_large * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_U_large_old, n_large * n_large * sizeof(float)));

  // Copy host arrays to device
  HIP_CHECK(hipMemcpy(d_U_small_old, h_U_small,
                      n_small * n_small * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_U_mid_old, h_U_mid, n_mid * n_mid * sizeof(float),
                      hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_U_large_old, h_U_large,
                      n_large * n_large * sizeof(float),
                      hipMemcpyHostToDevice));

  // Skip iteration for iteration multiplexing testing
  int n_skip_iter = 10;
  int next_skip = 0;
  int num_kernels = 3;

  // Perform Jacobi iterations
  for (int it = 0; it < iter; ++it) {
    bool skip_iteration = ((it + 1) % n_skip_iter == 0);
    if (not skip_iteration || (next_skip != 0)) {
      jacobi_iteration<<<gridSize_small, blockSize_small,
                         (blockSize_small.x + 2) * (blockSize_small.y + 2) *
                             sizeof(float)>>>(d_U_small_new, d_U_small_old,
                                              n_small);
    }

    if (not skip_iteration || (next_skip != 1)) {
      jacobi_iteration<<<gridSize_mid, blockSize_mid,
                         (blockSize_mid.x + 2) * (blockSize_mid.y + 2) *
                             sizeof(float)>>>(d_U_mid_new, d_U_mid_old, n_mid);
    }

    if (not skip_iteration || (next_skip != 2)) {
      jacobi_iteration<<<gridSize_large, blockSize_large,
                         (blockSize_large.x + 2) * (blockSize_large.y + 2) *
                             sizeof(float)>>>(d_U_large_new, d_U_large_old,
                                              n_large);
    }

    if (skip_iteration) {
      next_skip = (next_skip + 1) % num_kernels;
    }

    HIP_CHECK(hipDeviceSynchronize());

    // Swap pointers
    std::swap(d_U_small_new, d_U_small_old);
    std::swap(d_U_mid_new, d_U_mid_old);
    std::swap(d_U_large_new, d_U_large_old);
  }

  // Free device memory
  HIP_CHECK(hipFree(d_U_small_new));
  HIP_CHECK(hipFree(d_U_small_old));
  HIP_CHECK(hipFree(d_U_mid_new));
  HIP_CHECK(hipFree(d_U_mid_old));
  HIP_CHECK(hipFree(d_U_large_new));
  HIP_CHECK(hipFree(d_U_large_old));

  return 0;
}