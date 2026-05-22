// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Performance benchmark for rocjitsu emulation. Runs 3 HIP kernels with
// different memory access patterns, prints structured timing output for
// the runner script (run_perf_matrix.sh).
//
// Compiled with hipcc. Requires LD_PRELOAD=librocjitsu_kmd.so and
// RJ_CONFIG env var. Race detection is toggled via RJ_RACE=1.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime.h>
#include <unistd.h>
#include <vector>

// ---------------------------------------------------------------------------
// Problem sizes. Tune these so each kernel takes roughly 2 seconds under
// emulation without race detection.
// ---------------------------------------------------------------------------
#ifndef VECTOR_SCALE_N
#define VECTOR_SCALE_N (6144 * 1024)
#endif

#ifndef REDUCE_N
#define REDUCE_N (3328 * 1024)
#endif

#ifndef MATMUL_DIM
#define MATMUL_DIM 330
#endif

#define TILE_SIZE 16

#define HIP_CHECK(call)                                                                            \
  do {                                                                                             \
    hipError_t err = (call);                                                                       \
    if (err != hipSuccess) {                                                                       \
      fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err));     \
      _exit(1);                                                                                    \
    }                                                                                              \
  } while (0)

// ---------------------------------------------------------------------------
// Kernel 1: vector_scale — no LDS, pure VGPR work.
// Baseline for per-instruction plugin overhead without LDS validation.
// ---------------------------------------------------------------------------
__global__ void vector_scale_kernel(float *data, float scalar, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N)
    data[i] = data[i] * scalar + 1.0f;
}

static bool run_vector_scale(const char *filter) {
  if (filter && strcmp(filter, "vector_scale") != 0)
    return true;

  const int N = VECTOR_SCALE_N;
  const size_t bytes = N * sizeof(float);

  std::vector<float> h_data(N);
  for (int i = 0; i < N; ++i)
    h_data[i] = static_cast<float>(i % 100);

  float *d_data = nullptr;
  HIP_CHECK(hipMalloc(&d_data, bytes));
  HIP_CHECK(hipMemcpy(d_data, h_data.data(), bytes, hipMemcpyHostToDevice));

  const int blockSize = 256;
  const int gridSize = (N + blockSize - 1) / blockSize;
  const float scalar = 3.0f;

  auto t0 = std::chrono::steady_clock::now();
  vector_scale_kernel<<<gridSize, blockSize>>>(d_data, scalar, N);
  HIP_CHECK(hipDeviceSynchronize());
  auto t1 = std::chrono::steady_clock::now();

  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  std::vector<float> h_out(N);
  HIP_CHECK(hipMemcpy(h_out.data(), d_data, bytes, hipMemcpyDeviceToHost));

  int errors = 0;
  for (int i = 0; i < N; i += N / 16) {
    float expected = h_data[i] * scalar + 1.0f;
    if (h_out[i] != expected)
      ++errors;
  }

  printf("PERF vector_scale %.1f\n", ms);
  printf("CHECK vector_scale %s (%d errors)\n", errors == 0 ? "PASS" : "FAIL", errors);

  (void)hipFree(d_data);
  return errors == 0;
}

// ---------------------------------------------------------------------------
// Kernel 2: reduce_lds — shared memory reduction.
// Exercises race detection's LDS byte counters and barrier hooks.
// Each workgroup reduces blockDim.x elements to a single sum via LDS.
// ---------------------------------------------------------------------------
__global__ void reduce_lds_kernel(const float *input, float *output, int N) {
  extern __shared__ float sdata[];

  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (gid < N) ? input[gid] : 0.0f;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }

  if (tid == 0)
    output[blockIdx.x] = sdata[0];
}

static bool run_reduce_lds(const char *filter) {
  if (filter && strcmp(filter, "reduce_lds") != 0)
    return true;

  const int N = REDUCE_N;
  const int blockSize = 256;
  const int gridSize = (N + blockSize - 1) / blockSize;
  const size_t in_bytes = N * sizeof(float);
  const size_t out_bytes = gridSize * sizeof(float);

  std::vector<float> h_input(N);
  for (int i = 0; i < N; ++i)
    h_input[i] = static_cast<float>(i % 10);

  float *d_input = nullptr, *d_output = nullptr;
  HIP_CHECK(hipMalloc(&d_input, in_bytes));
  HIP_CHECK(hipMalloc(&d_output, out_bytes));
  HIP_CHECK(hipMemcpy(d_input, h_input.data(), in_bytes, hipMemcpyHostToDevice));

  auto t0 = std::chrono::steady_clock::now();
  reduce_lds_kernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_input, d_output, N);
  HIP_CHECK(hipDeviceSynchronize());
  auto t1 = std::chrono::steady_clock::now();

  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  std::vector<float> h_output(gridSize);
  HIP_CHECK(hipMemcpy(h_output.data(), d_output, out_bytes, hipMemcpyDeviceToHost));

  double gpu_sum = 0.0;
  for (int i = 0; i < gridSize; ++i)
    gpu_sum += h_output[i];

  double cpu_sum = 0.0;
  for (int i = 0; i < N; ++i)
    cpu_sum += h_input[i];

  bool ok = gpu_sum == cpu_sum;

  printf("PERF reduce_lds %.1f\n", ms);
  printf("CHECK reduce_lds %s (gpu=%.0f cpu=%.0f)\n", ok ? "PASS" : "FAIL", gpu_sum, cpu_sum);

  (void)hipFree(d_input);
  (void)hipFree(d_output);
  return ok;
}

// ---------------------------------------------------------------------------
// Kernel 3: matmul_tiled_lds — tiled matrix multiply using shared memory.
// Heavy LDS usage with barriers between tile load and compute phases.
// C = A * B, all square matrices of dimension MATMUL_DIM.
// ---------------------------------------------------------------------------
__global__ void matmul_tiled_kernel(const float *A, const float *B, float *C, int M, int N, int K) {
  __shared__ float As[TILE_SIZE][TILE_SIZE];
  __shared__ float Bs[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;

  float sum = 0.0f;

  for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; ++t) {
    int a_col = t * TILE_SIZE + threadIdx.x;
    int b_row = t * TILE_SIZE + threadIdx.y;

    As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
    Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

    __syncthreads();

    for (int k = 0; k < TILE_SIZE; ++k)
      sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

    __syncthreads();
  }

  if (row < M && col < N)
    C[row * N + col] = sum;
}

static bool run_matmul_tiled_lds(const char *filter) {
  if (filter && strcmp(filter, "matmul_tiled_lds") != 0)
    return true;

  const int DIM = MATMUL_DIM;
  const size_t elems = (size_t)DIM * DIM;
  const size_t bytes = elems * sizeof(float);

  std::vector<float> h_A(elems), h_B(elems);
  for (size_t i = 0; i < elems; ++i) {
    h_A[i] = ((i * 7 + 3) & 1) ? 1.0f : -1.0f;
    h_B[i] = ((i * 13 + 5) & 1) ? 1.0f : -1.0f;
  }

  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  HIP_CHECK(hipMalloc(&d_A, bytes));
  HIP_CHECK(hipMalloc(&d_B, bytes));
  HIP_CHECK(hipMalloc(&d_C, bytes));
  HIP_CHECK(hipMemcpy(d_A, h_A.data(), bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_B, h_B.data(), bytes, hipMemcpyHostToDevice));

  dim3 block(TILE_SIZE, TILE_SIZE);
  dim3 grid((DIM + TILE_SIZE - 1) / TILE_SIZE, (DIM + TILE_SIZE - 1) / TILE_SIZE);

  auto t0 = std::chrono::steady_clock::now();
  matmul_tiled_kernel<<<grid, block>>>(d_A, d_B, d_C, DIM, DIM, DIM);
  HIP_CHECK(hipDeviceSynchronize());
  auto t1 = std::chrono::steady_clock::now();

  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  std::vector<float> h_C(elems);
  HIP_CHECK(hipMemcpy(h_C.data(), d_C, bytes, hipMemcpyDeviceToHost));

  int errors = 0;
  for (int check = 0; check < 16; ++check) {
    int r = (check * 7) % DIM;
    int c = (check * 11) % DIM;
    float ref = 0.0f;
    for (int k = 0; k < DIM; ++k)
      ref += h_A[r * DIM + k] * h_B[k * DIM + c];
    if (h_C[r * DIM + c] != ref) {
      if (errors < 5)
        fprintf(stderr, "  mismatch [%d,%d]: got=%f ref=%f\n", r, c, h_C[r * DIM + c], ref);
      ++errors;
    }
  }

  printf("PERF matmul_tiled_lds %.1f\n", ms);
  printf("CHECK matmul_tiled_lds %s (%d spot-check errors)\n", errors == 0 ? "PASS" : "FAIL",
         errors);

  (void)hipFree(d_A);
  (void)hipFree(d_B);
  (void)hipFree(d_C);
  return errors == 0;
}

// ---------------------------------------------------------------------------
// Main: run all benchmarks (or a filtered subset).
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
  const char *filter = nullptr;
  for (int i = 1; i < argc; ++i) {
    if (strncmp(argv[i], "--filter=", 9) == 0)
      filter = argv[i] + 9;
  }

  bool ok = true;
  ok &= run_vector_scale(filter);
  ok &= run_reduce_lds(filter);
  ok &= run_matmul_tiled_lds(filter);

  fflush(stdout);
  (void)hipDeviceReset();
  _exit(ok ? 0 : 1);
}
