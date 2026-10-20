// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file race_bench.hip
/// @brief Standalone synthetic workloads for race-check performance measurements.
///
/// Usage: race_bench <stream|lds_reduce|gemm|histogram|compute> <size>
///
/// The reported kernel time includes a device synchronization around exactly
/// one timed launch. Host-to-device setup and result validation are outside the
/// timed region.

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(call)                                                                                \
  do {                                                                                             \
    const hipError_t error_ = (call);                                                              \
    if (error_ != hipSuccess) {                                                                    \
      std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(error_));           \
      std::exit(2);                                                                                \
    }                                                                                              \
  } while (0)

__global__ void stream_kernel(const float *a, const float *b, float *c, float scale, size_t n) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
       i += size_t(gridDim.x) * blockDim.x)
    c[i] = a[i] * scale + b[i];
}

__global__ void lds_reduce_kernel(const float *input, float *output, size_t n) {
  __shared__ float shared[256];
  const unsigned tid = threadIdx.x;
  const size_t i = blockIdx.x * size_t(blockDim.x) + tid;
  shared[tid] = i < n ? input[i] : 0.0f;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride)
      shared[tid] += shared[tid + stride];
    __syncthreads();
  }
  if (tid == 0)
    output[blockIdx.x] = shared[0];
}

constexpr int kTile = 16;

__global__ void gemm_kernel(const float *a, const float *b, float *c, int n) {
  __shared__ float tile_a[kTile][kTile];
  __shared__ float tile_b[kTile][kTile];
  const int row = blockIdx.y * kTile + threadIdx.y;
  const int column = blockIdx.x * kTile + threadIdx.x;
  float accumulator = 0.0f;
  for (int tile = 0; tile < n; tile += kTile) {
    tile_a[threadIdx.y][threadIdx.x] = a[row * n + tile + threadIdx.x];
    tile_b[threadIdx.y][threadIdx.x] = b[(tile + threadIdx.y) * n + column];
    __syncthreads();
    for (int k = 0; k < kTile; ++k)
      accumulator += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
    __syncthreads();
  }
  c[row * n + column] = accumulator;
}

__global__ void histogram_kernel(const unsigned *input, unsigned *bins, size_t n) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n;
       i += size_t(gridDim.x) * blockDim.x)
    atomicAdd(&bins[input[i] & 255u], 1u);
}

__global__ void compute_kernel(float *output, int iterations) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  float x = i * 1e-3f;
  float y = 1.0f;
  for (int iteration = 0; iteration < iterations; ++iteration) {
    y = fmaf(y, 0.999f, x);
    x = fmaf(x, 1.0001f, 1e-6f);
  }
  output[i] = x + y;
}

template <typename Launch> double timed(Launch &&launch) {
  CHECK(hipDeviceSynchronize());
  const auto start = std::chrono::steady_clock::now();
  launch();
  CHECK(hipGetLastError());
  CHECK(hipDeviceSynchronize());
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
      .count();
}

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s <stream|lds_reduce|gemm|histogram|compute> <size>\n", argv[0]);
    return 2;
  }

  const char *mode = argv[1];
  const size_t size = std::strtoull(argv[2], nullptr, 0);
  bool ok = true;
  double milliseconds = 0;

  if (!std::strcmp(mode, "stream")) {
    const size_t n = size;
    std::vector<float> a(n);
    std::vector<float> b(n);
    std::vector<float> c(n);
    for (size_t i = 0; i < n; ++i) {
      a[i] = float(i % 1000);
      b[i] = 1.0f;
    }
    float *device_a = nullptr;
    float *device_b = nullptr;
    float *device_c = nullptr;
    CHECK(hipMalloc(&device_a, n * sizeof(float)));
    CHECK(hipMalloc(&device_b, n * sizeof(float)));
    CHECK(hipMalloc(&device_c, n * sizeof(float)));
    CHECK(hipMemcpy(device_a, a.data(), n * sizeof(float), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(device_b, b.data(), n * sizeof(float), hipMemcpyHostToDevice));
    const unsigned blocks = unsigned((n + 255) / 256);
    milliseconds =
        timed([&] { stream_kernel<<<blocks, 256>>>(device_a, device_b, device_c, 2.0f, n); });
    CHECK(hipMemcpy(c.data(), device_c, n * sizeof(float), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < n && ok; ++i)
      ok = c[i] == a[i] * 2.0f + 1.0f;
    CHECK(hipFree(device_a));
    CHECK(hipFree(device_b));
    CHECK(hipFree(device_c));
  } else if (!std::strcmp(mode, "lds_reduce")) {
    const size_t n = size;
    const size_t blocks = (n + 255) / 256;
    std::vector<float> input(n, 1.0f);
    std::vector<float> output(blocks);
    float *device_input = nullptr;
    float *device_output = nullptr;
    CHECK(hipMalloc(&device_input, n * sizeof(float)));
    CHECK(hipMalloc(&device_output, blocks * sizeof(float)));
    CHECK(hipMemcpy(device_input, input.data(), n * sizeof(float), hipMemcpyHostToDevice));
    milliseconds = timed(
        [&] { lds_reduce_kernel<<<unsigned(blocks), 256>>>(device_input, device_output, n); });
    CHECK(hipMemcpy(output.data(), device_output, blocks * sizeof(float), hipMemcpyDeviceToHost));
    for (size_t block = 0; block < blocks && ok; ++block)
      ok = output[block] == float(std::min<size_t>(256, n - block * 256));
    CHECK(hipFree(device_input));
    CHECK(hipFree(device_output));
  } else if (!std::strcmp(mode, "gemm")) {
    const int n = int(size);
    if (n % kTile) {
      std::fprintf(stderr, "gemm size must be a multiple of %d\n", kTile);
      return 2;
    }
    std::vector<float> a(size_t(n) * n, 1.0f);
    std::vector<float> b(size_t(n) * n, 2.0f);
    std::vector<float> c(size_t(n) * n);
    float *device_a = nullptr;
    float *device_b = nullptr;
    float *device_c = nullptr;
    CHECK(hipMalloc(&device_a, a.size() * sizeof(float)));
    CHECK(hipMalloc(&device_b, b.size() * sizeof(float)));
    CHECK(hipMalloc(&device_c, c.size() * sizeof(float)));
    CHECK(hipMemcpy(device_a, a.data(), a.size() * sizeof(float), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(device_b, b.data(), b.size() * sizeof(float), hipMemcpyHostToDevice));
    const dim3 grid(n / kTile, n / kTile);
    const dim3 block(kTile, kTile);
    milliseconds = timed([&] { gemm_kernel<<<grid, block>>>(device_a, device_b, device_c, n); });
    CHECK(hipMemcpy(c.data(), device_c, c.size() * sizeof(float), hipMemcpyDeviceToHost));
    for (const float value : c)
      ok = ok && value == 2.0f * n;
    CHECK(hipFree(device_a));
    CHECK(hipFree(device_b));
    CHECK(hipFree(device_c));
  } else if (!std::strcmp(mode, "histogram")) {
    const size_t n = size;
    std::vector<unsigned> input(n);
    std::vector<unsigned> bins(256);
    for (size_t i = 0; i < n; ++i)
      input[i] = unsigned(i * 2654435761u);
    unsigned *device_input = nullptr;
    unsigned *device_bins = nullptr;
    CHECK(hipMalloc(&device_input, n * sizeof(unsigned)));
    CHECK(hipMalloc(&device_bins, bins.size() * sizeof(unsigned)));
    CHECK(hipMemcpy(device_input, input.data(), n * sizeof(unsigned), hipMemcpyHostToDevice));
    CHECK(hipMemset(device_bins, 0, bins.size() * sizeof(unsigned)));
    const unsigned blocks = unsigned(std::min<size_t>((n + 255) / 256, 1024));
    milliseconds = timed([&] { histogram_kernel<<<blocks, 256>>>(device_input, device_bins, n); });
    CHECK(
        hipMemcpy(bins.data(), device_bins, bins.size() * sizeof(unsigned), hipMemcpyDeviceToHost));
    size_t total = 0;
    for (const unsigned value : bins)
      total += value;
    ok = total == n;
    CHECK(hipFree(device_input));
    CHECK(hipFree(device_bins));
  } else if (!std::strcmp(mode, "compute")) {
    constexpr unsigned kBlocks = 256;
    float *device_output = nullptr;
    CHECK(hipMalloc(&device_output, kBlocks * 256 * sizeof(float)));
    milliseconds = timed([&] { compute_kernel<<<kBlocks, 256>>>(device_output, int(size)); });
    std::vector<float> output(kBlocks * 256);
    CHECK(hipMemcpy(output.data(), device_output, output.size() * sizeof(float),
                    hipMemcpyDeviceToHost));
    for (const float value : output)
      ok = ok && std::isfinite(value);
    CHECK(hipFree(device_output));
  } else {
    std::fprintf(stderr, "unknown mode %s\n", mode);
    return 2;
  }

  std::printf("kernel_ms=%.3f\n", milliseconds);
  std::fprintf(stderr, "%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
