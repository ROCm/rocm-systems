// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hazard_isolation.hip
/// @brief Focused clean workloads for address-race cost attribution.
///
/// Usage: hazard_isolation <global_gate|global_full|lds_lifecycle|barrier_only|
///                          lds_only|lds_barrier> <rounds>

#include <hip/hip_runtime.h>

#include <chrono>
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

constexpr unsigned kThreads = 256;
constexpr size_t kElements = 262144;

__global__ void global_gate_kernel(volatile unsigned *data, unsigned rounds) {
  for (unsigned round = 0; round < rounds; ++round) {
    for (size_t i = threadIdx.x; i < kElements; i += blockDim.x) {
      const unsigned value = data[i];
      data[i] = value + 1;
    }
  }
}

__global__ void global_full_kernel(volatile unsigned *data, unsigned rounds) {
  const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
  for (unsigned round = 0; round < rounds; ++round) {
    const unsigned value = data[i];
    data[i] = value + 1;
  }
}

__global__ void lds_lifecycle_kernel(unsigned *output) {
  if (threadIdx.x == 0)
    output[blockIdx.x] = blockIdx.x + 1;
}

__global__ void barrier_only_kernel(unsigned *output, unsigned rounds) {
  unsigned value = threadIdx.x;
  for (unsigned round = 0; round < rounds; ++round) {
    value += round & 1u;
    __syncthreads();
  }
  if (threadIdx.x == 0)
    output[0] = value;
}

__global__ void lds_only_kernel(unsigned *output, unsigned rounds) {
  __shared__ volatile unsigned shared[kThreads];
  unsigned value = threadIdx.x;
  for (unsigned round = 0; round < rounds; ++round) {
    value = threadIdx.x + round;
    shared[threadIdx.x] = value;
    asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
  }
  output[threadIdx.x] = value;
}

__global__ void lds_barrier_kernel(unsigned *output, unsigned rounds) {
  __shared__ volatile unsigned shared[kThreads];
  unsigned value = threadIdx.x;
  for (unsigned round = 0; round < rounds; ++round) {
    shared[threadIdx.x] = value + round;
    __syncthreads();
    value = shared[threadIdx.x];
    __syncthreads();
  }
  output[threadIdx.x] = value;
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
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s <mode> <rounds>\n", argv[0]);
    return 2;
  }
  const char *mode = argv[1];
  const unsigned rounds = std::strtoul(argv[2], nullptr, 0);
  double milliseconds = 0;
  bool ok = true;

  if (!std::strcmp(mode, "global_gate") || !std::strcmp(mode, "global_full")) {
    std::vector<unsigned> host(kElements, 1);
    unsigned *data = nullptr;
    CHECK(hipMalloc(&data, host.size() * sizeof(unsigned)));
    CHECK(hipMemcpy(data, host.data(), host.size() * sizeof(unsigned), hipMemcpyHostToDevice));
    if (!std::strcmp(mode, "global_gate")) {
      milliseconds = timed([&] { global_gate_kernel<<<1, kThreads>>>(data, rounds); });
    } else {
      milliseconds =
          timed([&] { global_full_kernel<<<kElements / kThreads, kThreads>>>(data, rounds); });
    }
    CHECK(hipMemcpy(host.data(), data, host.size() * sizeof(unsigned), hipMemcpyDeviceToHost));
    for (const unsigned value : host)
      ok = ok && value == rounds + 1;
    CHECK(hipFree(data));
  } else {
    std::vector<unsigned> host(kThreads);
    unsigned *output = nullptr;
    CHECK(hipMalloc(&output, host.size() * sizeof(unsigned)));
    if (!std::strcmp(mode, "lds_lifecycle")) {
      milliseconds = timed([&] {
        for (unsigned launch = 0; launch < rounds; ++launch)
          lds_lifecycle_kernel<<<1, kThreads, 4>>>(output);
      });
      CHECK(hipMemcpy(host.data(), output, sizeof(unsigned), hipMemcpyDeviceToHost));
      ok = host[0] == 1;
    } else if (!std::strcmp(mode, "barrier_only")) {
      milliseconds = timed([&] { barrier_only_kernel<<<1, kThreads, 4>>>(output, rounds); });
      CHECK(hipMemcpy(host.data(), output, sizeof(unsigned), hipMemcpyDeviceToHost));
      ok = host[0] == rounds / 2;
    } else if (!std::strcmp(mode, "lds_only")) {
      milliseconds = timed([&] { lds_only_kernel<<<1, kThreads>>>(output, rounds); });
      CHECK(hipMemcpy(host.data(), output, host.size() * sizeof(unsigned), hipMemcpyDeviceToHost));
      for (unsigned lane = 0; lane < kThreads; ++lane)
        ok = ok && host[lane] == lane + rounds - 1;
    } else if (!std::strcmp(mode, "lds_barrier")) {
      milliseconds = timed([&] { lds_barrier_kernel<<<1, kThreads>>>(output, rounds); });
      CHECK(hipMemcpy(host.data(), output, host.size() * sizeof(unsigned), hipMemcpyDeviceToHost));
      for (unsigned lane = 0; lane < kThreads; ++lane)
        ok = ok && host[lane] == lane + rounds * (rounds - 1) / 2;
    } else {
      std::fprintf(stderr, "unknown mode: %s\n", mode);
      return 2;
    }
    CHECK(hipFree(output));
  }

  std::printf("kernel_ms=%.3f\n", milliseconds);
  std::fprintf(stderr, "%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
