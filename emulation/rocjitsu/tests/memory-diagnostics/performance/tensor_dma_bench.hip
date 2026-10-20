// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file tensor_dma_bench.hip
/// @brief Focused tensor-DMA source-family performance workload.
///
/// Usage: tensor_dma_bench <rounds>

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
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

typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
constexpr unsigned kThreads = 32;
constexpr unsigned kBlocks = 64;

__global__ void tensor_source_kernel(const v4i *__restrict__ descriptor_group0,
                                     const v8i *__restrict__ descriptor_group1,
                                     int *__restrict__ output, unsigned rounds) {
  __shared__ int shared[kThreads];
  shared[threadIdx.x] = 0;
  __syncthreads();
  const v4i group0 = descriptor_group0[0];
  const v8i group1 = descriptor_group1[0];
  const v4i zero4 = {0, 0, 0, 0};
  int value = 0;
  for (unsigned round = 0; round < rounds; ++round) {
    __builtin_amdgcn_tensor_load_to_lds(group0, group1, zero4, zero4, 0);
    asm volatile("s_wait_tensorcnt 0" ::: "memory");
    value += shared[threadIdx.x];
  }
  output[blockIdx.x * blockDim.x + threadIdx.x] = value;
}

void build_tensor_descriptor(uint32_t *descriptor, uint64_t global_address) {
  std::memset(descriptor, 0, 20 * sizeof(uint32_t));
  descriptor[0] = 1;
  descriptor[1] = 0;
  descriptor[2] = static_cast<uint32_t>(global_address);
  descriptor[3] = static_cast<uint32_t>(global_address >> 32) & 0x1FFFFFFu;
  descriptor[3] |= 2u << 30;
  descriptor[4] = 2u << 16;
  descriptor[5] = kThreads << 16;
  descriptor[6] = 1u << 16;
  descriptor[7] = kThreads << 16;
  descriptor[8] = 1;
  descriptor[9] = kThreads;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <rounds>\n", argv[0]);
    return 2;
  }
  const unsigned rounds = std::strtoul(argv[1], nullptr, 0);
  std::vector<int> tensor(kThreads);
  for (unsigned i = 0; i < kThreads; ++i)
    tensor[i] = static_cast<int>(i + 1);

  int *device_tensor = nullptr;
  CHECK(hipMalloc(&device_tensor, tensor.size() * sizeof(int)));
  CHECK(
      hipMemcpy(device_tensor, tensor.data(), tensor.size() * sizeof(int), hipMemcpyHostToDevice));

  uint32_t descriptor[20];
  build_tensor_descriptor(descriptor, reinterpret_cast<uint64_t>(device_tensor));
  v4i *device_group0 = nullptr;
  v8i *device_group1 = nullptr;
  CHECK(hipMalloc(&device_group0, sizeof(v4i)));
  CHECK(hipMalloc(&device_group1, sizeof(v8i)));
  CHECK(hipMemcpy(device_group0, &descriptor[0], sizeof(v4i), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(device_group1, &descriptor[4], sizeof(v8i), hipMemcpyHostToDevice));

  constexpr size_t kOutputs = kBlocks * kThreads;
  int *device_output = nullptr;
  CHECK(hipMalloc(&device_output, kOutputs * sizeof(int)));

  CHECK(hipDeviceSynchronize());
  const auto start = std::chrono::steady_clock::now();
  tensor_source_kernel<<<kBlocks, kThreads>>>(device_group0, device_group1, device_output, rounds);
  CHECK(hipGetLastError());
  CHECK(hipDeviceSynchronize());
  const double milliseconds =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();

  std::vector<int> output(kOutputs);
  CHECK(
      hipMemcpy(output.data(), device_output, output.size() * sizeof(int), hipMemcpyDeviceToHost));

  bool ok = true;
  for (size_t i = 0; i < output.size(); ++i) {
    const int expected = tensor[i % kThreads] * static_cast<int>(rounds);
    if (output[i] != expected) {
      if (ok)
        std::fprintf(stderr, "mismatch[%zu]=%d expected=%d\n", i, output[i], expected);
      ok = false;
    }
  }

  CHECK(hipFree(device_tensor));
  CHECK(hipFree(device_group0));
  CHECK(hipFree(device_group1));
  CHECK(hipFree(device_output));
  std::printf("kernel_ms=%.3f\n", milliseconds);
  std::fprintf(stderr, "%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
