// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using I8 = int __attribute__((ext_vector_type(8)));
using I16 = int __attribute__((ext_vector_type(16)));
using F8 = float __attribute__((ext_vector_type(8)));
using F16 = float __attribute__((ext_vector_type(16)));

#ifndef MATRIX_CHAINS
#define MATRIX_CHAINS 4
#endif
constexpr int kChains = MATRIX_CHAINS;

static void check(hipError_t status) {
  if (status != hipSuccess) {
    std::fprintf(stderr, "HIP: %s\n", hipGetErrorString(status));
    std::exit(1);
  }
}

template <int Shape> struct Matrix;
template <> struct Matrix<64> {
  using A = I8;
  using B = I8;
  using C = F8;
  static constexpr int k = 64, a_regs = 8, b_regs = 8, c_regs = 8;
  static constexpr int ones = 0x38383838;
  __device__ static C multiply(A a, B b, C c) {
    return __builtin_amdgcn_wmma_f32_16x16x64_fp8_fp8(a, b, 0, c, false, false);
  }
};
template <> struct Matrix<128> {
  using A = I16;
  using B = I16;
  using C = F8;
  static constexpr int k = 128, a_regs = 16, b_regs = 16, c_regs = 8;
  static constexpr int ones = 0x38383838;
  __device__ static C multiply(A a, B b, C c) {
    return __builtin_amdgcn_wmma_f32_16x16x128_fp8_fp8(a, b, 0, c, false, false);
  }
};
template <> struct Matrix<32> {
  using A = I16;
  using B = I8;
  using C = F16;
  static constexpr int k = 128, a_regs = 16, b_regs = 8, c_regs = 16;
  static constexpr int ones = 0x22222222;
  __device__ static C multiply(A a, B b, C c) {
    return __builtin_amdgcn_wmma_f32_32x16x128_f4(a, b, 0, c);
  }
};

template <int Shape, bool Dependent>
__global__ void matrix_chains(const int *input, float *output, int iterations) {
  using M = Matrix<Shape>;
  typename M::A a;
  typename M::B b;
  for (int i = 0; i < M::a_regs; ++i)
    a[i] = input[i * 32 + threadIdx.x];
  for (int i = 0; i < M::b_regs; ++i)
    b[i] = input[512 + i * 32 + threadIdx.x];
  typename M::C accumulators[kChains];
  for (int chain = 0; chain != kChains; ++chain)
    for (int i = 0; i < M::c_regs; ++i)
      accumulators[chain][i] = static_cast<float>(blockIdx.x % 7 + chain);
      // Keep the loop and accumulators visible in disassembly. Each independent
      // chain depends on its own preceding iteration, so arithmetic cannot be hoisted.
#pragma clang loop unroll(disable)
  for (int i = 0; i < iterations; ++i) {
#pragma unroll
    for (int chain = 0; chain != kChains; ++chain)
      accumulators[chain] =
          M::multiply(a, b, accumulators[Dependent ? (chain + kChains - 1) % kChains : chain]);
  }
  for (int chain = 0; chain != kChains; ++chain)
    for (int reg = 0; reg != M::c_regs; ++reg)
      output[((blockIdx.x * kChains + chain) * M::c_regs + reg) * 32 + threadIdx.x] =
          accumulators[chain][reg];
}

template <int Shape, bool Dependent> int run(int blocks, int iterations, int repeats) {
  using M = Matrix<Shape>;
  std::vector<int> input(1024, M::ones);
  std::vector<float> output(static_cast<size_t>(blocks) * kChains * M::c_regs * 32);
  int *device_input;
  float *device_output;
  check(hipMalloc(&device_input, input.size() * sizeof(int)));
  check(hipMalloc(&device_output, output.size() * sizeof(float)));
  check(hipMemcpy(device_input, input.data(), input.size() * sizeof(int), hipMemcpyHostToDevice));
  const auto begin = std::chrono::steady_clock::now();
  for (int i = 0; i != repeats; ++i) {
    hipLaunchKernelGGL((matrix_chains<Shape, Dependent>), dim3(blocks), dim3(32), 0, 0,
                       device_input, device_output, iterations);
    check(hipGetLastError());
    check(hipDeviceSynchronize());
  }
  const auto end = std::chrono::steady_clock::now();
  check(hipMemcpy(output.data(), device_output, output.size() * sizeof(float),
                  hipMemcpyDeviceToHost));
  for (size_t i = 0; i != output.size(); ++i) {
    const size_t block = i / (kChains * M::c_regs * 32);
    const size_t chain = (i / (M::c_regs * 32)) % kChains;
    const float expected = Dependent
                               ? static_cast<float>(block % 7 + kChains - 1 +
                                                    ((iterations - 1) * kChains + chain + 1) * M::k)
                               : static_cast<float>(block % 7 + chain + iterations * M::k);
    if (output[i] != expected) {
      std::fprintf(stderr, "Mismatch %zu: got %.9g expected %.9g\n", i, output[i], expected);
      return 2;
    }
  }
  std::printf("PASS shape=%d dependent=%d blocks=%d iterations=%d repeats=%d matrix_ops=%lld "
              "host_dispatch_seconds=%.9f\n",
              Shape, Dependent, blocks, iterations, repeats,
              static_cast<long long>(blocks) * iterations * repeats * kChains,
              std::chrono::duration<double>(end - begin).count());
  check(hipFree(device_input));
  check(hipFree(device_output));
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 6) {
    std::fprintf(stderr, "Usage: %s SHAPE(64,128,32) BLOCKS ITERATIONS REPEATS DEPENDENT(0,1)\n",
                 argv[0]);
    return 1;
  }
  const int shape = std::atoi(argv[1]), blocks = std::atoi(argv[2]);
  const int iterations = std::atoi(argv[3]), repeats = std::atoi(argv[4]);
  const bool dependent = std::atoi(argv[5]);
  if (blocks <= 0 || iterations <= 0 || repeats <= 0)
    return 1;
#define RUN(S)                                                                                     \
  case S:                                                                                          \
    return dependent ? run<S, true>(blocks, iterations, repeats)                                   \
                     : run<S, false>(blocks, iterations, repeats)
  switch (shape) {
    RUN(64);
    RUN(128);
    RUN(32);
  default:
    return 1;
  }
#undef RUN
}
