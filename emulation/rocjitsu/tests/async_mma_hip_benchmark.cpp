// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <vector>

#ifndef MMA_TARGET
#error Compile with MMA_TARGET=1250, 1201, 950, or 942.
#endif
using I2 = int __attribute__((ext_vector_type(2)));
using I4 = int __attribute__((ext_vector_type(4)));
using I8 = int __attribute__((ext_vector_type(8)));
using H4 = _Float16 __attribute__((ext_vector_type(4)));
using H8 = _Float16 __attribute__((ext_vector_type(8)));
using H16 = _Float16 __attribute__((ext_vector_type(16)));
using F4 = float __attribute__((ext_vector_type(4)));
using F8 = float __attribute__((ext_vector_type(8)));
using F16 = float __attribute__((ext_vector_type(16)));
using F32 = float __attribute__((ext_vector_type(32)));
constexpr int kWave = MMA_TARGET == 942 || MMA_TARGET == 950 ? 64 : 32;
constexpr int kChains = 4;

static void check(hipError_t status) {
  if (status != hipSuccess) {
    std::fprintf(stderr, "HIP: %s\n", hipGetErrorString(status));
    std::exit(1);
  }
}

template <int Shape> struct Matrix;
#if MMA_TARGET == 1250
template <> struct Matrix<32> {
  using A = I8;
  using C = F8;
  static constexpr int k = 32, ones = 0x3c003c00;
  __device__ static C multiply(A a, A b, C c, int, int) {
    return __builtin_amdgcn_wmma_f32_16x16x32_f16(false, __builtin_bit_cast(H16, a), false,
                                                  __builtin_bit_cast(H16, b), 0, c, false, false);
  }
};
#elif MMA_TARGET == 1201
template <> struct Matrix<16> {
  using A = I4;
  using C = F8;
  static constexpr int k = 16, ones = 0x3c003c00;
  __device__ static C multiply(A a, A b, C c, int, int) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(__builtin_bit_cast(H8, a),
                                                            __builtin_bit_cast(H8, b), c);
  }
};
#else
#define HALF_MFMA(SHAPE, K, RESULT, BUILTIN)                                                       \
  template <> struct Matrix<SHAPE> {                                                               \
    using A = I2;                                                                                  \
    using C = RESULT;                                                                              \
    static constexpr int k = K, ones = 0x3c003c00;                                                 \
    __device__ static C multiply(A a, A b, C c, int, int) {                                        \
      return BUILTIN(__builtin_bit_cast(H4, a), __builtin_bit_cast(H4, b), c, 0, 0, 0);            \
    }                                                                                              \
  }
HALF_MFMA(2, 4, F32, __builtin_amdgcn_mfma_f32_32x32x4f16);
HALF_MFMA(4, 4, F16, __builtin_amdgcn_mfma_f32_16x16x4f16);
HALF_MFMA(16, 4, F4, __builtin_amdgcn_mfma_f32_4x4x4f16);
HALF_MFMA(8, 8, F16, __builtin_amdgcn_mfma_f32_32x32x8f16);
HALF_MFMA(116, 16, F4, __builtin_amdgcn_mfma_f32_16x16x16f16);
#undef HALF_MFMA
#if MMA_TARGET == 950
#define SCALED_MFMA(SHAPE, K, RESULT, BUILTIN)                                                     \
  template <> struct Matrix<SHAPE> {                                                               \
    using A = I8;                                                                                  \
    using C = RESULT;                                                                              \
    static constexpr int k = K, ones = 0x22222222;                                                 \
    __device__ static C multiply(A a, A b, C c, int sa, int sb) {                                  \
      return BUILTIN(a, b, c, 4, 4, 0, sa, 0, sb);                                                 \
    }                                                                                              \
  }
SCALED_MFMA(128, 128, F4, __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4);
SCALED_MFMA(64, 64, F16, __builtin_amdgcn_mfma_scale_f32_32x32x64_f8f6f4);
#undef SCALED_MFMA
#endif
#endif

template <int Shape, bool Dependent>
__global__ __launch_bounds__(kWave) void matrix_chains(const int *input, float *output,
                                                       int iterations) {
  using M = Matrix<Shape>;
  constexpr int a_regs = sizeof(typename M::A) / sizeof(int);
  constexpr int c_regs = sizeof(typename M::C) / sizeof(float);
  typename M::A a, b;
  for (int r = 0; r != a_regs; ++r) {
    a[r] = input[r * kWave + threadIdx.x];
    b[r] = input[8 * kWave + r * kWave + threadIdx.x];
  }
  const int sa = input[16 * kWave + threadIdx.x];
  const int sb = input[17 * kWave + threadIdx.x];
  typename M::C accumulators[kChains];
  for (int chain = 0; chain != kChains; ++chain)
    for (int r = 0; r != c_regs; ++r)
      accumulators[chain][r] = float(blockIdx.x % 7 + chain);
#pragma clang loop unroll(disable)
  for (int i = 0; i < iterations; ++i) {
#pragma unroll
    for (int chain = 0; chain != kChains; ++chain)
      accumulators[chain] = M::multiply(
          a, b, accumulators[Dependent ? (chain + kChains - 1) % kChains : chain], sa, sb);
  }
  for (int chain = 0; chain != kChains; ++chain)
    for (int r = 0; r != c_regs; ++r)
      output[((blockIdx.x * kChains + chain) * c_regs + r) * kWave + threadIdx.x] =
          accumulators[chain][r];
}

template <int Shape, bool Dependent> int run(unsigned blocks, int iterations) {
  using M = Matrix<Shape>;
  constexpr int c_regs = sizeof(typename M::C) / sizeof(float);
  std::vector<int> input(18 * kWave, M::ones);
  for (int i = 16 * kWave; i != 18 * kWave; ++i)
    input[i] = 0x7f7f7f7f; // E8M0 scale 1 in every byte.
  std::vector<float> output(size_t(blocks) * kChains * c_regs * kWave);
  int *device_input;
  float *device_output;
  check(hipMalloc(&device_input, input.size() * sizeof(int)));
  check(hipMalloc(&device_output, output.size() * sizeof(float)));
  check(hipMemcpy(device_input, input.data(), input.size() * sizeof(int), hipMemcpyHostToDevice));
  hipLaunchKernelGGL((matrix_chains<Shape, Dependent>), dim3(blocks), dim3(kWave), 0, 0,
                     device_input, device_output, iterations);
  check(hipGetLastError());
  check(hipDeviceSynchronize());
  check(hipMemcpy(output.data(), device_output, output.size() * sizeof(float),
                  hipMemcpyDeviceToHost));
  for (size_t i = 0; i != output.size(); ++i) {
    const size_t block = i / (kChains * c_regs * kWave);
    const size_t chain = i / (c_regs * kWave) % kChains;
    const float expected =
        Dependent ? float(block % 7 + kChains - 1 + ((iterations - 1) * kChains + chain + 1) * M::k)
                  : float(block % 7 + chain + iterations * M::k);
    if (output[i] != expected) {
      std::fprintf(stderr, "FAIL element=%zu got=%.9g expected=%.9g\n", i, output[i], expected);
      return 1;
    }
  }
  std::printf("PASS target=%d shape=%d dependent=%d blocks=%u iterations=%d instructions=%llu\n",
              MMA_TARGET, Shape, Dependent, blocks, iterations,
              static_cast<unsigned long long>(blocks) * iterations * kChains);
  check(hipFree(device_input));
  check(hipFree(device_output));
  return 0;
}
int main(int argc, char **argv) {
  if (argc != 5) {
    std::fprintf(stderr, "usage: %s SHAPE BLOCKS ITERATIONS DEPENDENT(0|1)\n", argv[0]);
    return 2;
  }
  const int shape = std::atoi(argv[1]), blocks = std::atoi(argv[2]);
  const int iterations = std::atoi(argv[3]), dependent = std::atoi(argv[4]);
  if (blocks <= 0 || iterations <= 0 || (dependent != 0 && dependent != 1))
    return 2;
#define RUN(S)                                                                                     \
  case S:                                                                                          \
    return dependent ? run<S, true>(blocks, iterations) : run<S, false>(blocks, iterations)
  switch (shape) {
#if MMA_TARGET == 1250
    RUN(32);
#elif MMA_TARGET == 1201
    RUN(16);
#else
    RUN(2);
    RUN(4);
    RUN(16);
    RUN(8);
    RUN(116);
#if MMA_TARGET == 950
    RUN(128);
    RUN(64);
#endif
#endif
  default:
    return 2;
  }
#undef RUN
}
