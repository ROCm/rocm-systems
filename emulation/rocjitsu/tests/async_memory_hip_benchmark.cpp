// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

using U4 = uint32_t __attribute__((ext_vector_type(4)));
using GlobalU4 = U4 __attribute__((address_space(1)));

static void check(hipError_t status) {
  if (status != hipSuccess) {
    std::fprintf(stderr, "HIP: %s\n", hipGetErrorString(status));
    std::exit(1);
  }
}

__host__ __device__ uint32_t arithmetic(uint32_t value) {
  return ((value << 5) | (value >> 27)) ^ 0x9e3779b9u;
}

// A global vector load followed by independent arithmetic and a dependent
// consumer. Each lane owns its input and output, including store-to-load cases.
template <int Work, bool StoreLoad>
__global__ void memory_overlap(const U4 *input, U4 *scratch, uint32_t *output, int iterations) {
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t value = tid;
  auto *global_input = (const GlobalU4 *)input;
  auto *global_scratch = (GlobalU4 *)scratch;
#pragma clang loop unroll(disable)
  for (int i = 0; i != iterations; ++i) {
    asm volatile("" ::: "memory"); // Keep a distinct access in every iteration.
    if constexpr (StoreLoad) {
      global_scratch[tid] = U4{value, value + 1, value + 2, value + 3};
      asm volatile("" ::: "memory");
    }
    U4 loaded = (StoreLoad ? global_scratch : global_input)[tid];
#pragma unroll
    for (int j = 0; j != Work; ++j)
      value = arithmetic(value);
    asm volatile("" : "+v"(loaded), "+v"(value));
    value += loaded[0] + loaded[1] + loaded[2] + loaded[3];
  }
  output[tid] = value;
}

template <int Work, bool StoreLoad> int run(unsigned blocks, int iterations) {
  const unsigned count = blocks * 32;
  std::vector<U4> input(count);
  std::vector<uint32_t> output(count);
  for (unsigned i = 0; i != count; ++i)
    input[i] = U4{i, i + 1, i + 2, i + 3};
  U4 *device_input, *scratch;
  uint32_t *device_output;
  check(hipMalloc(&device_input, count * sizeof(U4)));
  check(hipMalloc(&scratch, count * sizeof(U4)));
  check(hipMalloc(&device_output, count * sizeof(uint32_t)));
  check(hipMemcpy(device_input, input.data(), count * sizeof(U4), hipMemcpyHostToDevice));
  hipLaunchKernelGGL((memory_overlap<Work, StoreLoad>), dim3(blocks), dim3(32), 0, 0, device_input,
                     scratch, device_output, iterations);
  check(hipGetLastError());
  check(hipDeviceSynchronize());
  check(hipMemcpy(output.data(), device_output, count * sizeof(uint32_t), hipMemcpyDeviceToHost));
  for (unsigned tid = 0; tid != count; ++tid) {
    uint32_t expected = tid;
    for (int i = 0; i != iterations; ++i) {
      uint32_t loaded = 4 * (StoreLoad ? expected : tid) + 6;
      for (int j = 0; j != Work; ++j)
        expected = arithmetic(expected);
      expected += loaded;
    }
    if (output[tid] != expected) {
      std::fprintf(stderr, "FAIL lane=%u got=%u expected=%u\n", tid, output[tid], expected);
      return 1;
    }
  }
  std::printf("PASS memory work=%d store_load=%d blocks=%u iterations=%d\n", Work, StoreLoad,
              blocks, iterations);
  check(hipFree(device_output));
  check(hipFree(scratch));
  check(hipFree(device_input));
  return 0;
}

int main(int argc, char **argv) {
  if (argc != 5) {
    std::fprintf(stderr, "usage: %s WORK(0|32) STORE_LOAD(0|1) BLOCKS ITERATIONS\n", argv[0]);
    return 2;
  }
  const int work = std::atoi(argv[1]), store_load = std::atoi(argv[2]);
  const int blocks = std::atoi(argv[3]), iterations = std::atoi(argv[4]);
  if ((work != 0 && work != 32) || (store_load != 0 && store_load != 1) || blocks <= 0 ||
      iterations <= 0)
    return 2;
  if (work == 0)
    return store_load ? run<0, true>(blocks, iterations) : run<0, false>(blocks, iterations);
  return store_load ? run<32, true>(blocks, iterations) : run<32, false>(blocks, iterations);
}
