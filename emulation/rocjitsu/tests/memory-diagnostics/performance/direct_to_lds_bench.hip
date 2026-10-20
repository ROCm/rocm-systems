// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file direct_to_lds_bench.hip
/// @brief Focused direct-to-LDS source-family performance workload.
///
/// Usage: direct_to_lds_bench <rounds>

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(call)                                                                                \
  do {                                                                                             \
    const hipError_t error_ = (call);                                                              \
    if (error_ != hipSuccess) {                                                                    \
      std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(error_));           \
      std::exit(2);                                                                                \
    }                                                                                              \
  } while (0)

constexpr unsigned kThreads = 128;
constexpr unsigned kBlocks = 64;

__attribute__((amdgpu_flat_work_group_size(kThreads, kThreads))) __global__ void
direct_to_lds_kernel(const int *source, int *output, unsigned rounds) {
  __shared__ volatile int shared[kThreads];
  const int tid = threadIdx.x;
  const int lane = tid % 64;
  const int wave = tid / 64;
  const int *workgroup_source = source + blockIdx.x * blockDim.x;
  const uintptr_t base = reinterpret_cast<uintptr_t>(workgroup_source);
  const unsigned descriptor_word0 = static_cast<unsigned>(base);
  const unsigned descriptor_word1 = static_cast<unsigned>(base >> 32) & 0xFFFFu;
  const unsigned descriptor_word2 = kThreads * sizeof(int);
  const unsigned descriptor_word3 = (4u) | (7u << 12) | (4u << 15);
  const unsigned wave_byte_offset = __builtin_amdgcn_readfirstlane(wave * 64 * sizeof(int));

  int value = 0;
  for (unsigned round = 0; round < rounds; ++round) {
    asm volatile("s_mov_b32 s12, %0\n"
                 "s_mov_b32 s13, %1\n"
                 "s_mov_b32 s14, %2\n"
                 "s_mov_b32 s15, %3\n"
                 "s_mov_b32 m0, %5\n"
                 "buffer_load_dword %4, s[12:15], %6 offen lds\n"
                 "s_waitcnt vmcnt(0)\n"
                 :
                 : "s"(descriptor_word0), "s"(descriptor_word1), "s"(descriptor_word2),
                   "s"(descriptor_word3), "v"(lane * 4), "s"(wave_byte_offset),
                   "s"(wave_byte_offset)
                 : "s12", "s13", "s14", "s15", "memory");
    value += shared[tid];
  }
  output[blockIdx.x * blockDim.x + tid] = value;
}

int main(int argc, char **argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s <rounds>\n", argv[0]);
    return 2;
  }
  const unsigned rounds = std::strtoul(argv[1], nullptr, 0);
  constexpr size_t kElements = kBlocks * kThreads;
  std::vector<int> host(kElements);
  for (size_t i = 0; i < kElements; ++i)
    host[i] = static_cast<int>(i % 97);

  int *source = nullptr;
  int *output = nullptr;
  CHECK(hipMalloc(&source, kElements * sizeof(int)));
  CHECK(hipMalloc(&output, kElements * sizeof(int)));
  CHECK(hipMemcpy(source, host.data(), kElements * sizeof(int), hipMemcpyHostToDevice));
  CHECK(hipDeviceSynchronize());
  const auto start = std::chrono::steady_clock::now();
  direct_to_lds_kernel<<<kBlocks, kThreads>>>(source, output, rounds);
  CHECK(hipGetLastError());
  CHECK(hipDeviceSynchronize());
  const double milliseconds =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
  CHECK(hipMemcpy(host.data(), output, kElements * sizeof(int), hipMemcpyDeviceToHost));

  bool ok = true;
  for (size_t i = 0; i < kElements; ++i)
    ok = ok && host[i] == static_cast<int>(i % 97) * static_cast<int>(rounds);
  CHECK(hipFree(source));
  CHECK(hipFree(output));
  std::printf("kernel_ms=%.3f\n", milliseconds);
  std::fprintf(stderr, "%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
