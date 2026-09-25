// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Manual diagnostic for the asynchronous completion blind spot documented in
// docs/consan/validation/HIPKITTENS_CDNA4_ANALYSIS.md. The broken member is
// intentionally incorrect; its numerical failure is the native witness.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <vector>
#define HIP(call)                                                                                  \
  do {                                                                                             \
    auto e = (call);                                                                               \
    if (e != hipSuccess) {                                                                         \
      fprintf(stderr, "%s: %s\n", #call, hipGetErrorString(e));                                    \
      exit(2);                                                                                     \
    }                                                                                              \
  } while (0)
template <bool WaitBeforeBarrier>
__global__ __launch_bounds__(128) void async_completion_repro(const uint32_t *input,
                                                              uint32_t *result) {
  __shared__ uint32_t tile[64];
  uint32_t tid = threadIdx.x, lane = tid % 64;
  if (tid < 64)
    tile[lane] = 0;
  __syncthreads();
  uintptr_t ptr = reinterpret_cast<uintptr_t>(input + blockIdx.x * 4096);
  uint32_t d0 = ptr, d1 = (ptr >> 32) & 0xffff, d2 = 256, d3 = 4u | (7u << 12) | (4u << 15);
  uint32_t base =
      __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(reinterpret_cast<uintptr_t>(tile)));
  uint32_t wave = __builtin_amdgcn_readfirstlane(tid / 64), offset = lane * 4, addr = base + offset,
           value;
  asm volatile("v_mov_b32 %0, 0\n"
               "s_cmp_eq_u32 %1, 0\n"
               "s_cbranch_scc1 1f\n"
               "s_mov_b32 s12, %2\n"
               "s_mov_b32 s13, %3\n"
               "s_mov_b32 s14, %4\n"
               "s_mov_b32 s15, %5\n"
               "s_mov_b32 m0, %6\n"
               "buffer_load_dword %7, s[12:15], 0 offen lds\n"
               "s_cmp_eq_u32 %9, 0\n"
               "s_cbranch_scc1 2f\n"
               "s_waitcnt vmcnt(0)\n"
               "2: s_barrier\n"
               "s_waitcnt vmcnt(0)\n"
               "s_barrier\n"
               "s_branch 3f\n"
               "1: s_barrier\n"
               "ds_read_b32 %0, %8\n"
               "s_waitcnt lgkmcnt(0)\n"
               "s_barrier\n"
               "3: s_nop 0\n"
               : "=&v"(value)
               : "s"(wave), "s"(d0), "s"(d1), "s"(d2), "s"(d3), "s"(base), "v"(offset), "v"(addr),
                 "s"(uint32_t(WaitBeforeBarrier))
               : "s12", "s13", "s14", "s15", "scc", "memory");
  if (tid < 64)
    result[blockIdx.x * 64 + lane] = value;
}
int main(int argc, char **argv) {
  bool correct = argc > 1 && atoi(argv[1]);
  int blocks = argc > 2 ? atoi(argv[2]) : 256;
  std::vector<uint32_t> input(blocks * 4096), output(blocks * 64);
  for (size_t i = 0; i < input.size(); ++i)
    input[i] = 0x12340000 + (i % 4096);
  uint32_t *a, *b;
  HIP(hipMalloc(&a, input.size() * 4));
  HIP(hipMalloc(&b, output.size() * 4));
  HIP(hipMemcpy(a, input.data(), input.size() * 4, hipMemcpyHostToDevice));
  HIP(hipMemset(b, 0, output.size() * 4));
  if (correct)
    async_completion_repro<true><<<blocks, 128>>>(a, b);
  else
    async_completion_repro<false><<<blocks, 128>>>(a, b);
  HIP(hipGetLastError());
  HIP(hipDeviceSynchronize());
  HIP(hipMemcpy(output.data(), b, output.size() * 4, hipMemcpyDeviceToHost));
  unsigned mismatch = 0;
  for (size_t i = 0; i < output.size(); ++i)
    if (output[i] != 0x12340000 + i % 64)
      ++mismatch;
  printf("wait_before_barrier=%d blocks=%d mismatches=%u/%zu\n", correct, blocks, mismatch,
         output.size());
  HIP(hipFree(a));
  HIP(hipFree(b));
  return mismatch ? 1 : 0;
}
