// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hip_ds_transpose_acc_test.cpp
/// @brief Verify ds_read_b64_tr_b16 with acc=1 writes to AccVGPR, not VGPR.

#include <cstdint>
#include <hip/hip_runtime.h>
#include <vector>

#include <gtest/gtest.h>

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  int rc = RUN_ALL_TESTS();
  (void)hipDeviceReset();
  return rc;
}

#define HIP_ASSERT(call)                                                                           \
  do {                                                                                             \
    hipError_t err = (call);                                                                       \
    ASSERT_EQ(err, hipSuccess) << "HIP error: " << hipGetErrorString(err);                         \
  } while (0)

// out[0..63]   = v2 per lane (should be 0xC0DE if ds_read went to AccVGPR)
// out[64..127] = a2 per lane (should differ from 0xF00D if ds_read wrote here)
__global__ void ds_transpose_acc_kernel(uint32_t *out) {
  __shared__ uint32_t lds[128];
  int tid = threadIdx.x;

  lds[tid] = 0xCAFE0000u | tid;
  if (tid + 64 < 128)
    lds[tid + 64] = 0xBEEF0000u | (tid + 64);
  __syncthreads();

  int lds_addr = static_cast<int>(reinterpret_cast<uintptr_t>(lds) & 0xFFFF);

  uint32_t v2_val, a2_val;
  uint32_t v2_init = 0xC0DEu;
  uint32_t a2_init = 0xF00Du;

  asm volatile("v_accvgpr_write_b32 a2, %2\n"
               "v_accvgpr_write_b32 a3, %2\n"
               "v_mov_b32 v2, %3\n"
               "v_mov_b32 v3, %3\n"
               "ds_read_b64_tr_b16 a[2:3], %4\n"
               "s_waitcnt lgkmcnt(0)\n"
               "v_mov_b32 %0, v2\n"
               "v_accvgpr_read_b32 %1, a2\n"
               : "=v"(v2_val), "=v"(a2_val)
               : "v"(a2_init), "v"(v2_init), "v"(lds_addr)
               : "v2", "v3", "a2", "a3");

  out[tid] = v2_val;
  out[64 + tid] = a2_val;
}

TEST(DsTransposeAccTest, AccBitWritesToAccVGPR) {
  constexpr int N = 128;
  uint32_t *d_out = nullptr;
  HIP_ASSERT(hipMalloc(&d_out, N * sizeof(uint32_t)));
  HIP_ASSERT(hipMemset(d_out, 0, N * sizeof(uint32_t)));

  ds_transpose_acc_kernel<<<1, 64>>>(d_out);
  HIP_ASSERT(hipDeviceSynchronize());

  std::vector<uint32_t> h(N);
  HIP_ASSERT(hipMemcpy(h.data(), d_out, N * sizeof(uint32_t), hipMemcpyDeviceToHost));
  (void)hipFree(d_out);

  // v2 should be 0xC0DE for all lanes — the ds_read with acc=1 should not
  // have touched VGPR v2.
  int v2_ok = 0;
  for (int i = 0; i < 64; ++i)
    if (h[i] == 0xC0DEu)
      v2_ok++;
  EXPECT_EQ(v2_ok, 64) << "VGPR v2 was clobbered by ds_read — acc bit not respected";

  // a2 should have changed from the initial 0xF00D for at least some lanes
  // (the transpose shuffle may produce 0xF00D for a few lanes by coincidence,
  // but not for all 64).
  int a2_unchanged = 0;
  for (int i = 0; i < 64; ++i)
    if (h[64 + i] == 0xF00Du)
      a2_unchanged++;
  EXPECT_LT(a2_unchanged, 64) << "AccVGPR a2 was never written by ds_read";
}
