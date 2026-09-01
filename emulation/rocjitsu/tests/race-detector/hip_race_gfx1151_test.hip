// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hip_race_gfx1151_test.hip
/// @brief Small gfx1151 HIP race detector smoke tests.

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include "race_test_support.hpp"

#include <cstdint>
#include <vector>

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  int rc = RUN_ALL_TESTS();
  (void)hipDeviceReset();
  return rc;
}

namespace {

using rocjitsu::test::FindingCount;
using rocjitsu::test::RaceExpectation;

using Gfx1151RaceTest = rocjitsu::test::RaceTestBase;

__global__ void vgpr_no_race_kernel(const float *src, float *dst) {
  int tid = threadIdx.x;
  float val;
  asm volatile("global_load_dword %0, %1, off\n"
               "s_waitcnt vmcnt(0)\n"
               : "=v"(val)
               : "v"(&src[tid]));
  dst[tid] = val;
}

static constexpr RaceExpectation kVgprRaceExpectation{
    .findings = FindingCount::OneOrMore,
    .type = "VGPR",
};

__global__ void vgpr_race_kernel(const float *src, float *dst) {
  int tid = threadIdx.x;
  float val;
  asm volatile("global_load_dword %0, %1, off\n"
               "s_nop 0\n"
               : "=v"(val)
               : "v"(&src[tid]));
  dst[tid] = val;
}

// --- Standalone split wait instructions ---

__global__ void standalone_vmcnt_waits_for_vmem_kernel(const float *src, float *dst) {
  int tid = threadIdx.x;
  float val;
  asm volatile("global_load_dword %0, %1, off\n"
               "s_waitcnt_vmcnt null, 0\n"
               : "=v"(val)
               : "v"(&src[tid]));
  dst[tid] = val;
}

__global__ void standalone_vmcnt_does_not_wait_for_lds_kernel(int *dst) {
  __shared__ int lds[32];
  int tid = threadIdx.x;
  lds[tid] = tid + 1;
  __syncthreads();

  int val;
  asm volatile("ds_read_b32 %0, %1\n"
               "s_waitcnt_vmcnt null, 0\n"
               : "=v"(val)
               : "v"(tid * 4)
               : "memory");
  dst[tid] = val;
}

__global__ void standalone_lgkmcnt_waits_for_lds_kernel(int *dst) {
  __shared__ int lds[32];
  int tid = threadIdx.x;
  lds[tid] = tid + 1;
  __syncthreads();

  int val;
  asm volatile("ds_read_b32 %0, %1\n"
               "s_waitcnt_lgkmcnt null, 0\n"
               : "=v"(val)
               : "v"(tid * 4)
               : "memory");
  dst[tid] = val;
}

__global__ void standalone_lgkmcnt_does_not_wait_for_vmem_kernel(const float *src, float *dst) {
  int tid = threadIdx.x;
  float val;
  asm volatile("global_load_dword %0, %1, off\n"
               "s_waitcnt_lgkmcnt null, 0\n"
               : "=v"(val)
               : "v"(&src[tid]));
  dst[tid] = val;
}

// D16 loads write only one half of the destination dword on RDNA. This kernel
// initializes the high half, performs a low-half global_load_d16_u8, waits for
// vmcnt, and then stores the merged dword. The test below checks two things:
// rocjitsu preserves the untouched high half, and the internal merge read used
// during memory completion is not reported as an instruction-visible VGPR read.
__global__ void d16_global_load_merge_kernel(const uint8_t *src, uint32_t *dst) {
  int tid = threadIdx.x;
  uint32_t val = 0xABCD0000u;
  asm volatile("global_load_d16_u8 %0, %1, off\n"
               "s_waitcnt vmcnt(0)\n"
               : "+v"(val)
               : "v"(&src[tid])
               : "memory");
  dst[tid] = val;
}

// High-half D16 loads have the same internal merge requirement, but preserve the
// low half and write the loaded byte into bits [31:16].
__global__ void d16_global_load_hi_merge_kernel(const uint8_t *src, uint32_t *dst) {
  int tid = threadIdx.x;
  uint32_t val = 0x0000ABCDu;
  asm volatile("global_load_d16_hi_u8 %0, %1, off\n"
               "s_waitcnt vmcnt(0)\n"
               : "+v"(val)
               : "v"(&src[tid])
               : "memory");
  dst[tid] = val;
}

// Same D16 load shape as d16_global_load_merge_kernel, but this intentionally
// omits s_waitcnt before storing the VGPR. The store consumes the load
// destination while the low-half load is still outstanding, so the race detector
// should report a real VGPR race after the memory-completion self-race is fixed.
static constexpr RaceExpectation kD16GlobalLoadMissingWaitcntRaceExpectation{
    .findings = FindingCount::OneOrMore,
    .type = "VGPR",
    .producer = "global_load_d16_u8",
};

__global__ void d16_global_load_missing_waitcnt_kernel(const uint8_t *src, uint32_t *dst) {
  int tid = threadIdx.x;
  uint32_t val = 0xABCD0000u;
  asm volatile("global_load_d16_u8 %0, %1, off\n"
               "s_nop 0\n"
               : "+v"(val)
               : "v"(&src[tid])
               : "memory");
  dst[tid] = val;
}

static constexpr RaceExpectation kWawGlobalLoadThenWmmaExpectation{
    .findings = FindingCount::OneOrMore,
    .type = "VGPR",
    .access = "write",
    .producer = "global_load",
    .consumer = "v_wmma",
};

__global__ void waw_global_load_then_wmma_kernel(const int *src, int *out) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  asm volatile("global_load_dwordx4 v[0:3], %0, off\n"
               // BUG: no s_waitcnt before WMMA writes the same VGPR region.
               "v_wmma_f32_16x16x16_f16 v[0:7], v[8:15], v[16:23], 0\n"
               "s_waitcnt vmcnt(0)\n"
               :
               : "v"(&src[tid * 4])
               : "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11",
                 "v12", "v13", "v14", "v15", "v16", "v17", "v18", "v19", "v20", "v21", "v22",
                 "v23", "memory");
  out[tid] = tid;
}

__attribute__((amdgpu_flat_work_group_size(64, 64))) __global__ void
lds_wave32_cross_wave_safe_kernel(int *out) {
  __shared__ int lds[64];
  int tid = threadIdx.x;
  lds[tid] = tid + 100;
  __syncthreads();
  out[tid] = lds[(tid + 32) & 63];
}

static constexpr RaceExpectation kLdsWave32CrossWaveRaceExpectation{
    .findings = FindingCount::OneOrMore,
    .type = "LDS",
};

__attribute__((amdgpu_flat_work_group_size(64, 64))) __global__ void
lds_wave32_cross_wave_race_kernel(int *out) {
  __shared__ int lds[64];
  int tid = threadIdx.x;
  lds[tid] = tid + 100;
  if (tid >= 32)
    out[tid] = lds[tid - 32];
}

TEST_F(Gfx1151RaceTest, vgpr_waitcnt) {
  auto *src = allocWithData<float>(32);
  auto *dst = alloc<float>(32);
  vgpr_no_race_kernel<<<1, 32>>>(src, dst);
  sync();
  ExpectNoRace();
}

TEST_F(Gfx1151RaceTest, vgpr_waitcnt_race) {
  auto *src = allocWithData<float>(32);
  auto *dst = alloc<float>(32);
  vgpr_race_kernel<<<1, 32>>>(src, dst);
  sync();
  ExpectRace(kVgprRaceExpectation);
}

TEST_F(Gfx1151RaceTest, standalone_vmcnt_waits_for_vmem) {
  auto *src = allocWithData<float>(32);
  auto *dst = alloc<float>(32);
  standalone_vmcnt_waits_for_vmem_kernel<<<1, 32>>>(src, dst);
  sync();
  ExpectNoRace();
}

TEST_F(Gfx1151RaceTest, standalone_vmcnt_does_not_wait_for_lds) {
  standalone_vmcnt_does_not_wait_for_lds_kernel<<<1, 32>>>(alloc<int>(32));
  sync();
  ExpectRace(kVgprRaceExpectation);
}

TEST_F(Gfx1151RaceTest, standalone_lgkmcnt_waits_for_lds) {
  standalone_lgkmcnt_waits_for_lds_kernel<<<1, 32>>>(alloc<int>(32));
  sync();
  ExpectNoRace();
}

TEST_F(Gfx1151RaceTest, standalone_lgkmcnt_does_not_wait_for_vmem) {
  auto *src = allocWithData<float>(32);
  auto *dst = alloc<float>(32);
  standalone_lgkmcnt_does_not_wait_for_vmem_kernel<<<1, 32>>>(src, dst);
  sync();
  ExpectRace(kVgprRaceExpectation);
}

TEST_F(Gfx1151RaceTest, d16_global_load_merge_no_race) {
  auto *src = allocWithData<uint8_t>(32);
  auto *dst = alloc<uint32_t>(32);
  d16_global_load_merge_kernel<<<1, 32>>>(src, dst);
  sync();

  std::vector<uint32_t> h(32);
  (void)hipMemcpy(h.data(), dst, h.size() * sizeof(uint32_t), hipMemcpyDeviceToHost);
  for (uint32_t i = 0; i < h.size(); ++i)
    EXPECT_EQ(h[i], 0xABCD0000u | i);
  ExpectNoRace();
}

TEST_F(Gfx1151RaceTest, d16_global_load_hi_merge_no_race) {
  auto *src = allocWithData<uint8_t>(32);
  auto *dst = alloc<uint32_t>(32);
  d16_global_load_hi_merge_kernel<<<1, 32>>>(src, dst);
  sync();

  std::vector<uint32_t> h(32);
  (void)hipMemcpy(h.data(), dst, h.size() * sizeof(uint32_t), hipMemcpyDeviceToHost);
  for (uint32_t i = 0; i < h.size(); ++i)
    EXPECT_EQ(h[i], (i << 16) | 0xABCDu);
  ExpectNoRace();
}

TEST_F(Gfx1151RaceTest, d16_global_load_missing_waitcnt_race) {
  auto *src = allocWithData<uint8_t>(32);
  auto *dst = alloc<uint32_t>(32);
  d16_global_load_missing_waitcnt_kernel<<<1, 32>>>(src, dst);
  sync();

  ExpectRace(kD16GlobalLoadMissingWaitcntRaceExpectation);
}

TEST_F(Gfx1151RaceTest, lds_wave32_cross_wave) {
  lds_wave32_cross_wave_safe_kernel<<<1, 64>>>(alloc<int>(64));
  sync();
  ExpectNoRace();
}

TEST_F(Gfx1151RaceTest, lds_wave32_cross_wave_race) {
  lds_wave32_cross_wave_race_kernel<<<1, 64>>>(alloc<int>(64));
  sync();
  ExpectRace(kLdsWave32CrossWaveRaceExpectation);
}

TEST_F(Gfx1151RaceTest, waw_global_load_then_wmma) {
  auto *src = allocWithData<int>(128);
  auto *out = alloc<int>(32);
  waw_global_load_then_wmma_kernel<<<1, 32>>>(src, out);
  sync();
  ExpectRace(kWawGlobalLoadThenWmmaExpectation);
}

} // namespace
