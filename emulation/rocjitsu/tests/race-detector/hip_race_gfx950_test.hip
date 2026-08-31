// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// HIP race detection end-to-end tests.
//
// Each test launches one or more kernels through the rocjitsu emulator
// with the race detector selected in the test config. After hipDeviceSynchronize(),
// the test parses the structured race log to verify that races were
// (or were not) detected.
//
// ctest runs each TEST() in its own process (via --gtest_filter) so that
// each test gets a fresh output directory (RJ_SINK_DIR). The race plugin
// writes to RJ_SINK_DIR/race.log via the plugin sink system.
//
// Naming convention:
//   _race suffix = expects race, no suffix = expects no race.
//   Paired tests share a base name (e.g. vgpr_waitcnt / vgpr_waitcnt_race).
//
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include "race_test_support.hpp"

#include <cstdint>
#include <cstdio>
#include <vector>

using rocjitsu::test::FindingCount;
using rocjitsu::test::RaceExpectation;

using Gfx950RaceTest = rocjitsu::test::RaceTestBase;

// ============================================================
// Tests
// ============================================================

// --- VGPR waitcnt ---

__global__ void vgpr_no_race_kernel(const float *src, float *dst) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float val;
  asm volatile(
      "global_load_dword %0, %1, off\n"
      "s_waitcnt vmcnt(0)\n"
      : "=v"(val) : "v"(&src[tid])
  );
  dst[tid] = val;
}

static constexpr RaceExpectation kVgprWaitcntRaceExpectation{
    .kernel = "vgpr_race_kernel",
    .type = "VGPR",
    .access = "read",
    .wave = 0,
    .lane = 0,
    .producer = "global_load_dword",
    .consumer = "global_store_dword",
};

__global__ void vgpr_race_kernel(const float *src, float *dst) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float val;
  asm volatile(
      "global_load_dword %0, %1, off\n"
      // BUG: missing s_waitcnt vmcnt(0).
      "s_nop 0\n"
      : "=v"(val) : "v"(&src[tid])
  );
  dst[tid] = val;
}

TEST_F(Gfx950RaceTest, vgpr_waitcnt) {
  auto *src = allocWithData<float>(64), *dst = alloc<float>(64);
  vgpr_no_race_kernel<<<1, 64>>>(src, dst);
  sync();
  ExpectNoRace();
}

TEST_F(Gfx950RaceTest, vgpr_waitcnt_race) {
  auto *src = allocWithData<float>(64), *dst = alloc<float>(64);
  vgpr_race_kernel<<<1, 64>>>(src, dst);
  sync();
  ExpectRace(kVgprWaitcntRaceExpectation);
}

// --- SGPR waitcnt ---

__global__ void sgpr_no_race_kernel(const int *src, int *dst) {
  int val;
  asm volatile(
      "s_load_dword %0, %1, 0x0\n"
      "s_waitcnt lgkmcnt(0)\n"
      "s_add_u32 %0, %0, 1\n"
      : "=s"(val) : "s"(src)
  );
  dst[threadIdx.x] = val;
}

static constexpr RaceExpectation kSgprWaitcntRaceExpectation{
    .type = "SGPR",
    .wave = 0,
    .producer = "s_load_dword",
    .consumer = "s_add_u32",
};

__global__ void sgpr_race_kernel(const int *src, int *dst) {
  int val;
  asm volatile(
      "s_load_dword %0, %1, 0x0\n"
      // BUG: missing s_waitcnt lgkmcnt(0).
      "s_add_u32 %0, %0, 1\n"
      : "=s"(val) : "s"(src)
  );
  dst[threadIdx.x] = val;
}

TEST_F(Gfx950RaceTest, sgpr_waitcnt) {
  auto *src = allocWithData<int>(64), *dst = alloc<int>(64);
  sgpr_no_race_kernel<<<1, 64>>>(src, dst);
  sync();
  ExpectNoRace();
}

TEST_F(Gfx950RaceTest, sgpr_waitcnt_race) {
  auto *src = allocWithData<int>(64), *dst = alloc<int>(64);
  sgpr_race_kernel<<<1, 64>>>(src, dst);
  sync();
  ExpectRace(kSgprWaitcntRaceExpectation);
}

// --- LDS cross-wave ---

__attribute__((amdgpu_flat_work_group_size(128, 128)))
__global__ void lds_cross_wave_safe_kernel(int *out) {
  __shared__ int lds[128];
  int tid = threadIdx.x;
  lds[tid] = tid + 100;
  __syncthreads();
  out[tid] = lds[(tid + 64) % 128];
}

static constexpr RaceExpectation kLdsCrossWaveRaceExpectation{
    .findings = FindingCount::OneOrMore,
    .type = "LDS",
    .wave = 0,
    .producer = "ds_write_b32",
    .consumer = "ds_read_b32",
};

__attribute__((amdgpu_flat_work_group_size(128, 128)))
__global__ void lds_cross_wave_race_kernel(int *out) {
  __shared__ int lds[128];
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;
  lds[tid] = tid + 100;
  // BUG: no __syncthreads() before cross-wave read.
  if (blockIdx.x == 3)
    out[gid] = lds[(tid + 1) % 128];
  else
    out[gid] = lds[tid];
}

TEST_F(Gfx950RaceTest, lds_cross_wave) {
  lds_cross_wave_safe_kernel<<<1, 128>>>(alloc<int>(128));
  sync();
  ExpectNoRace();
}

TEST_F(Gfx950RaceTest, lds_cross_wave_race) {
  lds_cross_wave_race_kernel<<<5, 128>>>(alloc<int>(5 * 128));
  sync();
  ExpectRace(kLdsCrossWaveRaceExpectation);
}

// --- LDS same-wave instruction ordering ---

__attribute__((amdgpu_flat_work_group_size(64, 64)))
__global__ void lds_same_wave_write_read_order_kernel(uint32_t *out) {
  uint32_t lane = threadIdx.x;
  uint32_t read_lane = (lane + 1) & 63;
  uint32_t value = lane + 1;
  uint32_t observed;
  asm volatile(
      "ds_write_b8 %1, %2\n"
      "ds_read_u8 %0, %3\n"
      "s_waitcnt lgkmcnt(0)\n"
      : "=&v"(observed)
      : "v"(lane), "v"(value), "v"(read_lane)
      : "memory"
  );
  out[lane] = observed;
}

__attribute__((amdgpu_flat_work_group_size(64, 64)))
__global__ void lds_same_wave_read_write_order_kernel(uint32_t *out) {
  uint32_t lane = threadIdx.x;
  uint32_t read_lane = (lane + 1) & 63;
  uint32_t old_value = lane + 1;
  uint32_t new_value = old_value ^ 0xff;
  uint32_t observed_old;
  uint32_t observed_new;

  // Establish the old value, then test that the following same-wave DS read
  // completes logically before the later DS write to the overlapping bytes.
  asm volatile(
      "ds_write_b8 %0, %1\n"
      "s_waitcnt lgkmcnt(0)\n"
      :
      : "v"(lane), "v"(old_value)
      : "memory"
  );
  asm volatile(
      "ds_read_u8 %0, %2\n"
      "ds_write_b8 %1, %3\n"
      "s_waitcnt lgkmcnt(0)\n"
      : "=&v"(observed_old)
      : "v"(lane), "v"(read_lane), "v"(new_value)
      : "memory"
  );
  asm volatile(
      "ds_read_u8 %0, %1\n"
      "s_waitcnt lgkmcnt(0)\n"
      : "=v"(observed_new)
      : "v"(lane)
      : "memory"
  );

  out[2 * lane] = observed_old;
  out[2 * lane + 1] = observed_new;
}

TEST_F(Gfx950RaceTest, lds_same_wave_order) {
  auto *write_read_out = alloc<uint32_t>(64);
  lds_same_wave_write_read_order_kernel<<<1, 64, 64>>>(write_read_out);
  sync();

  std::vector<uint32_t> write_read_host(64);
  (void)hipMemcpy(write_read_host.data(), write_read_out,
                  write_read_host.size() * sizeof(uint32_t), hipMemcpyDeviceToHost);
  for (uint32_t lane = 0; lane < write_read_host.size(); ++lane)
    EXPECT_EQ(write_read_host[lane], ((lane + 1) & 63) + 1);

  auto *read_write_out = alloc<uint32_t>(128);
  lds_same_wave_read_write_order_kernel<<<1, 64, 64>>>(read_write_out);
  sync();

  std::vector<uint32_t> read_write_host(128);
  (void)hipMemcpy(read_write_host.data(), read_write_out,
                  read_write_host.size() * sizeof(uint32_t), hipMemcpyDeviceToHost);
  for (uint32_t lane = 0; lane < 64; ++lane) {
    EXPECT_EQ(read_write_host[2 * lane], ((lane + 1) & 63) + 1);
    EXPECT_EQ(read_write_host[2 * lane + 1], (lane + 1) ^ 0xff);
  }
  ExpectNoRace();
}

// --- Global-to-LDS buffer ---
// These kernels use buffer_load_dword with the LDS flag to load data from
// global memory directly into LDS (bypassing VGPRs). Designed for 128-thread
// workgroups (2 waves of 64). Each wave sets M0 to its LDS region offset and
// uses soffset for its global byte offset within the workgroup's buffer slice.

__attribute__((amdgpu_flat_work_group_size(128, 128)))
__global__ void buffer_load_lds_kernel(const int *src, int *out) {
  __shared__ int lds[128];
  int tid = threadIdx.x;
  int lane = tid % 64;
  int wave = tid / 64;
  lds[tid] = -1;
  __syncthreads();

  const int *wg_src = src + blockIdx.x * blockDim.x;
  uintptr_t base = reinterpret_cast<uintptr_t>(wg_src);
  unsigned w0 = static_cast<unsigned>(base);
  unsigned w1 = static_cast<unsigned>(base >> 32) & 0xFFFFu;
  unsigned w2 = 128 * sizeof(int);
  unsigned w3 = (4u) | (7u << 12) | (4u << 15);

  unsigned wave_byte_offset =
      __builtin_amdgcn_readfirstlane(wave * 64 * sizeof(int));
  asm volatile(
      "s_mov_b32 s12, %0\n"
      "s_mov_b32 s13, %1\n"
      "s_mov_b32 s14, %2\n"
      "s_mov_b32 s15, %3\n"
      "s_mov_b32 m0, %5\n"
      "buffer_load_dword %4, s[12:15], %6 offen lds\n"
      "s_waitcnt vmcnt(0)\n"
      :
      : "s"(w0), "s"(w1), "s"(w2), "s"(w3),
        "v"(lane * 4), "s"(wave_byte_offset), "s"(wave_byte_offset)
      : "s12", "s13", "s14", "s15", "memory"
  );

  __syncthreads();
  int gid = blockIdx.x * blockDim.x + tid;
  out[gid] = lds[tid];
}

static constexpr RaceExpectation kGlobalToLdsBufferRaceExpectation{
    .findings = FindingCount::OneOrMore,
    .type = "LDS",
    .wave = 0,
    .lane = 0,
    .producer = "buffer_load_dword",
    .between = "s_waitcnt",
    .consumer = "ds_read_b32",
};

__attribute__((amdgpu_flat_work_group_size(128, 128)))
__global__ void buffer_load_lds_race_kernel(const int *src, int *out) {
  __shared__ int lds[128];
  int tid = threadIdx.x;
  int lane = tid % 64;
  int wave = tid / 64;

  const int *wg_src = src + blockIdx.x * blockDim.x;
  uintptr_t base = reinterpret_cast<uintptr_t>(wg_src);
  unsigned w0 = static_cast<unsigned>(base);
  unsigned w1 = static_cast<unsigned>(base >> 32) & 0xFFFFu;
  unsigned w2 = 128 * sizeof(int);
  unsigned w3 = (4u) | (7u << 12) | (4u << 15);

  unsigned wave_byte_offset =
      __builtin_amdgcn_readfirstlane(wave * 64 * sizeof(int));
  asm volatile(
      "s_mov_b32 s12, %0\n"
      "s_mov_b32 s13, %1\n"
      "s_mov_b32 s14, %2\n"
      "s_mov_b32 s15, %3\n"
      "s_mov_b32 m0, %5\n"
      "buffer_load_dword %4, s[12:15], %6 offen lds\n"
      "s_waitcnt vmcnt(3)\n"
      "s_waitcnt vmcnt(2)\n"
      "s_waitcnt vmcnt(1)\n"
      // BUG: missing s_waitcnt vmcnt(0).
      :
      : "s"(w0), "s"(w1), "s"(w2), "s"(w3),
        "v"(lane * 4), "s"(wave_byte_offset), "s"(wave_byte_offset)
      : "s12", "s13", "s14", "s15", "memory"
  );

  int val;
  asm volatile(
      "ds_read_b32 %0, %1\n"
      "s_waitcnt lgkmcnt(0)\n"
      : "=v"(val) : "v"(tid * 4)
  );
  int gid = blockIdx.x * blockDim.x + tid;
  out[gid] = val;
}

TEST_F(Gfx950RaceTest, global_to_lds_buffer) {
  constexpr int N = 3 * 128;
  auto *src = allocWithData<int>(N), *out = alloc<int>(N);
  buffer_load_lds_kernel<<<3, 128>>>(src, out);
  sync();

  std::vector<int> h_src(N), h_out(N);
  for (int i = 0; i < N; ++i) h_src[i] = i;
  (void)hipMemcpy(h_out.data(), out, N * sizeof(int), hipMemcpyDeviceToHost);
  int errors = 0;
  for (int i = 0; i < N; ++i)
    if (h_out[i] != h_src[i] && errors++ < 5)
      fprintf(stderr, "MISMATCH [%d]: got %d, expected %d\n",
              i, h_out[i], h_src[i]);
  EXPECT_EQ(errors, 0) << errors << "/" << N << " mismatches";
}

TEST_F(Gfx950RaceTest, global_to_lds_buffer_race) {
  buffer_load_lds_race_kernel<<<3, 128>>>(alloc<int>(384), alloc<int>(384));
  sync();
  ExpectRace(kGlobalToLdsBufferRaceExpectation);
}

// --- Partial vmcnt ---

static constexpr RaceExpectation kPartialVmcntRaceExpectation{
    .kernel = "partial_vmcnt_kernel",
    .type = "VGPR",
    .wave = 0,
    .lane = 0,
    .producer = "global_load_dword",
    .between = "s_waitcnt",
    .consumer = "v_add_f32",
};

__global__ void partial_vmcnt_kernel(const float *src_a, const float *src_b,
                                     float *dst) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float a, b;
  asm volatile(
      "global_load_dword %0, %2, off\n"   // oldest (vmcnt event 0)
      "global_load_dword %1, %3, off\n"   // newest (vmcnt event 1)
      "s_waitcnt vmcnt(1)\n"              // drain oldest only
      : "=v"(a), "=v"(b)
      : "v"(&src_a[tid]), "v"(&src_b[tid])
  );
  // Reading b races (still pending). Reading a is safe.
  dst[tid] = a + b;
}

TEST_F(Gfx950RaceTest, partial_vmcnt_race) {
  partial_vmcnt_kernel<<<1, 64>>>(alloc<float>(64), alloc<float>(64),
                                   alloc<float>(64));
  sync();
  ExpectRace(kPartialVmcntRaceExpectation);
}

// --- Multiple race ---

static constexpr RaceExpectation kMultipleRaceExpectation{
    .findings = FindingCount::OneOrMore,
    .type = "VGPR",
    .producer = "global_load_dword",
    .consumer = "v_add_f32",
};

__global__ void two_races_kernel(const float *src_a, const float *src_b,
                                 float *dst) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float a, b;
  asm volatile(
      "global_load_dword %0, %2, off\n"
      "global_load_dword %1, %3, off\n"
      // No waitcnt at all.
      : "=v"(a), "=v"(b)
      : "v"(&src_a[tid]), "v"(&src_b[tid])
  );
  dst[tid] = a + b;
}

TEST_F(Gfx950RaceTest, multiple_race) {
  two_races_kernel<<<1, 64>>>(alloc<float>(64), alloc<float>(64),
                               alloc<float>(64));
  sync();
  // Both loads race against v_add_f32, but dedup by PC collapses them to 1.
  ExpectRace(kMultipleRaceExpectation);
}

// --- Multi-kernel ---

__global__ void clean_kernel_a(const float *src, float *dst) {
  dst[threadIdx.x] = src[threadIdx.x];
}

__global__ void clean_kernel_b(const float *src, float *dst) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  dst[tid] = src[tid] + 1.0f;
}

static constexpr RaceExpectation kMultiKernelRaceExpectation{
    .kernel = "racy_kernel",
    .type = "VGPR",
    .wave = 0,
    .lane = 0,
    .producer = "global_load_dword",
    .consumer = "global_store_dword",
};

__global__ void racy_kernel(const float *src, float *dst) {
  int tid = threadIdx.x;
  float val;
  asm volatile(
      "global_load_dword %0, %1, off\n"
      // No waitcnt.
      : "=v"(val) : "v"(&src[tid])
  );
  dst[tid] = val;
}

TEST_F(Gfx950RaceTest, multi_kernel) {
  auto *s = alloc<float>(1000), *d = alloc<float>(1000);
  clean_kernel_a<<<1, 64>>>(s, d);
  clean_kernel_b<<<3, 128>>>(s, d);
  clean_kernel_a<<<5, 64>>>(s, d);
  clean_kernel_b<<<2, 64>>>(s, d);
  sync();
  ExpectNoRace();
}

TEST_F(Gfx950RaceTest, multi_kernel_race) {
  auto *s = alloc<float>(1000), *d = alloc<float>(1000);
  clean_kernel_a<<<1, 64>>>(s, d);
  clean_kernel_a<<<3, 128>>>(s, d);
  racy_kernel<<<1, 64>>>(s, d);
  clean_kernel_a<<<5, 64>>>(s, d);
  sync();
  ExpectRace(kMultiKernelRaceExpectation);
}

// --- Multi-workgroup ---

__global__ void multi_wg_clean_kernel(const float *src, float *dst) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float val;
  asm volatile(
      "global_load_dword %0, %1, off\n"
      "s_waitcnt vmcnt(0)\n"
      : "=v"(val) : "v"(&src[tid])
  );
  dst[tid] = val;
}

static constexpr RaceExpectation kMultiWorkgroupRaceExpectation{
    .findings = FindingCount::OneOrMore,
    .type = "VGPR",
    .wave = 0,
    .lane = 0,
    .producer = "global_load_dword",
    .consumer = "global_store_dword",
};

__global__ void multi_wg_race_kernel(const float *src, float *dst) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float val;
  asm volatile(
      "global_load_dword %0, %1, off\n"
      // No waitcnt.
      : "=v"(val) : "v"(&src[tid])
  );
  dst[tid] = val;
}

TEST_F(Gfx950RaceTest, multi_workgroup) {
  auto *s = alloc<float>(256), *d = alloc<float>(256);
  multi_wg_clean_kernel<<<4, 64>>>(s, d);
  sync();
  ExpectNoRace();
}

TEST_F(Gfx950RaceTest, multi_workgroup_race) {
  auto *s = alloc<float>(401 * 64), *d = alloc<float>(401 * 64);
  multi_wg_race_kernel<<<401, 64>>>(s, d);
  sync();
  ExpectRace(kMultiWorkgroupRaceExpectation);
}

// --- Mixed counters ---

static constexpr RaceExpectation kMixedCountersRaceExpectation{
    .findings = FindingCount::OneOrMore,
    .type = "SGPR",
    .wave = 0,
    .producer = "s_load_dword",
    .between = "s_waitcnt",
    .consumer = "v_cvt_f32_i32",
};

__global__ void mixed_counters_kernel(const float *vsrc, const int *ssrc,
                                      float *dst) {
  int tid = threadIdx.x;
  float vval;
  int sval;
  asm volatile(
      "global_load_dword %0, %2, off\n"
      "s_load_dword %1, %3, 0x0\n"
      "s_waitcnt vmcnt(0)\n"   // drains global_load only
      // lgkmcnt not drained.
      : "=v"(vval), "=s"(sval)
      : "v"(&vsrc[tid]), "s"(ssrc)
  );
  dst[tid] = vval + static_cast<float>(sval);
}

TEST_F(Gfx950RaceTest, mixed_counters_race) {
  mixed_counters_kernel<<<1, 64>>>(alloc<float>(64), alloc<int>(64),
                                    alloc<float>(64));
  sync();
  ExpectRace(kMixedCountersRaceExpectation);
}

// --- Stress: wavefront reuse ---

__global__ void stress_clean_a(const float *src, float *dst) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  dst[tid] = src[tid];
}

__global__ void stress_clean_b(const float *src, float *dst) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  dst[tid] = src[tid] + 1.0f;
}

static constexpr RaceExpectation kStressWavefrontReuseRaceExpectation{
    .findings = FindingCount::OneOrMore,
    .type = "VGPR",
    .wave = 0,
    .lane = 0,
    .producer = "global_load_dword",
    .consumer = "global_store_dword",
};

__global__ void stress_racy(const float *src, float *dst) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float val;
  asm volatile(
      "global_load_dword %0, %1, off\n"
      // No waitcnt.
      : "=v"(val) : "v"(&src[tid])
  );
  dst[tid] = val;
}

TEST_F(Gfx950RaceTest, stress_wavefront_reuse) {
  auto *s = alloc<float>(64000), *d = alloc<float>(64000);
  stress_clean_a<<<1000, 64>>>(s, d);
  stress_clean_b<<<1000, 64>>>(s, d);
  sync();
  ExpectNoRace();
}

TEST_F(Gfx950RaceTest, stress_wavefront_reuse_race) {
  auto *s = alloc<float>(64000), *d = alloc<float>(64000);
  stress_clean_a<<<1000, 64>>>(s, d);
  stress_racy<<<1000, 64>>>(s, d);
  sync();
  ExpectRace(kStressWavefrontReuseRaceExpectation);
}

// --- EXEC mask ---
// 128 threads (2 waves). Only first 60 lanes per wave write to LDS.
// Wave 1 reads from wave 0's LDS region. With barrier = safe; without = race.

__attribute__((amdgpu_flat_work_group_size(128, 128)))
__global__ void exec_mask_safe_kernel(int *out) {
  __shared__ int lds[128];
  int tid = threadIdx.x;
  if (tid % 64 < 60)
    lds[tid] = tid + 100;
  __syncthreads();
  if (tid >= 64)
    out[tid] = lds[tid - 64];
}

static constexpr RaceExpectation kExecMaskRaceExpectation{
    .findings = FindingCount::OneOrMore,
    .type = "LDS",
    .wave = 1,
    .lane = 0,
    .producer = "ds_write_b32",
    .consumer = "ds_read_b32",
};

__attribute__((amdgpu_flat_work_group_size(128, 128)))
__global__ void exec_mask_race_kernel(int *out) {
  __shared__ int lds[128];
  int tid = threadIdx.x;
  if (tid % 64 < 60)
    lds[tid] = tid + 100;
  // BUG: no barrier before cross-wave read.
  if (tid >= 64)
    out[tid] = lds[tid - 64];
}

TEST_F(Gfx950RaceTest, exec_mask) {
  exec_mask_safe_kernel<<<1, 128>>>(alloc<int>(128));
  sync();
  ExpectNoRace();
}

TEST_F(Gfx950RaceTest, exec_mask_race) {
  exec_mask_race_kernel<<<1, 128>>>(alloc<int>(128));
  sync();
  ExpectRace(kExecMaskRaceExpectation);
}

// --- EXEC mask: same-wave vs cross-wave boundary ---
//
// 256 threads (4 waves of 64). All threads write lds[tid] = tid, barrier,
// then a subset of lanes overwrite lds[tid+32] = tid. Finally all threads
// read out[tid] = lds[tid].
//
// The conditional overwrite runs under a partial EXEC mask (the if-branch).
// The key variable is WHICH lanes are active:
//
//   lanes [0,32):  lds[tid+32] stays within the same wave (max target
//                  within wave is lds[63] for wave 0). No cross-wave write.
//
//   lanes [1,33):  lane 32 writes lds[tid+32] = lds[64], which is wave 1's
//                  region. One lane crosses the wave boundary.
//
// Synchronization between the overwrite and the read:
//
//   s_waitcnt lgkmcnt(0) — drains the wave's own pending LDS writes.
//       Sufficient for same-wave cross-lane dependencies (lane 0 writes
//       lds[32], lane 32 reads lds[32]). NOT sufficient for cross-wave
//       dependencies (wave 0 writes lds[64], wave 1 reads lds[64]).
//
//   s_barrier (__syncthreads) — synchronizes all waves in the workgroup.
//       Required for cross-wave dependencies.

__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void exec_mask_same_wave_kernel(int *out) {
  __shared__ int lds[256];
  int tid = threadIdx.x;
  lds[tid] = tid;
  __syncthreads();
  if (tid % 64 < 32)
    lds[(tid + 32) % 256] = tid;
  asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
  out[tid] = lds[tid];
}

static constexpr RaceExpectation kExecMaskCrossWaveRaceExpectation{
    .findings = FindingCount::OneOrMore,
    .type = "LDS",
    .wave = 1,
    .lane = 0,
    .producer = "ds_write_b32",
    .between = "s_waitcnt",
    .consumer = "ds_read_b32",
};

__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void exec_mask_cross_wave_race_kernel(int *out) {
  __shared__ int lds[256];
  int tid = threadIdx.x;
  lds[tid] = tid;
  __syncthreads();
  int lane = tid % 64;
  if (lane >= 1 && lane < 33 && tid + 32 < 256)
    lds[tid + 32] = tid;
  // lgkmcnt drain resolves same-wave deps but NOT cross-wave — race.
  asm volatile("s_waitcnt lgkmcnt(0)\n" ::: "memory");
  out[tid] = lds[tid];
}

__attribute__((amdgpu_flat_work_group_size(256, 256)))
__global__ void exec_mask_cross_wave_barrier_kernel(int *out) {
  __shared__ int lds[256];
  int tid = threadIdx.x;
  lds[tid] = tid;
  __syncthreads();
  int lane = tid % 64;
  if (lane >= 1 && lane < 33 && tid + 32 < 256)
    lds[tid + 32] = tid;
  // Full barrier resolves cross-wave deps — safe.
  __syncthreads();
  out[tid] = lds[tid];
}

// Host-side simulation of the LDS state for numerical validation.
static std::vector<int> simulate_exec_mask_lds(int n, int lo, int hi) {
  std::vector<int> lds(n);
  for (int i = 0; i < n; ++i) lds[i] = i;
  for (int i = 0; i < n; ++i) {
    int lane = i % 64;
    if (lane >= lo && lane < hi && i + 32 < n)
      lds[i + 32] = i;
  }
  return lds; // lds[tid] == expected out[tid]
}

TEST_F(Gfx950RaceTest, exec_mask_same_wave) {
  auto *out = alloc<int>(256);
  exec_mask_same_wave_kernel<<<1, 256>>>(out);
  sync();
  ExpectNoRace();

  auto expected = simulate_exec_mask_lds(256, 0, 32);
  std::vector<int> h_out(256);
  (void)hipMemcpy(h_out.data(), out, 256 * sizeof(int), hipMemcpyDeviceToHost);
  int errors = 0;
  for (int i = 0; i < 256; ++i)
    if (h_out[i] != expected[i] && errors++ < 5)
      fprintf(stderr, "MISMATCH [%d]: got %d, expected %d\n",
              i, h_out[i], expected[i]);
  EXPECT_EQ(errors, 0) << errors << "/256 mismatches";
}

TEST_F(Gfx950RaceTest, exec_mask_cross_wave_race) {
  exec_mask_cross_wave_race_kernel<<<1, 256>>>(alloc<int>(256));
  sync();
  ExpectRace(kExecMaskCrossWaveRaceExpectation);
}

TEST_F(Gfx950RaceTest, exec_mask_cross_wave) {
  auto *out = alloc<int>(256);
  exec_mask_cross_wave_barrier_kernel<<<1, 256>>>(out);
  sync();
  ExpectNoRace();

  auto expected = simulate_exec_mask_lds(256, 1, 33);
  std::vector<int> h_out(256);
  (void)hipMemcpy(h_out.data(), out, 256 * sizeof(int), hipMemcpyDeviceToHost);
  int errors = 0;
  for (int i = 0; i < 256; ++i) {
    if (h_out[i] != expected[i]) {
      if (errors++ < 10)
        fprintf(stderr, "MISMATCH [%d]: got %d, expected %d\n",
                i, h_out[i], expected[i]);
    }
  }
  EXPECT_EQ(errors, 0) << errors << "/256 mismatches";
}

// --- LDS transpose ---
//
// 128 threads (2 waves of 64) swap data through LDS. Each thread writes its
// tid to lds[tid], then reads from the other wave's half: lds[(tid+64)%128].
// Wave 0 (tids 0-63) reads addresses 64-127, wave 1 (tids 64-127) reads
// addresses 0-63 — every read crosses the wave boundary.
//
// s_waitcnt lgkmcnt(0) drains this wave's own LDS writes but does NOT
// synchronize with the other wave. Only s_barrier guarantees that both
// waves have completed their writes before either wave reads.

__attribute__((amdgpu_flat_work_group_size(128, 128)))
__global__ void lds_transpose_safe_kernel(int *out) {
  __shared__ int lds[128];
  int tid = threadIdx.x;
  int base = static_cast<int>(reinterpret_cast<uintptr_t>(lds) & 0xFFFF);
  int write_addr = base + tid * 4;
  int read_addr = base + ((tid + 64) % 128) * 4;
  int val;
  asm volatile(
      "ds_write_b32 %1, %2\n"
      "s_waitcnt lgkmcnt(0)\n"
      "s_barrier\n"
      "ds_read_b32 %0, %3\n"
      "s_waitcnt lgkmcnt(0)\n"
      : "=v"(val)
      : "v"(write_addr), "v"(tid), "v"(read_addr)
  );
  out[tid] = val;
}

static constexpr RaceExpectation kLdsTransposeRaceExpectation{
    .findings = FindingCount::OneOrMore,
    .type = "LDS",
    .producer = "ds_write_b32",
    .consumer = "ds_read_b32",
};

__attribute__((amdgpu_flat_work_group_size(128, 128)))
__global__ void lds_transpose_race_kernel(int *out) {
  __shared__ int lds[128];
  int tid = threadIdx.x;
  int base = static_cast<int>(reinterpret_cast<uintptr_t>(lds) & 0xFFFF);
  int write_addr = base + tid * 4;
  int read_addr = base + ((tid + 64) % 128) * 4;
  int val;
  asm volatile(
      "ds_write_b32 %1, %2\n"
      "s_waitcnt lgkmcnt(0)\n"
      // BUG: missing s_barrier — wave 0 reads wave 1's region before it's written.
      "ds_read_b32 %0, %3\n"
      "s_waitcnt lgkmcnt(0)\n"
      : "=v"(val)
      : "v"(write_addr), "v"(tid), "v"(read_addr)
  );
  out[tid] = val;
}

TEST_F(Gfx950RaceTest, lds_transpose) {
  auto *out = alloc<int>(128);
  lds_transpose_safe_kernel<<<1, 128>>>(out);
  sync();
  ExpectNoRace();

  std::vector<int> h_out(128);
  (void)hipMemcpy(h_out.data(), out, 128 * sizeof(int), hipMemcpyDeviceToHost);
  int errors = 0;
  for (int i = 0; i < 128; ++i)
    if (h_out[i] != (i + 64) % 128 && errors++ < 5)
      fprintf(stderr, "MISMATCH [%d]: got %d, expected %d\n",
              i, h_out[i], (i + 64) % 128);
  EXPECT_EQ(errors, 0) << errors << "/128 mismatches";
}

TEST_F(Gfx950RaceTest, lds_transpose_race) {
  lds_transpose_race_kernel<<<1, 128>>>(alloc<int>(128));
  sync();
  ExpectRace(kLdsTransposeRaceExpectation);
}

// --- Dual-offset LDS ---
// ds_write2_b32 writes two dwords per lane to LDS at two offsets.
// Cross-wave read without barrier = race; with barrier = safe.

__attribute__((amdgpu_flat_work_group_size(128, 128)))
__global__ void dual_offset_lds_safe_kernel(int *out) {
  __shared__ int lds[256];
  int tid = threadIdx.x;
  asm volatile(
      "ds_write2_b32 %0, %1, %2 offset0:0 offset1:1\n"
      "s_waitcnt lgkmcnt(0)\n"
      : : "v"(tid * 8), "v"(tid), "v"(tid + 100) : "memory"
  );
  __syncthreads();
  if (tid >= 64) {
    int val;
    asm volatile(
        "ds_read_b32 %0, %1\n"
        "s_waitcnt lgkmcnt(0)\n"
        : "=v"(val) : "v"((tid - 64) * 8)
    );
    out[tid] = val;
  }
}

static constexpr RaceExpectation kDualOffsetLdsRaceExpectation{
    .findings = FindingCount::OneOrMore,
    .type = "LDS",
    .wave = 1,
    .lane = 0,
    .producer = "ds_write2_b32",
    .between = "s_waitcnt",
    .consumer = "ds_read_b32",
};

__attribute__((amdgpu_flat_work_group_size(128, 128)))
__global__ void dual_offset_lds_race_kernel(int *out) {
  __shared__ int lds[256];
  int tid = threadIdx.x;
  asm volatile(
      "ds_write2_b32 %0, %1, %2 offset0:0 offset1:1\n"
      "s_waitcnt lgkmcnt(0)\n"
      : : "v"(tid * 8), "v"(tid), "v"(tid + 100) : "memory"
  );
  // BUG: no barrier before cross-wave read.
  if (tid >= 64) {
    int val;
    asm volatile(
        "ds_read_b32 %0, %1\n"
        "s_waitcnt lgkmcnt(0)\n"
        : "=v"(val) : "v"((tid - 64) * 8)
    );
    out[tid] = val;
  }
}

TEST_F(Gfx950RaceTest, dual_offset_lds) {
  dual_offset_lds_safe_kernel<<<1, 128>>>(alloc<int>(128));
  sync();
  ExpectNoRace();
}

TEST_F(Gfx950RaceTest, dual_offset_lds_race) {
  dual_offset_lds_race_kernel<<<1, 128>>>(alloc<int>(128));
  sync();
  ExpectRace(kDualOffsetLdsRaceExpectation);
}

// --- Scratch store vmcnt accounting ---
// scratch_store consumes a vmcnt slot. A global_load followed by two
// scratch_stores has 3 outstanding vmcnt events. vmcnt(2) drains the
// oldest (the global_load), so reading the loaded register is safe.

__global__ void scratch_vmcnt_safe_kernel(const float *src, float *dst) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float val;
  float tmp1 = 1.0f, tmp2 = 2.0f;
  asm volatile(
      "global_load_dword %0, %3, off\n"       // vmcnt event 0 (oldest)
      "scratch_store_byte %4, %1, off\n"       // vmcnt event 1
      "scratch_store_byte %4, %2, off offset:1\n" // vmcnt event 2 (newest)
      "s_waitcnt vmcnt(2)\n"                   // drain oldest → load complete
      : "=v"(val)
      : "v"(tmp1), "v"(tmp2), "v"(&src[tid]), "v"(0)
  );
  dst[tid] = val;
}

static constexpr RaceExpectation kScratchVmcntRaceExpectation{
    .findings = FindingCount::OneOrMore,
    .type = "VGPR",
    .producer = "global_load_dword",
};

__global__ void scratch_vmcnt_race_kernel(const float *src, float *dst) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float val;
  float tmp1 = 1.0f;
  asm volatile(
      "global_load_dword %0, %2, off\n"    // vmcnt event 0 (oldest)
      "scratch_store_byte %3, %1, off\n"   // vmcnt event 1 (newest)
      "s_waitcnt vmcnt(2)\n"               // BUG: 2 outstanding, need vmcnt(1)
      : "=v"(val)
      : "v"(tmp1), "v"(&src[tid]), "v"(0)
  );
  dst[tid] = val;
}

TEST_F(Gfx950RaceTest, scratch_vmcnt) {
  auto *src = allocWithData<float>(64), *dst = alloc<float>(64);
  scratch_vmcnt_safe_kernel<<<1, 64>>>(src, dst);
  sync();
  ExpectNoRace();
}

TEST_F(Gfx950RaceTest, scratch_vmcnt_race) {
  auto *src = allocWithData<float>(64), *dst = alloc<float>(64);
  scratch_vmcnt_race_kernel<<<1, 64>>>(src, dst);
  sync();
  ExpectRace(kScratchVmcntRaceExpectation);
}

// gfx950 enables SRAM ECC, so D16 loads write the full 32-bit VGPR and the
// race detector registers a full-byte-mask pending write. Independent
// low/high-half D16 behavior applies only to targets without SRAM ECC and is
// covered by the architecture-parameterized C++ plugin tests.

// --- VGPR WAW: pending load then instruction write ---

static constexpr RaceExpectation kWawGlobalLoadThenAluExpectation{
    .findings = FindingCount::OneOrMore,
    .type = "VGPR",
    .access = "write",
    .producer = "global_load_dword",
    .consumer = "v_add_f32",
};

__global__ void waw_global_load_then_alu_kernel(const float *src, float *dst) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float val;
  float tmp = static_cast<float>(tid);
  asm volatile(
      "global_load_dword %0, %1, off\n"
      // BUG: no s_waitcnt before ALU write to the same register.
      // Use a separate source so this isolates WAW rather than RAW.
      "v_add_f32 %0, %2, %2\n"
      : "=&v"(val) : "v"(&src[tid]), "v"(tmp)
  );
  // Keep the racy load and overwrite in one asm block. This separate wait only
  // orders the later store and is not part of the race being exercised.
  asm volatile("s_waitcnt vmcnt(0)\n" ::: "memory");
  dst[tid] = val;
}

__global__ void waw_global_load_then_alu_safe_kernel(const float *src, float *dst) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  float val;
  float tmp = static_cast<float>(tid);
  asm volatile(
      "global_load_dword %0, %1, off\n"
      "s_waitcnt vmcnt(0)\n"
      "v_add_f32 %0, %2, %2\n"
      : "=&v"(val) : "v"(&src[tid]), "v"(tmp)
  );
  dst[tid] = val;
}

static constexpr RaceExpectation kWawGlobalLoadThenMfmaExpectation{
    .findings = FindingCount::OneOrMore,
    .type = "VGPR",
    .access = "write",
    .producer = "global_load_dwordx4",
    .consumer = "v_mfma",
};

__global__ void waw_global_load_then_mfma_kernel(const int *src, int *out) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  asm volatile(
      "global_load_dwordx4 v[0:3], %0, off\n"
      // BUG: no s_waitcnt before MFMA writes the same VGPR region.
      "v_mfma_f32_4x4x1_16b_f32 v[0:3], v4, v5, 0\n"
      "s_waitcnt vmcnt(0)\n"
      :
      : "v"(&src[tid * 4])
      : "v0", "v1", "v2", "v3", "v4", "v5", "memory"
  );
  out[tid] = tid;
}

TEST_F(Gfx950RaceTest, waw_global_load_then_alu) {
  auto *src = allocWithData<float>(64), *dst = alloc<float>(64);
  waw_global_load_then_alu_kernel<<<1, 64>>>(src, dst);
  sync();
  ExpectRace(kWawGlobalLoadThenAluExpectation);
}

TEST_F(Gfx950RaceTest, waw_global_load_then_alu_safe) {
  auto *src = allocWithData<float>(64), *dst = alloc<float>(64);
  waw_global_load_then_alu_safe_kernel<<<1, 64>>>(src, dst);
  sync();
  ExpectNoRace();
}

TEST_F(Gfx950RaceTest, waw_global_load_then_mfma) {
  auto *src = allocWithData<int>(256), *out = alloc<int>(64);
  waw_global_load_then_mfma_kernel<<<1, 64>>>(src, out);
  sync();
  ExpectRace(kWawGlobalLoadThenMfmaExpectation);
}

// --- f64 VGPR race (64-bit global load + use without waitcnt) ---
// Loads a double via global_load_dwordx2 into a VGPR pair, then
// consumes it with v_fma_f64 without an intervening s_waitcnt.
// Exercises 64-bit VGPR race detection through the scalar read_lane64
// path (v_fma_f64 does not currently have a SIMD fast-path probe).

static constexpr RaceExpectation kF64VgprRaceExpectation{
    .findings = FindingCount::OneOrMore,
    .type = "VGPR",
};

__global__ void f64_vgpr_race_kernel(const double *src, double *dst) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  double val, result;
  asm volatile(
      "global_load_dwordx2 %0, %2, off\n"
      "s_nop 0\n"
      "v_fma_f64 %1, %0, 1.0, 1.0\n"
      : "=&v"(val), "=v"(result) : "v"(&src[tid])
  );
  dst[tid] = result;
}

TEST_F(Gfx950RaceTest, f64_vgpr_race) {
  double *src = allocWithData<double>(64);
  double *dst = alloc<double>(64);
  f64_vgpr_race_kernel<<<1, 64>>>(src, dst);
  sync();
  ExpectRace(kF64VgprRaceExpectation);
}

__global__ void f64_vgpr_safe_kernel(const double *src, double *dst) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  double val, result;
  asm volatile(
      "global_load_dwordx2 %0, %2, off\n"
      "s_waitcnt vmcnt(0)\n"
      "v_fma_f64 %1, %0, 1.0, 1.0\n"
      : "=&v"(val), "=v"(result) : "v"(&src[tid])
  );
  dst[tid] = result;
}

TEST_F(Gfx950RaceTest, f64_vgpr_safe) {
  double *src = allocWithData<double>(64);
  double *dst = alloc<double>(64);
  f64_vgpr_safe_kernel<<<1, 64>>>(src, dst);
  sync();
  ExpectNoRace();
}

static constexpr RaceExpectation kF64VgprWawExpectation{
    .findings = FindingCount::OneOrMore,
    .type = "VGPR",
    .access = "write",
    .producer = "global_load_dwordx2",
    .consumer = "v_fma_f64",
};

__global__ void f64_vgpr_waw_kernel(const double *src, double *dst) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  double val;
  double tmp = static_cast<double>(tid) + 1.0;
  asm volatile(
      "global_load_dwordx2 %0, %1, off\n"
      // BUG: no s_waitcnt before a 64-bit ALU write to the same VGPR pair.
      "v_fma_f64 %0, %2, 1.0, 1.0\n"
      "s_waitcnt vmcnt(0)\n"
      : "=&v"(val) : "v"(&src[tid]), "v"(tmp)
  );
  dst[tid] = val;
}

TEST_F(Gfx950RaceTest, f64_vgpr_waw) {
  double *src = allocWithData<double>(64);
  double *dst = alloc<double>(64);
  f64_vgpr_waw_kernel<<<1, 64>>>(src, dst);
  sync();
  ExpectRace(kF64VgprWawExpectation);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
