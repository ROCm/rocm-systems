// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "mma_test_util.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/shared/mma_exec.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <random>

namespace {

using namespace rocjitsu;
using Clock = std::chrono::steady_clock;

// Shared with the SIMD MFMA/WMMA benchmarks via mma_test_util.h (single source of
// truth) so register-file config and iteration count stay consistent across the
// benchmark suite and CI runtime.
constexpr uint32_t WF_SIZE = mma_test::MFMA_WF_SIZE;
constexpr uint32_t SGPRS_PER_WF = mma_test::SGPRS_PER_WF;
constexpr uint32_t VGPRS_PER_WF = mma_test::VGPRS_PER_WF;
constexpr int ITERATIONS = mma_test::BENCH_ITERATIONS;

// VGPR-file offsets for the operands, kept compact so the largest shape's dst
// range stays within VGPRS_PER_WF (=256): s0 [10,58), s1 [50,98), dst [128,144).
constexpr uint32_t S0_OFF = 10;
constexpr uint32_t S1_OFF = 50;
constexpr uint32_t DST_OFF = 128;

struct MfmaBenchFixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  amdgpu::Wavefront *wf = nullptr;
  uint32_t vb = 0;

  MfmaBenchFixture() : gpu_mem("mfma_bench_mem"), l2("mfma_bench_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_mfma_bench", cfg, &gpu_mem, &l2);
    // Guard against a failed CU create / wavefront dispatch: leave wf null and let
    // the per-test ASSERT_NE(fx.wf, nullptr) report it cleanly instead of
    // dereferencing null here (which would crash before any assertion runs).
    if (!cu)
      return;
    wf = cu->dispatch_wf(0, 0, SGPRS_PER_WF, VGPRS_PER_WF);
    if (!wf)
      return;
    vb = wf->vgpr_alloc().base;
    wf->set_exec(~0ULL);
  }

  void seed_f16(uint64_t seed, uint32_t s0_regs, uint32_t s1_regs, uint32_t dst_regs) {
    std::mt19937_64 rng(seed);
    auto rand_f16x2 = [&]() -> uint32_t {
      float f1 = static_cast<float>(rng() % 100) / 50.0f - 1.0f;
      float f2 = static_cast<float>(rng() % 100) / 50.0f - 1.0f;
      uint16_t h1 = util::f32_to_f16(f1);
      uint16_t h2 = util::f32_to_f16(f2);
      return static_cast<uint32_t>(h2) << 16 | h1;
    };
    for (uint32_t r = 0; r < s0_regs; ++r)
      for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
        cu->write_vgpr(vb + S0_OFF + r, lane, rand_f16x2());
    for (uint32_t r = 0; r < s1_regs; ++r)
      for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
        cu->write_vgpr(vb + S1_OFF + r, lane, rand_f16x2());
    for (uint32_t r = 0; r < dst_regs; ++r)
      for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
        cu->write_vgpr(vb + DST_OFF + r, lane, 0);
  }

  void seed_i8(uint64_t seed, uint32_t s0_regs, uint32_t s1_regs, uint32_t dst_regs) {
    std::mt19937_64 rng(seed);
    for (uint32_t r = 0; r < s0_regs; ++r)
      for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
        cu->write_vgpr(vb + S0_OFF + r, lane, static_cast<uint32_t>(rng()));
    for (uint32_t r = 0; r < s1_regs; ++r)
      for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
        cu->write_vgpr(vb + S1_OFF + r, lane, static_cast<uint32_t>(rng()));
    for (uint32_t r = 0; r < dst_regs; ++r)
      for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
        cu->write_vgpr(vb + DST_OFF + r, lane, 0);
  }
};

struct MfmaStats {
  double ns_per_inst = 0;
  double mips = 0;
  uint64_t total_ns = 0;
};

template <typename Fn> MfmaStats time_mfma(Fn fn, int iterations) {
  for (int i = 0; i < 100; ++i)
    fn();
  auto t0 = Clock::now();
  for (int i = 0; i < iterations; ++i)
    fn();
  auto t1 = Clock::now();
  MfmaStats s;
  s.total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
  s.ns_per_inst = static_cast<double>(s.total_ns) / iterations;
  s.mips = (1e9 / s.ns_per_inst) / 1e6;
  return s;
}

void report(const char *label, const MfmaStats &s) {
  std::printf("\n  === %s (CDNA4, wave64) ===\n"
              "  iterations: %d\n"
              "  %7.1f ns/inst  (%6.2f MIPS)  wall %.1f ms\n",
              label, ITERATIONS, s.ns_per_inst, s.mips, static_cast<double>(s.total_ns) / 1e6);
}

TEST(MfmaBenchmark, F32_32x32x8_F16) {
  MfmaBenchFixture fx;
  ASSERT_NE(fx.cu, nullptr);
  ASSERT_NE(fx.wf, nullptr);
  constexpr uint64_t SEED = 0xAF3A'3232'0008ULL;
  fx.seed_f16(SEED, 4, 4, 16);
  uint32_t dst = fx.vb + DST_OFF, s0 = fx.vb + S0_OFF, s1 = fx.vb + S1_OFF, s2 = fx.vb + DST_OFF;
  auto s = time_mfma(
      [&] {
        amdgpu::exec_f32(*fx.cu, 32, 32, 8, 1, 16, dst, s0, s1, s2, amdgpu::extract_f16,
                         amdgpu::extract_f16, amdgpu::ACC_FROM_VGPR);
      },
      ITERATIONS);
  report("v_mfma_f32_32x32x8_f16", s);
  EXPECT_GT(s.mips, 0.01);
}

TEST(MfmaBenchmark, F32_16x16x32_F16) {
  MfmaBenchFixture fx;
  ASSERT_NE(fx.cu, nullptr);
  ASSERT_NE(fx.wf, nullptr);
  constexpr uint64_t SEED = 0xAF3A'1616'0032ULL;
  fx.seed_f16(SEED, 8, 8, 4);
  uint32_t dst = fx.vb + DST_OFF, s0 = fx.vb + S0_OFF, s1 = fx.vb + S1_OFF, s2 = fx.vb + DST_OFF;
  auto s = time_mfma(
      [&] {
        amdgpu::exec_f32(*fx.cu, 16, 16, 32, 1, 16, dst, s0, s1, s2, amdgpu::extract_f16,
                         amdgpu::extract_f16, amdgpu::ACC_FROM_VGPR);
      },
      ITERATIONS);
  report("v_mfma_f32_16x16x32_f16", s);
  EXPECT_GT(s.mips, 0.01);
}

TEST(MfmaBenchmark, F32_4x4x4_F16) {
  MfmaBenchFixture fx;
  ASSERT_NE(fx.cu, nullptr);
  ASSERT_NE(fx.wf, nullptr);
  constexpr uint64_t SEED = 0xAF3A'0404'0004ULL;
  fx.seed_f16(SEED, 1, 1, 1);
  uint32_t dst = fx.vb + DST_OFF, s0 = fx.vb + S0_OFF, s1 = fx.vb + S1_OFF, s2 = fx.vb + DST_OFF;
  auto s = time_mfma(
      [&] {
        amdgpu::exec_f32(*fx.cu, 4, 4, 4, 4, 16, dst, s0, s1, s2, amdgpu::extract_f16,
                         amdgpu::extract_f16, amdgpu::ACC_FROM_VGPR);
      },
      ITERATIONS);
  report("v_mfma_f32_4x4x4_f16", s);
  EXPECT_GT(s.mips, 0.01);
}

TEST(MfmaBenchmark, I32_32x32x8_I8) {
  MfmaBenchFixture fx;
  ASSERT_NE(fx.cu, nullptr);
  ASSERT_NE(fx.wf, nullptr);
  constexpr uint64_t SEED = 0xAF3A'3232'1818ULL;
  fx.seed_i8(SEED, 2, 2, 16);
  uint32_t dst = fx.vb + DST_OFF, s0 = fx.vb + S0_OFF, s1 = fx.vb + S1_OFF, s2 = fx.vb + DST_OFF;
  auto s = time_mfma(
      [&] { amdgpu::exec_i32_i8(*fx.cu, 32, 32, 8, 1, dst, s0, s1, s2, amdgpu::ACC_FROM_VGPR); },
      ITERATIONS);
  report("v_mfma_i32_32x32x8_i8", s);
  EXPECT_GT(s.mips, 0.01);
}

} // namespace
