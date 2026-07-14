// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file wmma_simd_benchmark.cpp
/// @brief A/B microbenchmark: SIMD vs forced-scalar execution of the shared
/// WMMA/SWMMAC execute kernels (exec_wmma_f32 / exec_wmma_i32 / exec_wmma_f16 /
/// exec_swmmac_*), covering the gfx1250 (RDNA4, wave32) matrix shapes that route
/// through them.
///
/// Both modes run in the same process; util::set_force_scalar_for_testing flips
/// the kernel between its dense native-width matmul (SIMD) and the per-output
/// scalar triple-loop. The benchmark drives the execute kernels directly (no
/// decode) so it isolates the matmul body, then reports ns/op and speedup and
/// checks SIMD-vs-scalar agreement:
///   - i32 output: integer MAC is exact, bit-identical.
///   - f32/f16/bf16 output: SIMD uses fused FMA (matching the hardware) while the
///     scalar reference is non-fused, so results agree to a small tolerance.

#include "mma_test_util.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/shared/mma_exec.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/data_types.h"
#include "util/simd.h"
#include "util/simd_test_hooks.h"

#include <gtest/gtest.h>

#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>

namespace {

using namespace rocjitsu;
using Clock = std::chrono::steady_clock;

// Shared with the MFMA benchmark via mma_test_util.h (single source of truth).
constexpr uint32_t WF_SIZE = mma_test::WMMA_WF_SIZE;
constexpr uint32_t SGPRS_PER_WF = mma_test::SGPRS_PER_WF;
constexpr uint32_t VGPRS_PER_WF = mma_test::VGPRS_PER_WF;
constexpr int ITERATIONS = mma_test::BENCH_ITERATIONS;

// VGPR layout: A inputs at s0, B inputs at s1, accumulator/output at S2, sparse
// metadata at INDEX. The 32-register spacing clears the per-operand footprint of
// every shape benchmarked here.
constexpr uint32_t S0_OFF = 0;
constexpr uint32_t S1_OFF = 32;
constexpr uint32_t S2_OFF = 64;
constexpr uint32_t INDEX_OFF = 96;
constexpr uint32_t OUT_REGS = 8; // dst window read back for the correctness check.

constexpr uint32_t INDEX_ENTRIES = 16;
constexpr uint32_t INDEX_KEY = 0;

enum class Cmp { IntExact, F32Tol, F16Tol, Bf16Tol };

struct BenchFixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  amdgpu::Wavefront *wf = nullptr;
  uint32_t vbase = 0;

  BenchFixture() : gpu_mem("wmma_simd_bench_mem"), l2("wmma_simd_bench_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_GFX1250;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_wmma_simd", cfg, &gpu_mem, &l2);
    wf = cu->dispatch_wf(0, 0, SGPRS_PER_WF, VGPRS_PER_WF);
    if (wf)
      vbase = wf->vgpr_alloc().base;
  }

  // Pack `regs` words of small floats (bit_width 16/32 per element) into the
  // VGPR range [vbase+off ..). Both modes read the same bytes, so the exact WMMA
  // lane layout is irrelevant to the A/B comparison.
  template <typename Gen> void seed(uint32_t off, uint32_t regs, uint32_t bit_width, Gen gen) {
    for (uint32_t reg = 0; reg < regs; ++reg)
      for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
        uint32_t word = 0;
        if (bit_width == 32) {
          word = std::bit_cast<uint32_t>(gen());
        } else { // 16-bit f16
          uint16_t lo = util::f32_to_f16(gen());
          uint16_t hi = util::f32_to_f16(gen());
          word = lo | (static_cast<uint32_t>(hi) << 16);
        }
        cu->write_vgpr(vbase + off + reg, lane, word);
      }
  }

  // Pack `regs` words of small bf16 values into the VGPR range.
  template <typename Gen> void seed_bf16(uint32_t off, uint32_t regs, Gen gen) {
    for (uint32_t reg = 0; reg < regs; ++reg)
      for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
        uint16_t lo = util::f32_to_bf16(gen());
        uint16_t hi = util::f32_to_bf16(gen());
        cu->write_vgpr(vbase + off + reg, lane, lo | (static_cast<uint32_t>(hi) << 16));
      }
  }

  // Pack `regs` words of small fp8 (e4m3) codes into the VGPR range. Both modes
  // read the same bytes, so the codes are equally valid for the bf8 paths.
  template <typename Gen> void seed_f8(uint32_t off, uint32_t regs, Gen gen) {
    for (uint32_t reg = 0; reg < regs; ++reg)
      for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
        uint32_t word = 0;
        for (uint32_t byte = 0; byte < 4; ++byte)
          word |= static_cast<uint32_t>(util::f32_to_fp8_e4m3(gen())) << (byte * 8);
        cu->write_vgpr(vbase + off + reg, lane, word);
      }
  }

  template <typename Gen> void seed_i8(uint32_t off, uint32_t regs, Gen gen) {
    for (uint32_t reg = 0; reg < regs; ++reg)
      for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
        uint32_t word = 0;
        for (uint32_t byte = 0; byte < 4; ++byte)
          word |= (static_cast<uint32_t>(static_cast<uint8_t>(gen())) << (byte * 8));
        cu->write_vgpr(vbase + off + reg, lane, word);
      }
  }

  void seed_words(uint32_t off, uint32_t regs, uint32_t value) {
    for (uint32_t reg = 0; reg < regs; ++reg)
      for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
        cu->write_vgpr(vbase + off + reg, lane, value);
  }

  std::vector<uint32_t> snapshot_out() const {
    std::vector<uint32_t> out(OUT_REGS * WF_SIZE);
    for (uint32_t reg = 0; reg < OUT_REGS; ++reg)
      for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
        out[reg * WF_SIZE + lane] = cu->read_vgpr(vbase + S2_OFF + reg, lane);
    return out;
  }
};

double rel_err(float a, float b) {
  if (!std::isfinite(a) || !std::isfinite(b))
    return 0.0;
  return std::abs(static_cast<double>(a) - b) / (std::abs(a) + 1e-6);
}

void compare(const char *label, Cmp cmp, const std::vector<uint32_t> &sc,
             const std::vector<uint32_t> &sd) {
  double max_rel = 0.0;
  for (size_t i = 0; i < sc.size(); ++i) {
    switch (cmp) {
    case Cmp::IntExact: {
      EXPECT_EQ(sc[i], sd[i]) << label << ": int divergence at index " << i;
      break;
    }
    case Cmp::F32Tol:
      max_rel =
          std::max(max_rel, rel_err(std::bit_cast<float>(sc[i]), std::bit_cast<float>(sd[i])));
      break;
    case Cmp::F16Tol:
      for (uint32_t h = 0; h < 2; ++h) {
        float a = util::f16_to_f32(static_cast<uint16_t>((sc[i] >> (h * 16)) & 0xFFFF));
        float b = util::f16_to_f32(static_cast<uint16_t>((sd[i] >> (h * 16)) & 0xFFFF));
        max_rel = std::max(max_rel, rel_err(a, b));
      }
      break;
    case Cmp::Bf16Tol:
      for (uint32_t h = 0; h < 2; ++h) {
        float a = util::bf16_to_f32(static_cast<uint16_t>((sc[i] >> (h * 16)) & 0xFFFF));
        float b = util::bf16_to_f32(static_cast<uint16_t>((sd[i] >> (h * 16)) & 0xFFFF));
        max_rel = std::max(max_rel, rel_err(a, b));
      }
      break;
    }
  }
  if (cmp != Cmp::IntExact) {
    EXPECT_LT(max_rel, 5e-3) << label << ": fused-vs-nonfused relative error too large";
  }
}

// Drive one WMMA kernel A/B: run scalar then SIMD, compare dst, then time both.
void bench(const char *label, BenchFixture &fx, const std::function<void()> &run, double macs,
           Cmp cmp) {
  ASSERT_NE(fx.cu, nullptr);
  ASSERT_NE(fx.wf, nullptr);

  util::set_force_scalar_for_testing(true);
  run();
  auto result_scalar = fx.snapshot_out();
  util::set_force_scalar_for_testing(false);
  run();
  auto result_simd = fx.snapshot_out();
  compare(label, cmp, result_scalar, result_simd);

  auto time_mode = [&](bool force_scalar) -> double {
    util::set_force_scalar_for_testing(force_scalar);
    for (int i = 0; i < 50; ++i)
      run();
    auto t0 = Clock::now();
    for (int i = 0; i < ITERATIONS; ++i)
      run();
    auto t1 = Clock::now();
    util::set_force_scalar_for_testing(false);
    return std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() /
           static_cast<double>(ITERATIONS);
  };
  double sc = time_mode(true);
  double sd = time_mode(false);
  std::printf("\n  === %s (gfx1250, wave32) ===\n"
              "  MACs/op: %.0f   scalar: %9.1f ns   simd: %9.1f ns   speedup: %5.2fx\n",
              label, macs, sc, sd, (sd > 0) ? sc / sd : 0.0);
  EXPECT_GT(sc, 0.0);
  EXPECT_GT(sd, 0.0);
}

} // namespace

// Dense WMMA, f32 output, f16 input (the common transformer shape).
TEST(WmmaSimdBenchmark, F32_16x16x16_f16) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 16, N = 16, K = 16, bits = 16;
  fx.seed(S0_OFF, 16, bits, mma_test::SmallGen(1));
  fx.seed(S1_OFF, 16, bits, mma_test::SmallGen(2));
  auto run = [&] {
    amdgpu::exec_wmma_f32(*fx.cu, M, N, K, bits, fx.vbase + S2_OFF, fx.vbase + S0_OFF,
                          fx.vbase + S1_OFF, fx.vbase + S2_OFF, amdgpu::extract_f16,
                          amdgpu::extract_f16, /*const_acc=*/0);
  };
  bench("v_wmma_f32_16x16x16_f16", fx, run, double(M) * N * K, Cmp::F32Tol);
}

// Dense WMMA, f16 output (packed 2-per-word), f16 input.
TEST(WmmaSimdBenchmark, F16_16x16x16_f16) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 16, N = 16, K = 16, bits = 16;
  fx.seed(S0_OFF, 16, bits, mma_test::SmallGen(3));
  fx.seed(S1_OFF, 16, bits, mma_test::SmallGen(4));
  auto run = [&] {
    amdgpu::exec_wmma_f16(*fx.cu, M, N, K, bits, fx.vbase + S2_OFF, fx.vbase + S0_OFF,
                          fx.vbase + S1_OFF, fx.vbase + S2_OFF, amdgpu::extract_f16,
                          amdgpu::extract_f16, /*const_acc=*/0);
  };
  bench("v_wmma_f16_16x16x16_f16", fx, run, double(M) * N * K, Cmp::F16Tol);
}

// Dense WMMA, bf16 output (packed 2-per-word), bf16 input.
TEST(WmmaSimdBenchmark, Bf16_16x16x16_bf16) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 16, N = 16, K = 16, bits = 16;
  fx.seed(S0_OFF, 16, bits, mma_test::SmallGen(5));
  fx.seed(S1_OFF, 16, bits, mma_test::SmallGen(6));
  auto run = [&] {
    amdgpu::exec_wmma_bf16(*fx.cu, M, N, K, bits, fx.vbase + S2_OFF, fx.vbase + S0_OFF,
                           fx.vbase + S1_OFF, fx.vbase + S2_OFF, amdgpu::extract_bf16,
                           amdgpu::extract_bf16, /*const_acc=*/0);
  };
  bench("v_wmma_bf16_16x16x16_bf16", fx, run, double(M) * N * K, Cmp::Bf16Tol);
}

// Dense WMMA, i32 output, i8 input — integer MAC is exact (bit-identical).
TEST(WmmaSimdBenchmark, I32_16x16x16_i8) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 16, N = 16, K = 16;
  fx.seed_i8(S0_OFF, 16, mma_test::SmallI8Gen(7));
  fx.seed_i8(S1_OFF, 16, mma_test::SmallI8Gen(8));
  auto run = [&] {
    amdgpu::exec_wmma_i32_i8(*fx.cu, M, N, K, fx.vbase + S2_OFF, fx.vbase + S0_OFF,
                             fx.vbase + S1_OFF, fx.vbase + S2_OFF, /*const_acc=*/0);
  };
  bench("v_wmma_i32_16x16x16_i8", fx, run, double(M) * N * K, Cmp::IntExact);
}

// Sparse SWMMAC, f32 output, f16 input (per-row 2:4 gather).
TEST(WmmaSimdBenchmark, SwmmacF32_16x16x32_f16) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 16, N = 16, K = 32, bits = 16;
  fx.seed(S0_OFF, 16, bits, mma_test::SmallGen(9));
  fx.seed(S1_OFF, 16, bits, mma_test::SmallGen(10));
  fx.seed_words(INDEX_OFF, 2, 0xE4E4E4E4u); // arbitrary but valid 2-bit index set
  auto run = [&] {
    amdgpu::exec_swmmac_f32(*fx.cu, M, N, K, bits, fx.vbase + S2_OFF, fx.vbase + S0_OFF,
                            fx.vbase + S1_OFF, fx.vbase + S2_OFF, fx.vbase + INDEX_OFF,
                            INDEX_ENTRIES, INDEX_KEY, amdgpu::extract_f16, amdgpu::extract_f16,
                            /*const_acc=*/0);
  };
  bench("v_swmmac_f32_16x16x32_f16", fx, run, double(M) * N * (K / 2), Cmp::F32Tol);
}

// Sparse SWMMAC, i32 output, i8 input — exact.
TEST(WmmaSimdBenchmark, SwmmacI32_16x16x32_i8) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 16, N = 16, K = 32;
  fx.seed_i8(S0_OFF, 16, mma_test::SmallI8Gen(11));
  fx.seed_i8(S1_OFF, 16, mma_test::SmallI8Gen(12));
  fx.seed_words(INDEX_OFF, 2, 0xE4E4E4E4u);
  auto run = [&] {
    amdgpu::exec_swmmac_i32_i8(*fx.cu, M, N, K, fx.vbase + S2_OFF, fx.vbase + S0_OFF,
                               fx.vbase + S1_OFF, fx.vbase + S2_OFF, fx.vbase + INDEX_OFF,
                               INDEX_ENTRIES, INDEX_KEY, /*const_acc=*/0);
  };
  bench("v_swmmac_i32_16x16x32_i8", fx, run, double(M) * N * (K / 2), Cmp::IntExact);
}

// Dense WMMA, f32 output, f16 input, K=32 — the real gfx1250 shape, generic path.
TEST(WmmaSimdBenchmark, F32_16x16x32_f16) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 16, N = 16, K = 32, bits = 16;
  fx.seed(S0_OFF, 16, bits, mma_test::SmallGen(13));
  fx.seed(S1_OFF, 16, bits, mma_test::SmallGen(14));
  auto run = [&] {
    amdgpu::exec_wmma_f32(*fx.cu, M, N, K, bits, fx.vbase + S2_OFF, fx.vbase + S0_OFF,
                          fx.vbase + S1_OFF, fx.vbase + S2_OFF, amdgpu::extract_f16,
                          amdgpu::extract_f16, /*const_acc=*/0);
  };
  bench("v_wmma_f32_16x16x32_f16", fx, run, double(M) * N * K, Cmp::F32Tol);
}

// Dedicated constexpr-dims + F16C specialization for the same shape.
TEST(WmmaSimdBenchmark, F32_16x16x32_f16_Specialized) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 16, N = 16, K = 32, bits = 16;
  fx.seed(S0_OFF, 16, bits, mma_test::SmallGen(13));
  fx.seed(S1_OFF, 16, bits, mma_test::SmallGen(14));
  auto run = [&] {
    amdgpu::exec_wmma_f32_16x16x32_f16(*fx.cu, fx.vbase + S2_OFF, fx.vbase + S0_OFF,
                                       fx.vbase + S1_OFF, fx.vbase + S2_OFF, /*const_acc=*/0);
  };
  bench("v_wmma_f32_16x16x32_f16 [specialized]", fx, run, double(M) * N * K, Cmp::F32Tol);
}

// Dense WMMA, f32 output, f32 input (no convert), specialized.
TEST(WmmaSimdBenchmark, F32_16x16x4_f32_Specialized) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 16, N = 16, K = 4, bits = 32;
  fx.seed(S0_OFF, 16, bits, mma_test::SmallGen(41));
  fx.seed(S1_OFF, 16, bits, mma_test::SmallGen(42));
  auto run = [&] {
    amdgpu::exec_wmma_f32_f32_spec<16, 16, 4>(*fx.cu, fx.vbase + S2_OFF, fx.vbase + S0_OFF,
                                              fx.vbase + S1_OFF, fx.vbase + S2_OFF,
                                              /*const_acc=*/0);
  };
  bench("v_wmma_f32_16x16x4_f32 [specialized]", fx, run, double(M) * N * K, Cmp::F32Tol);
}

// Dense WMMA, f32 output, bf16 input, K=32 — generic baseline vs specialized
// (constexpr + bf16 bulk zero-extend convert).
TEST(WmmaSimdBenchmark, F32_16x16x32_bf16) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 16, N = 16, K = 32, bits = 16;
  fx.seed_bf16(S0_OFF, 16, mma_test::SmallGen(61));
  fx.seed_bf16(S1_OFF, 16, mma_test::SmallGen(62));
  auto run = [&] {
    amdgpu::exec_wmma_f32(*fx.cu, M, N, K, bits, fx.vbase + S2_OFF, fx.vbase + S0_OFF,
                          fx.vbase + S1_OFF, fx.vbase + S2_OFF, amdgpu::extract_bf16,
                          amdgpu::extract_bf16, /*const_acc=*/0);
  };
  bench("v_wmma_f32_16x16x32_bf16", fx, run, double(M) * N * K, Cmp::F32Tol);
}

TEST(WmmaSimdBenchmark, F32_16x16x32_bf16_Specialized) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 16, N = 16, K = 32;
  fx.seed_bf16(S0_OFF, 16, mma_test::SmallGen(61));
  fx.seed_bf16(S1_OFF, 16, mma_test::SmallGen(62));
  auto run = [&] {
    amdgpu::exec_wmma_f32_16x16x32_bf16(*fx.cu, fx.vbase + S2_OFF, fx.vbase + S0_OFF,
                                        fx.vbase + S1_OFF, fx.vbase + S2_OFF, /*const_acc=*/0);
  };
  bench("v_wmma_f32_16x16x32_bf16 [specialized]", fx, run, double(M) * N * K, Cmp::F32Tol);
}

// Dense WMMA, bf16 output, bf16 input, K=32 — specialized packed16 path.
TEST(WmmaSimdBenchmark, Bf16_16x16x32_bf16_Specialized) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 16, N = 16, K = 32;
  fx.seed_bf16(S0_OFF, 16, mma_test::SmallGen(63));
  fx.seed_bf16(S1_OFF, 16, mma_test::SmallGen(64));
  auto run = [&] {
    amdgpu::exec_wmma_bf16_spec<16, 16, 32>(*fx.cu, fx.vbase + S2_OFF, fx.vbase + S0_OFF,
                                            fx.vbase + S1_OFF, fx.vbase + S2_OFF, /*const_acc=*/0);
  };
  bench("v_wmma_bf16_16x16x32_bf16 [specialized]", fx, run, double(M) * N * K, Cmp::Bf16Tol);
}

// Dense WMMA, i32 output, iu8 input, K=64 — the real gfx1250 integer shape,
// generic baseline vs specialized (constexpr + i8 bulk sign-extend convert).
TEST(WmmaSimdBenchmark, I32_16x16x64_iu8) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 16, N = 16, K = 64;
  fx.seed_i8(S0_OFF, 16, mma_test::SmallI8Gen(71));
  fx.seed_i8(S1_OFF, 16, mma_test::SmallI8Gen(72));
  auto run = [&] {
    amdgpu::exec_wmma_i32(*fx.cu, M, N, K, 8, fx.vbase + S2_OFF, fx.vbase + S0_OFF,
                          fx.vbase + S1_OFF, fx.vbase + S2_OFF, amdgpu::extract_i8,
                          amdgpu::extract_u8, /*clamp=*/false, /*const_acc=*/0);
  };
  bench("v_wmma_i32_16x16x64_iu8", fx, run, double(M) * N * K, Cmp::IntExact);
}

TEST(WmmaSimdBenchmark, I32_16x16x64_iu8_Specialized) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 16, N = 16, K = 64;
  fx.seed_i8(S0_OFF, 16, mma_test::SmallI8Gen(71));
  fx.seed_i8(S1_OFF, 16, mma_test::SmallI8Gen(72));
  auto run = [&] {
    amdgpu::exec_wmma_i32_16x16x64_iu8(*fx.cu, fx.vbase + S2_OFF, fx.vbase + S0_OFF,
                                       fx.vbase + S1_OFF, fx.vbase + S2_OFF, /*a_signed=*/true,
                                       /*b_signed=*/false, /*clamp=*/false, /*const_acc=*/0);
  };
  bench("v_wmma_i32_16x16x64_iu8 [specialized]", fx, run, double(M) * N * K, Cmp::IntExact);
}

// Dense WMMA, f32 output, fp8 input, K=64 — generic baseline vs specialized
// (constexpr + 256-entry LUT bulk convert).
TEST(WmmaSimdBenchmark, F32_16x16x64_fp8) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 16, N = 16, K = 64;
  fx.seed_f8(S0_OFF, 16, mma_test::SmallGen(81));
  fx.seed_f8(S1_OFF, 16, mma_test::SmallGen(82));
  auto run = [&] {
    amdgpu::exec_wmma_f32(*fx.cu, M, N, K, 8, fx.vbase + S2_OFF, fx.vbase + S0_OFF,
                          fx.vbase + S1_OFF, fx.vbase + S2_OFF, amdgpu::extract_fp8,
                          amdgpu::extract_fp8, /*const_acc=*/0);
  };
  bench("v_wmma_f32_16x16x64_fp8_fp8", fx, run, double(M) * N * K, Cmp::F32Tol);
}

TEST(WmmaSimdBenchmark, F32_16x16x64_fp8_Specialized) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 16, N = 16, K = 64;
  fx.seed_f8(S0_OFF, 16, mma_test::SmallGen(81));
  fx.seed_f8(S1_OFF, 16, mma_test::SmallGen(82));
  auto run = [&] {
    amdgpu::exec_wmma_f32_f8_spec<16, 16, 64, true, true>(*fx.cu, fx.vbase + S2_OFF,
                                                          fx.vbase + S0_OFF, fx.vbase + S1_OFF,
                                                          fx.vbase + S2_OFF, /*const_acc=*/0);
  };
  bench("v_wmma_f32_16x16x64_fp8_fp8 [specialized]", fx, run, double(M) * N * K, Cmp::F32Tol);
}

// Dense WMMA, f32 output, bf8 input, K=128 — specialized.
TEST(WmmaSimdBenchmark, F32_16x16x128_bf8_Specialized) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 16, N = 16, K = 128;
  fx.seed_f8(S0_OFF, 16, mma_test::SmallGen(83));
  fx.seed_f8(S1_OFF, 16, mma_test::SmallGen(84));
  auto run = [&] {
    amdgpu::exec_wmma_f32_f8_spec<16, 16, 128, false, false>(*fx.cu, fx.vbase + S2_OFF,
                                                             fx.vbase + S0_OFF, fx.vbase + S1_OFF,
                                                             fx.vbase + S2_OFF, /*const_acc=*/0);
  };
  bench("v_wmma_f32_16x16x128_bf8_bf8 [specialized]", fx, run, double(M) * N * K, Cmp::F32Tol);
}

// Dense WMMA, f16 output, fp8 input, K=64 — specialized packed16 path.
TEST(WmmaSimdBenchmark, F16_16x16x64_fp8_Specialized) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 16, N = 16, K = 64;
  fx.seed_f8(S0_OFF, 16, mma_test::SmallGen(85));
  fx.seed_f8(S1_OFF, 16, mma_test::SmallGen(86));
  auto run = [&] {
    amdgpu::exec_wmma_f16_f8_spec<16, 16, 64, true, true>(*fx.cu, fx.vbase + S2_OFF,
                                                          fx.vbase + S0_OFF, fx.vbase + S1_OFF,
                                                          fx.vbase + S2_OFF, /*const_acc=*/0);
  };
  bench("v_wmma_f16_16x16x64_fp8_fp8 [specialized]", fx, run, double(M) * N * K, Cmp::F16Tol);
}

// Dense WMMA, f16 output, f16 input, K=32 — specialized (constexpr + F16C).
TEST(WmmaSimdBenchmark, F16_16x16x32_f16_Specialized) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 16, N = 16, K = 32, bits = 16;
  fx.seed(S0_OFF, 16, bits, mma_test::SmallGen(51));
  fx.seed(S1_OFF, 16, bits, mma_test::SmallGen(52));
  auto run = [&] {
    amdgpu::exec_wmma_f16_spec<16, 16, 32>(*fx.cu, fx.vbase + S2_OFF, fx.vbase + S0_OFF,
                                           fx.vbase + S1_OFF, fx.vbase + S2_OFF, /*const_acc=*/0);
  };
  bench("v_wmma_f16_16x16x32_f16 [specialized]", fx, run, double(M) * N * K, Cmp::F16Tol);
}
