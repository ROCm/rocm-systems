// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file mfma_simd_benchmark.cpp
/// @brief A/B microbenchmark: SIMD vs forced-scalar execution of the shared
/// MFMA execute kernels (exec_f32 / exec_i32_i8 / exec_f64), covering the
/// CDNA4 MFMA shapes that route through them.
///
/// Both modes run in the same process; the `amdgpu::simd_force_scalar()`
/// thread-local flag flips the kernel between its dense native-width matmul
/// (SIMD) and the per-output scalar triple-loop. The benchmark drives the
/// execute kernels directly (no decode) so it isolates the matmul body, then
/// reports ns/MFMA and speedup and checks SIMD-vs-scalar agreement:
///   - i32/i8 output: integer MAC is exact, bit-identical.
///   - f32/f64 output: SIMD uses fused FMA (matching GFX9 MFMA hardware) while
///     the scalar reference is non-fused, so results agree to a small relative
///     tolerance, not bit-exactly.

#include "mma_test_util.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/shared/mma_exec.h"
#include "rocjitsu/isa/arch/amdgpu/shared/simd_glue.h"
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

// Shared with the WMMA benchmark via mma_test_util.h (single source of truth).
constexpr uint32_t WF_SIZE = mma_test::MFMA_WF_SIZE;
constexpr uint32_t SGPRS_PER_WF = mma_test::SGPRS_PER_WF;
constexpr uint32_t VGPRS_PER_WF = mma_test::VGPRS_PER_WF;
constexpr int ITERATIONS = mma_test::BENCH_ITERATIONS;

// VGPR layout: A inputs at s0, B inputs at s1, output at DST. The 64-register
// spacing comfortably clears the small per-operand register footprint of every
// shape benchmarked here.
constexpr uint32_t S0_OFF = 0;
constexpr uint32_t S1_OFF = 64;
constexpr uint32_t DST_OFF = 128;
constexpr uint32_t OUT_REGS = 32; // dst window read back for the correctness check.

struct BenchFixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  amdgpu::Wavefront *wf = nullptr;
  uint32_t vbase = 0;

  BenchFixture() : gpu_mem("mfma_simd_bench_mem"), l2("mfma_simd_bench_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_mfma_simd", cfg, &gpu_mem, &l2);
    wf = cu->dispatch_wf(0, 0, SGPRS_PER_WF, VGPRS_PER_WF);
    if (wf)
      vbase = wf->vgpr_alloc().base;
  }

  // Pack `n` small floats produced by `gen` into the source VGPR range
  // [vbase+off .. ), bit_width bits per element (8/16/32), row-major over
  // (reg, lane). The exact layout doesn't matter for the benchmark — both
  // modes read the same bytes — only that values are finite and bounded so
  // the accumulated matmul stays finite.
  template <typename Gen> void seed(uint32_t off, uint32_t regs, uint32_t bit_width, Gen gen) {
    for (uint32_t reg = 0; reg < regs; ++reg) {
      for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
        uint32_t word = 0;
        if (bit_width == 32) {
          word = std::bit_cast<uint32_t>(gen());
        } else if (bit_width == 16) {
          uint16_t lo = util::f32_to_f16(gen());
          uint16_t hi = util::f32_to_f16(gen());
          word = lo | (static_cast<uint32_t>(hi) << 16);
        } else { // 8-bit fp8 e4m3
          for (uint32_t byte = 0; byte < 4; ++byte)
            word |= static_cast<uint32_t>(util::f32_to_fp8_e4m3(gen())) << (byte * 8);
        }
        cu->write_vgpr(vbase + off + reg, lane, word);
      }
    }
  }

  // Pack `regs` words of small bf16 values into the source VGPR range.
  template <typename Gen> void seed_bf16(uint32_t off, uint32_t regs, Gen gen) {
    for (uint32_t reg = 0; reg < regs; ++reg)
      for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
        uint16_t lo = util::f32_to_bf16(gen());
        uint16_t hi = util::f32_to_bf16(gen());
        cu->write_vgpr(vbase + off + reg, lane, lo | (static_cast<uint32_t>(hi) << 16));
      }
  }

  // Pack `regs` words of small int8 values into the source VGPR range.
  template <typename Gen> void seed_i8(uint32_t off, uint32_t regs, Gen gen) {
    for (uint32_t reg = 0; reg < regs; ++reg)
      for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
        uint32_t word = 0;
        for (uint32_t byte = 0; byte < 4; ++byte)
          word |= (static_cast<uint32_t>(static_cast<uint8_t>(gen())) << (byte * 8));
        cu->write_vgpr(vbase + off + reg, lane, word);
      }
  }

  // Pack `regs` pairs (lo,hi) of small f64 values into the source range.
  template <typename Gen> void seed_f64(uint32_t off, uint32_t reg_pairs, Gen gen) {
    for (uint32_t p = 0; p < reg_pairs; ++p)
      for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
        uint64_t bits = std::bit_cast<uint64_t>(static_cast<double>(gen()));
        cu->write_vgpr(vbase + off + 2 * p, lane, static_cast<uint32_t>(bits));
        cu->write_vgpr(vbase + off + 2 * p + 1, lane, static_cast<uint32_t>(bits >> 32));
      }
  }

  std::vector<uint32_t> snapshot_out() const {
    std::vector<uint32_t> out(OUT_REGS * WF_SIZE);
    for (uint32_t reg = 0; reg < OUT_REGS; ++reg)
      for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
        out[reg * WF_SIZE + lane] = cu->read_vgpr(vbase + DST_OFF + reg, lane);
    return out;
  }
};

struct ModeStats {
  double ns_per_inst = 0;
  double gmacs = 0; // billion MACs/s
  uint64_t total_ns = 0;
};

ModeStats time_mode(const std::function<void()> &run, bool force_scalar, double macs_per_call) {
  util::set_force_scalar_for_testing(force_scalar);
  for (int i = 0; i < 50; ++i) // warmup
    run();
  auto t0 = Clock::now();
  for (int i = 0; i < ITERATIONS; ++i)
    run();
  auto t1 = Clock::now();
  util::set_force_scalar_for_testing(false);
  ModeStats s;
  s.total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
  s.ns_per_inst = static_cast<double>(s.total_ns) / ITERATIONS;
  s.gmacs = (macs_per_call * ITERATIONS) / static_cast<double>(s.total_ns); // MACs/ns == G/s
  return s;
}

// Drive one MFMA kernel A/B. `run` invokes the kernel once (reads s0/s1,
// writes dst). `is_int` selects exact vs tolerance correctness comparison.
void bench(const char *label, BenchFixture &fx, const std::function<void()> &run, double macs,
           bool is_int) {
  ASSERT_NE(fx.cu, nullptr);
  ASSERT_NE(fx.wf, nullptr);

  util::set_force_scalar_for_testing(true);
  run();
  auto result_scalar = fx.snapshot_out();
  util::set_force_scalar_for_testing(false);
  run();
  auto result_simd = fx.snapshot_out();

  if (is_int) {
    for (size_t i = 0; i < result_scalar.size(); ++i)
      EXPECT_EQ(result_scalar[i], result_simd[i]) << label << ": int divergence at index " << i;
  } else {
    double max_rel = 0.0;
    for (size_t i = 0; i < result_scalar.size(); ++i) {
      float a = std::bit_cast<float>(result_scalar[i]);
      float b = std::bit_cast<float>(result_simd[i]);
      if (!std::isfinite(a) || !std::isfinite(b))
        continue;
      double denom = std::abs(a) + 1e-6;
      max_rel = std::max(max_rel, std::abs(static_cast<double>(a) - b) / denom);
    }
    EXPECT_LT(max_rel, 1e-3) << label << ": fused-vs-nonfused relative error too large";
  }

  ModeStats sc = time_mode(run, /*force_scalar=*/true, macs);
  ModeStats sd = time_mode(run, /*force_scalar=*/false, macs);
  double speedup = (sd.ns_per_inst > 0) ? (sc.ns_per_inst / sd.ns_per_inst) : 0.0;
  std::printf("\n  === %s (CDNA4, wave64) ===\n"
              "  iterations: %d   MACs/MFMA: %.0f\n"
              "  scalar : %9.1f ns/MFMA  (%6.2f GMAC/s)  wall %.1f ms\n"
              "  simd   : %9.1f ns/MFMA  (%6.2f GMAC/s)  wall %.1f ms\n"
              "  speedup: %5.2fx\n",
              label, ITERATIONS, macs, sc.ns_per_inst, sc.gmacs,
              static_cast<double>(sc.total_ns) / 1e6, sd.ns_per_inst, sd.gmacs,
              static_cast<double>(sd.total_ns) / 1e6, speedup);

  EXPECT_GT(sc.gmacs, 0.0) << label << ": scalar baseline produced no work";
  EXPECT_GT(sd.gmacs, 0.0) << label << ": simd path produced no work";
}

} // namespace

// v_mfma_f32_16x16x32_f16: the dominant MFMA shape in fp16 transformer
// forward passes (N=16 == one AVX-512 f32 lane group).
TEST(MfmaSimdBenchmark, F32_16x16x32_f16) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 16, N = 16, K = 32, B = 1, bits = 16;
  fx.seed(S0_OFF, /*regs=*/8, bits, mma_test::SmallGen(1));
  fx.seed(S1_OFF, /*regs=*/8, bits, mma_test::SmallGen(2));
  auto run = [&] {
    amdgpu::exec_f32(*fx.cu, M, N, K, B, bits, fx.vbase + DST_OFF, fx.vbase + S0_OFF,
                     fx.vbase + S1_OFF, 0, amdgpu::extract_f16, amdgpu::extract_f16,
                     /*const_acc=*/0);
  };
  bench("v_mfma_f32_16x16x32_f16", fx, run, double(M) * N * K * B, /*is_int=*/false);
}

// Dedicated hot-path specialization (the function the generated OPT-125M call
// site actually invokes): constexpr dims + F16C bulk f16->f32 conversion.
TEST(MfmaSimdBenchmark, F32_16x16x32_f16_Specialized) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 16, N = 16, K = 32, B = 1, bits = 16;
  fx.seed(S0_OFF, /*regs=*/8, bits, mma_test::SmallGen(1));
  fx.seed(S1_OFF, /*regs=*/8, bits, mma_test::SmallGen(2));
  auto run = [&] {
    amdgpu::exec_f32_mfma_f16_spec<16, 16, 32>(*fx.cu, fx.vbase + DST_OFF, fx.vbase + S0_OFF,
                                               fx.vbase + S1_OFF, 0, /*const_acc=*/0, /*cbsz=*/0,
                                               /*abid=*/0, /*blgp=*/0);
  };
  bench("v_mfma_f32_16x16x32_f16 [specialized]", fx, run, double(M) * N * K * B, /*is_int=*/false);
}

// Remaining cdna4 f16 MFMA shapes, now also specialized.
TEST(MfmaSimdBenchmark, F32_32x32x8_f16_Specialized) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 32, N = 32, K = 8, B = 1, bits = 16;
  fx.seed(S0_OFF, 8, bits, mma_test::SmallGen(21));
  fx.seed(S1_OFF, 8, bits, mma_test::SmallGen(22));
  auto run = [&] {
    amdgpu::exec_f32_mfma_f16_spec<32, 32, 8>(*fx.cu, fx.vbase + DST_OFF, fx.vbase + S0_OFF,
                                              fx.vbase + S1_OFF, 0, 0, 0, 0, 0);
  };
  bench("v_mfma_f32_32x32x8_f16 [specialized]", fx, run, double(M) * N * K * B, /*is_int=*/false);
}

TEST(MfmaSimdBenchmark, F32_16x16x16_f16_Specialized) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 16, N = 16, K = 16, B = 1, bits = 16;
  fx.seed(S0_OFF, 8, bits, mma_test::SmallGen(23));
  fx.seed(S1_OFF, 8, bits, mma_test::SmallGen(24));
  auto run = [&] {
    amdgpu::exec_f32_mfma_f16_spec<16, 16, 16>(*fx.cu, fx.vbase + DST_OFF, fx.vbase + S0_OFF,
                                               fx.vbase + S1_OFF, 0, 0, 0, 0, 0);
  };
  bench("v_mfma_f32_16x16x16_f16 [specialized]", fx, run, double(M) * N * K * B, /*is_int=*/false);
}

TEST(MfmaSimdBenchmark, F32_32x32x16_f16_Specialized) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 32, N = 32, K = 16, B = 1, bits = 16;
  fx.seed(S0_OFF, 8, bits, mma_test::SmallGen(25));
  fx.seed(S1_OFF, 8, bits, mma_test::SmallGen(26));
  auto run = [&] {
    amdgpu::exec_f32_mfma_f16_spec<32, 32, 16>(*fx.cu, fx.vbase + DST_OFF, fx.vbase + S0_OFF,
                                               fx.vbase + S1_OFF, 0, 0, 0, 0, 0);
  };
  bench("v_mfma_f32_32x32x16_f16 [specialized]", fx, run, double(M) * N * K * B, /*is_int=*/false);
}

// v_mfma_f32_32x32x16_f16: N=32 == two AVX-512 f32 lane groups.
TEST(MfmaSimdBenchmark, F32_32x32x16_f16) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 32, N = 32, K = 16, B = 1, bits = 16;
  fx.seed(S0_OFF, 8, bits, mma_test::SmallGen(3));
  fx.seed(S1_OFF, 8, bits, mma_test::SmallGen(4));
  auto run = [&] {
    amdgpu::exec_f32(*fx.cu, M, N, K, B, bits, fx.vbase + DST_OFF, fx.vbase + S0_OFF,
                     fx.vbase + S1_OFF, 0, amdgpu::extract_f16, amdgpu::extract_f16,
                     /*const_acc=*/0);
  };
  bench("v_mfma_f32_32x32x16_f16", fx, run, double(M) * N * K * B, /*is_int=*/false);
}

// bf16 shapes: generic baseline + the four specialized kernels (bf16 bulk
// zero-extend convert instead of F16C).
TEST(MfmaSimdBenchmark, F32_16x16x32_bf16) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 16, N = 16, K = 32, B = 1, bits = 16;
  fx.seed_bf16(S0_OFF, 8, mma_test::SmallGen(41));
  fx.seed_bf16(S1_OFF, 8, mma_test::SmallGen(42));
  auto run = [&] {
    amdgpu::exec_f32(*fx.cu, M, N, K, B, bits, fx.vbase + DST_OFF, fx.vbase + S0_OFF,
                     fx.vbase + S1_OFF, 0, amdgpu::extract_bf16, amdgpu::extract_bf16,
                     /*const_acc=*/0);
  };
  bench("v_mfma_f32_16x16x32_bf16", fx, run, double(M) * N * K * B, /*is_int=*/false);
}

TEST(MfmaSimdBenchmark, F32_16x16x32_bf16_Specialized) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 16, N = 16, K = 32, B = 1;
  fx.seed_bf16(S0_OFF, 8, mma_test::SmallGen(41));
  fx.seed_bf16(S1_OFF, 8, mma_test::SmallGen(42));
  auto run = [&] {
    amdgpu::exec_f32_mfma_bf16_spec<16, 16, 32>(*fx.cu, fx.vbase + DST_OFF, fx.vbase + S0_OFF,
                                                fx.vbase + S1_OFF, 0, 0, 0, 0, 0);
  };
  bench("v_mfma_f32_16x16x32_bf16 [specialized]", fx, run, double(M) * N * K * B,
        /*is_int=*/false);
}

TEST(MfmaSimdBenchmark, F32_32x32x8_bf16_Specialized) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 32, N = 32, K = 8, B = 1;
  fx.seed_bf16(S0_OFF, 8, mma_test::SmallGen(43));
  fx.seed_bf16(S1_OFF, 8, mma_test::SmallGen(44));
  auto run = [&] {
    amdgpu::exec_f32_mfma_bf16_spec<32, 32, 8>(*fx.cu, fx.vbase + DST_OFF, fx.vbase + S0_OFF,
                                               fx.vbase + S1_OFF, 0, 0, 0, 0, 0);
  };
  bench("v_mfma_f32_32x32x8_bf16 [specialized]", fx, run, double(M) * N * K * B, /*is_int=*/false);
}

TEST(MfmaSimdBenchmark, F32_16x16x16_bf16_Specialized) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 16, N = 16, K = 16, B = 1;
  fx.seed_bf16(S0_OFF, 8, mma_test::SmallGen(45));
  fx.seed_bf16(S1_OFF, 8, mma_test::SmallGen(46));
  auto run = [&] {
    amdgpu::exec_f32_mfma_bf16_spec<16, 16, 16>(*fx.cu, fx.vbase + DST_OFF, fx.vbase + S0_OFF,
                                                fx.vbase + S1_OFF, 0, 0, 0, 0, 0);
  };
  bench("v_mfma_f32_16x16x16_bf16 [specialized]", fx, run, double(M) * N * K * B,
        /*is_int=*/false);
}

TEST(MfmaSimdBenchmark, F32_32x32x16_bf16_Specialized) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 32, N = 32, K = 16, B = 1;
  fx.seed_bf16(S0_OFF, 8, mma_test::SmallGen(47));
  fx.seed_bf16(S1_OFF, 8, mma_test::SmallGen(48));
  auto run = [&] {
    amdgpu::exec_f32_mfma_bf16_spec<32, 32, 16>(*fx.cu, fx.vbase + DST_OFF, fx.vbase + S0_OFF,
                                                fx.vbase + S1_OFF, 0, 0, 0, 0, 0);
  };
  bench("v_mfma_f32_32x32x16_bf16 [specialized]", fx, run, double(M) * N * K * B,
        /*is_int=*/false);
}

// v_mfma_f32_16x16x32_bf8 (fp8 path, in_bits=8, N=16).
TEST(MfmaSimdBenchmark, F32_16x16x32_fp8) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 16, N = 16, K = 32, B = 1, bits = 8;
  fx.seed(S0_OFF, 8, bits, mma_test::SmallGen(5));
  fx.seed(S1_OFF, 8, bits, mma_test::SmallGen(6));
  auto run = [&] {
    amdgpu::exec_f32(*fx.cu, M, N, K, B, bits, fx.vbase + DST_OFF, fx.vbase + S0_OFF,
                     fx.vbase + S1_OFF, 0, amdgpu::extract_fp8, amdgpu::extract_fp8,
                     /*const_acc=*/0);
  };
  bench("v_mfma_f32_16x16x32_fp8", fx, run, double(M) * N * K * B, /*is_int=*/false);
}

// Dense fp8/bf8 shapes: dedicated constexpr specializations (256-entry LUT
// bulk convert instead of per-element extract_fp8/extract_bf8).
TEST(MfmaSimdBenchmark, F32_16x16x32_fp8_Specialized) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 16, N = 16, K = 32, B = 1, bits = 8;
  fx.seed(S0_OFF, 8, bits, mma_test::SmallGen(5));
  fx.seed(S1_OFF, 8, bits, mma_test::SmallGen(6));
  auto run = [&] {
    amdgpu::exec_f32_mfma_f8_spec<16, 16, 32, true, true>(*fx.cu, fx.vbase + DST_OFF,
                                                          fx.vbase + S0_OFF, fx.vbase + S1_OFF, 0,
                                                          /*const_acc=*/0, 0, 0, 0);
  };
  bench("v_mfma_f32_16x16x32_fp8_fp8 [specialized]", fx, run, double(M) * N * K * B,
        /*is_int=*/false);
}

TEST(MfmaSimdBenchmark, F32_32x32x16_bf8_Specialized) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 32, N = 32, K = 16, B = 1, bits = 8;
  fx.seed(S0_OFF, 8, bits, mma_test::SmallGen(15));
  fx.seed(S1_OFF, 8, bits, mma_test::SmallGen(16));
  auto run = [&] {
    amdgpu::exec_f32_mfma_f8_spec<32, 32, 16, false, false>(*fx.cu, fx.vbase + DST_OFF,
                                                            fx.vbase + S0_OFF, fx.vbase + S1_OFF, 0,
                                                            /*const_acc=*/0, 0, 0, 0);
  };
  bench("v_mfma_f32_32x32x16_bf8_bf8 [specialized]", fx, run, double(M) * N * K * B,
        /*is_int=*/false);
}

// v_mfma_i32_16x16x32_i8: integer path, exact SIMD-vs-scalar agreement.
TEST(MfmaSimdBenchmark, I32_16x16x32_i8) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 16, N = 16, K = 32, B = 1;
  fx.seed_i8(S0_OFF, 8, mma_test::SmallI8Gen(7));
  fx.seed_i8(S1_OFF, 8, mma_test::SmallI8Gen(8));
  auto run = [&] {
    amdgpu::exec_i32_i8(*fx.cu, M, N, K, B, fx.vbase + DST_OFF, fx.vbase + S0_OFF,
                        fx.vbase + S1_OFF, 0, /*const_acc=*/0);
  };
  bench("v_mfma_i32_16x16x32_i8", fx, run, double(M) * N * K * B, /*is_int=*/true);
}

// v_mfma_i32_32x32x16_i8: integer, N=32.
TEST(MfmaSimdBenchmark, I32_32x32x16_i8) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 32, N = 32, K = 16, B = 1;
  fx.seed_i8(S0_OFF, 8, mma_test::SmallI8Gen(9));
  fx.seed_i8(S1_OFF, 8, mma_test::SmallI8Gen(10));
  auto run = [&] {
    amdgpu::exec_i32_i8(*fx.cu, M, N, K, B, fx.vbase + DST_OFF, fx.vbase + S0_OFF,
                        fx.vbase + S1_OFF, 0, /*const_acc=*/0);
  };
  bench("v_mfma_i32_32x32x16_i8", fx, run, double(M) * N * K * B, /*is_int=*/true);
}

// Dense i8 shapes: dedicated constexpr specializations (i8 bulk sign-extend
// convert instead of per-element extract_i8).
TEST(MfmaSimdBenchmark, I32_16x16x32_i8_Specialized) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 16, N = 16, K = 32, B = 1;
  fx.seed_i8(S0_OFF, 8, mma_test::SmallI8Gen(7));
  fx.seed_i8(S1_OFF, 8, mma_test::SmallI8Gen(8));
  auto run = [&] {
    amdgpu::exec_i32_mfma_i8_spec<16, 16, 32>(*fx.cu, fx.vbase + DST_OFF, fx.vbase + S0_OFF,
                                              fx.vbase + S1_OFF, 0, /*const_acc=*/0);
  };
  bench("v_mfma_i32_16x16x32_i8 [specialized]", fx, run, double(M) * N * K * B, /*is_int=*/true);
}

TEST(MfmaSimdBenchmark, I32_32x32x16_i8_Specialized) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 32, N = 32, K = 16, B = 1;
  fx.seed_i8(S0_OFF, 8, mma_test::SmallI8Gen(9));
  fx.seed_i8(S1_OFF, 8, mma_test::SmallI8Gen(10));
  auto run = [&] {
    amdgpu::exec_i32_mfma_i8_spec<32, 32, 16>(*fx.cu, fx.vbase + DST_OFF, fx.vbase + S0_OFF,
                                              fx.vbase + S1_OFF, 0, /*const_acc=*/0);
  };
  bench("v_mfma_i32_32x32x16_i8 [specialized]", fx, run, double(M) * N * K * B, /*is_int=*/true);
}

TEST(MfmaSimdBenchmark, I32_16x16x64_i8_Specialized) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 16, N = 16, K = 64, B = 1;
  fx.seed_i8(S0_OFF, 16, mma_test::SmallI8Gen(11));
  fx.seed_i8(S1_OFF, 16, mma_test::SmallI8Gen(12));
  auto run = [&] {
    amdgpu::exec_i32_mfma_i8_spec<16, 16, 64>(*fx.cu, fx.vbase + DST_OFF, fx.vbase + S0_OFF,
                                              fx.vbase + S1_OFF, 0, /*const_acc=*/0);
  };
  bench("v_mfma_i32_16x16x64_i8 [specialized]", fx, run, double(M) * N * K * B, /*is_int=*/true);
}

TEST(MfmaSimdBenchmark, I32_32x32x32_i8_Specialized) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 32, N = 32, K = 32, B = 1;
  fx.seed_i8(S0_OFF, 16, mma_test::SmallI8Gen(13));
  fx.seed_i8(S1_OFF, 16, mma_test::SmallI8Gen(14));
  auto run = [&] {
    amdgpu::exec_i32_mfma_i8_spec<32, 32, 32>(*fx.cu, fx.vbase + DST_OFF, fx.vbase + S0_OFF,
                                              fx.vbase + S1_OFF, 0, /*const_acc=*/0);
  };
  bench("v_mfma_i32_32x32x32_i8 [specialized]", fx, run, double(M) * N * K * B, /*is_int=*/true);
}

// v_mfma_scale_f32_16x16x128_f8f6f4: MX-scaled fp8 path (exec_f32_scaled),
// in_bits=8, K=128 (4 K-blocks), N=16. Scales seeded to e8m0 127 (factor 1).
TEST(MfmaSimdBenchmark, F32Scaled_16x16x128_fp8) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 16, N = 16, K = 128, B = 1, bits = 8;
  constexpr uint32_t SCALE_A = 192, SCALE_B = 200;
  fx.seed(S0_OFF, 8, bits, mma_test::SmallGen(13));
  fx.seed(S1_OFF, 8, bits, mma_test::SmallGen(14));
  for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
    fx.cu->write_vgpr(fx.vbase + SCALE_A, lane, 0x7F7F7F7Fu); // e8m0 = 127 per block
    fx.cu->write_vgpr(fx.vbase + SCALE_B, lane, 0x7F7F7F7Fu);
  }
  auto run = [&] {
    amdgpu::exec_f32_scaled(*fx.cu, M, N, K, B, bits, fx.vbase + DST_OFF, fx.vbase + S0_OFF,
                            fx.vbase + S1_OFF, 0, amdgpu::extract_fp8, amdgpu::extract_fp8,
                            /*const_acc=*/0, /*cbsz=*/0, /*abid=*/0, /*blgp=*/0, fx.vbase + SCALE_A,
                            fx.vbase + SCALE_B);
  };
  bench("v_mfma_scale_f32_16x16x128_fp8", fx, run, double(M) * N * K * B, /*is_int=*/false);
}

// v_mfma_f64_16x16x4_f64: f64 path (native<double> == 8 lanes, N=16).
TEST(MfmaSimdBenchmark, F64_16x16x4_f64) {
  SKIP_IF_NO_SIMD();
  if (util::native<double>::size() <= 1)
    GTEST_SKIP() << "no native f64 SIMD";
  BenchFixture fx;
  constexpr uint32_t M = 16, N = 16, K = 4, B = 1;
  fx.seed_f64(S0_OFF, 8, mma_test::SmallGen(11));
  fx.seed_f64(S1_OFF, 8, mma_test::SmallGen(12));
  auto run = [&] {
    amdgpu::exec_f64(*fx.cu, M, N, K, B, fx.vbase + DST_OFF, fx.vbase + S0_OFF, fx.vbase + S1_OFF,
                     0, /*const_acc=*/0);
  };
  bench("v_mfma_f64_16x16x4_f64", fx, run, double(M) * N * K * B, /*is_int=*/false);
}

// f32-input MFMA shapes (no F16C convert; specialization gain is the
// addressing-unroll + raw vgpr_data + no-heap, not a convert).
TEST(MfmaSimdBenchmark, F32_32x32x2_f32_Specialized) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 32, N = 32, K = 2, B = 1, bits = 32;
  fx.seed(S0_OFF, 16, bits, mma_test::SmallGen(31));
  fx.seed(S1_OFF, 16, bits, mma_test::SmallGen(32));
  auto run = [&] {
    amdgpu::exec_f32_mfma_f32_spec<32, 32, 2, 1>(*fx.cu, fx.vbase + DST_OFF, fx.vbase + S0_OFF,
                                                 fx.vbase + S1_OFF, 0, 0, 0, 0, 0);
  };
  bench("v_mfma_f32_32x32x2_f32 [specialized]", fx, run, double(M) * N * K * B, /*is_int=*/false);
}

TEST(MfmaSimdBenchmark, F32_16x16x4_f32_Specialized) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 16, N = 16, K = 4, B = 1, bits = 32;
  fx.seed(S0_OFF, 16, bits, mma_test::SmallGen(33));
  fx.seed(S1_OFF, 16, bits, mma_test::SmallGen(34));
  auto run = [&] {
    amdgpu::exec_f32_mfma_f32_spec<16, 16, 4, 1>(*fx.cu, fx.vbase + DST_OFF, fx.vbase + S0_OFF,
                                                 fx.vbase + S1_OFF, 0, 0, 0, 0, 0);
  };
  bench("v_mfma_f32_16x16x4_f32 [specialized]", fx, run, double(M) * N * K * B, /*is_int=*/false);
}

TEST(MfmaSimdBenchmark, F32_32x32x1x2_f32_Specialized) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 32, N = 32, K = 1, B = 2, bits = 32;
  fx.seed(S0_OFF, 16, bits, mma_test::SmallGen(35));
  fx.seed(S1_OFF, 16, bits, mma_test::SmallGen(36));
  auto run = [&] {
    amdgpu::exec_f32_mfma_f32_spec<32, 32, 1, 2>(*fx.cu, fx.vbase + DST_OFF, fx.vbase + S0_OFF,
                                                 fx.vbase + S1_OFF, 0, 0, 0, 0, 0);
  };
  bench("v_mfma_f32_32x32x1_f32 (B=2) [specialized]", fx, run, double(M) * N * K * B,
        /*is_int=*/false);
}

// Batched f16/bf16/i8 MFMA shapes, now also specialized (per-batch block
// matmul over the same constexpr-unrolled kernel).
TEST(MfmaSimdBenchmark, F32_32x32x4x2_f16_Specialized) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 32, N = 32, K = 4, B = 2, bits = 16;
  fx.seed(S0_OFF, 2, bits, mma_test::SmallGen(51));
  fx.seed(S1_OFF, 2, bits, mma_test::SmallGen(52));
  auto run = [&] {
    amdgpu::exec_f32_mfma_f16_spec<32, 32, 4, 2>(*fx.cu, fx.vbase + DST_OFF, fx.vbase + S0_OFF,
                                                 fx.vbase + S1_OFF, 0, 0, 0, 0, 0);
  };
  bench("v_mfma_f32_32x32x4_f16 (B=2) [specialized]", fx, run, double(M) * N * K * B,
        /*is_int=*/false);
}

TEST(MfmaSimdBenchmark, F32_16x16x4x4_f16_Specialized) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 16, N = 16, K = 4, B = 4, bits = 16;
  fx.seed(S0_OFF, 2, bits, mma_test::SmallGen(53));
  fx.seed(S1_OFF, 2, bits, mma_test::SmallGen(54));
  auto run = [&] {
    amdgpu::exec_f32_mfma_f16_spec<16, 16, 4, 4>(*fx.cu, fx.vbase + DST_OFF, fx.vbase + S0_OFF,
                                                 fx.vbase + S1_OFF, 0, 0, 0, 0, 0);
  };
  bench("v_mfma_f32_16x16x4_f16 (B=4) [specialized]", fx, run, double(M) * N * K * B,
        /*is_int=*/false);
}

TEST(MfmaSimdBenchmark, F32_32x32x4x2_bf16_Specialized) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 32, N = 32, K = 4, B = 2;
  fx.seed_bf16(S0_OFF, 2, mma_test::SmallGen(55));
  fx.seed_bf16(S1_OFF, 2, mma_test::SmallGen(56));
  auto run = [&] {
    amdgpu::exec_f32_mfma_bf16_spec<32, 32, 4, 2>(*fx.cu, fx.vbase + DST_OFF, fx.vbase + S0_OFF,
                                                  fx.vbase + S1_OFF, 0, 0, 0, 0, 0);
  };
  bench("v_mfma_f32_32x32x4_bf16 (B=2) [specialized]", fx, run, double(M) * N * K * B,
        /*is_int=*/false);
}

TEST(MfmaSimdBenchmark, F32_16x16x4x4_bf16_Specialized) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 16, N = 16, K = 4, B = 4;
  fx.seed_bf16(S0_OFF, 2, mma_test::SmallGen(57));
  fx.seed_bf16(S1_OFF, 2, mma_test::SmallGen(58));
  auto run = [&] {
    amdgpu::exec_f32_mfma_bf16_spec<16, 16, 4, 4>(*fx.cu, fx.vbase + DST_OFF, fx.vbase + S0_OFF,
                                                  fx.vbase + S1_OFF, 0, 0, 0, 0, 0);
  };
  bench("v_mfma_f32_16x16x4_bf16 (B=4) [specialized]", fx, run, double(M) * N * K * B,
        /*is_int=*/false);
}

TEST(MfmaSimdBenchmark, I32_32x32x4x2_i8_Specialized) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 32, N = 32, K = 4, B = 2;
  fx.seed_i8(S0_OFF, 1, mma_test::SmallI8Gen(15));
  fx.seed_i8(S1_OFF, 1, mma_test::SmallI8Gen(16));
  auto run = [&] {
    amdgpu::exec_i32_mfma_i8_spec<32, 32, 4, 2>(*fx.cu, fx.vbase + DST_OFF, fx.vbase + S0_OFF,
                                                fx.vbase + S1_OFF, 0, /*const_acc=*/0);
  };
  bench("v_mfma_i32_32x32x4_i8 (B=2) [specialized]", fx, run, double(M) * N * K * B,
        /*is_int=*/true);
}

TEST(MfmaSimdBenchmark, I32_16x16x4x4_i8_Specialized) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  constexpr uint32_t M = 16, N = 16, K = 4, B = 4;
  fx.seed_i8(S0_OFF, 1, mma_test::SmallI8Gen(17));
  fx.seed_i8(S1_OFF, 1, mma_test::SmallI8Gen(18));
  auto run = [&] {
    amdgpu::exec_i32_mfma_i8_spec<16, 16, 4, 4>(*fx.cu, fx.vbase + DST_OFF, fx.vbase + S0_OFF,
                                                fx.vbase + S1_OFF, 0, /*const_acc=*/0);
  };
  bench("v_mfma_i32_16x16x4_i8 (B=4) [specialized]", fx, run, double(M) * N * K * B,
        /*is_int=*/true);
}
