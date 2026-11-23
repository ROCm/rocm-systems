// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Decoded matrix microbenchmarks include operand routing and VGPR access.
// Compare forced-scalar and automatic dispatch on the same register inputs.

#include "compute_unit_test_util.h"
#include "decode_test_util.h"
#include "mma_test_util.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/shared/mma_exec.h"
#include "rocjitsu/isa/arch/amdgpu/shared/simd_glue.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/isa_family.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/data_types.h"
#include "util/simd.h"
#include "util/simd_test_hooks.h"

#include <gtest/gtest.h>

#include <array>
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

constexpr uint32_t vgpr_src(uint32_t reg) { return 256u + reg; }

struct BenchFixture {
  amdgpu::GpuMemory gpu_mem;
  amdgpu::L2Cache l2;
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  amdgpu::Wavefront *wf = nullptr;
  uint32_t vbase = 0;

  BenchFixture() : gpu_mem("mfma_simd_bench_mem"), l2("mfma_simd_bench_l2") {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = rocjitsu::IsaFamily::Cdna4;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = SGPRS_PER_WF;
    cfg.vgprs_per_wf = VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    rocjitsu::test::initialize_compute_unit(cu, "cu_mfma_simd", cfg, &gpu_mem, &l2);
    if (!cu)
      return;
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
TEST(MfmaSimdBenchmark, F32Scaled_16x16x128_fp8) {
  SKIP_IF_NO_SIMD();
  BenchFixture fx;
  ASSERT_FALSE(::testing::Test::HasFatalFailure());
  constexpr uint32_t M = 16, N = 16, K = 128, B = 1;
  constexpr uint32_t SCALE_A = 192, SCALE_B = 200;
  fx.seed(S0_OFF, 8, /*bit_width=*/8, mma_test::SmallGen(13));
  fx.seed(S1_OFF, 8, /*bit_width=*/8, mma_test::SmallGen(14));
  for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
    fx.cu->write_vgpr(fx.vbase + SCALE_A, lane, 0x7F7F7F7Fu); // e8m0 = 127 per block
    fx.cu->write_vgpr(fx.vbase + SCALE_B, lane, 0x7F7F7F7Fu);
  }
  auto decoder = Decoder::create(rocjitsu::IsaFamily::Cdna4);
  ASSERT_NE(decoder, nullptr);
  const auto words = mma_test::make_cdna4_mfma_scale_words(
      45, /*abid=*/1, vgpr_src(SCALE_A), vgpr_src(SCALE_B), /*a_byte=*/0,
      /*b_byte=*/0, /*cbsz=*/0, /*blgp=*/0, DST_OFF, vgpr_src(S0_OFF), vgpr_src(S1_OFF),
      vgpr_src(32));
  std::unique_ptr<Instruction> inst(decode_valid(*decoder, words.data()));
  ASSERT_NE(inst, nullptr);
  ASSERT_EQ(std::string_view(inst->mnemonic()), "v_mfma_scale_f32_16x16x128_f8f6f4");

  auto run = [&] { (void)fx.cu->execute_instruction(inst.get(), *fx.wf); };
  bench("v_mfma_scale_f32_16x16x128_f8f6f4 [generated]", fx, run, double(M) * N * K * B,
        /*is_int=*/false);
}
