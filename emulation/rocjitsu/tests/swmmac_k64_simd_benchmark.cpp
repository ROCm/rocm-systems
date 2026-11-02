// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file swmmac_k64_simd_benchmark.cpp
/// @brief Decoded-instruction A/B benchmark for the five gfx1250 K=64
/// F16/BF16 SWMMAC forms targeted by the AVX-512 fast paths.

#include "decode_test_util.h"
#include "mma_test_util.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/data_types.h"
#include "util/simd.h"
#include "util/simd_test_hooks.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <vector>

namespace {

using namespace rocjitsu;
using Clock = std::chrono::steady_clock;

constexpr uint32_t WF_SIZE = mma_test::WMMA_WF_SIZE;
constexpr uint32_t A_OFF = 0;
constexpr uint32_t B_OFF = 32;
constexpr uint32_t ACC_OFF = 64;
constexpr uint32_t INDEX_OFF = 96;
constexpr double SPARSE_MACS = 16.0 * 16.0 * (64.0 / 2.0);

enum class Format { F16, BF16, F32 };

struct BenchCase {
  uint16_t opcode;
  const char *label;
  Format input;
  Format accumulator;
  Format output;
  uint32_t accumulator_regs;
  uint32_t output_regs;
  uint32_t seed;
};

constexpr std::array BENCH_CASES{
    BenchCase{cdna5::kVSwmmacF3216x16x64F16Vop3p, "v_swmmac_f32_16x16x64_f16", Format::F16,
              Format::F32, Format::F32, 8, 8, 101},
    BenchCase{cdna5::kVSwmmacF3216x16x64Bf16Vop3p, "v_swmmac_f32_16x16x64_bf16", Format::BF16,
              Format::F32, Format::F32, 8, 8, 111},
    BenchCase{cdna5::kVSwmmacF1616x16x64F16Vop3p, "v_swmmac_f16_16x16x64_f16", Format::F16,
              Format::F16, Format::F16, 4, 4, 121},
    BenchCase{cdna5::kVSwmmacBf1616x16x64Bf16Vop3p, "v_swmmac_bf16_16x16x64_bf16", Format::BF16,
              Format::BF16, Format::BF16, 4, 4, 131},
    // BF16F32 reads eight F32 C registers but writes four packed-BF16 D
    // registers. Keep those two widths distinct in both reset and comparison.
    BenchCase{cdna5::kVSwmmacBf16f3216x16x64Bf16Vop3p, "v_swmmac_bf16f32_16x16x64_bf16",
              Format::BF16, Format::F32, Format::BF16, 8, 4, 141},
};

class ForceScalarGuard {
public:
  ForceScalarGuard() : original_(util::force_scalar()) {}
  ~ForceScalarGuard() { util::set_force_scalar_for_testing(original_); }

  ForceScalarGuard(const ForceScalarGuard &) = delete;
  ForceScalarGuard &operator=(const ForceScalarGuard &) = delete;

private:
  bool original_;
};

struct BenchFixture {
  amdgpu::GpuMemory gpu_mem{"swmmac_k64_bench_mem"};
  amdgpu::L2Cache l2{"swmmac_k64_bench_l2"};
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  amdgpu::Wavefront *wf = nullptr;
  uint32_t vbase = 0;

  BenchFixture() {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = ROCJITSU_CODE_ARCH_CDNA5;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = mma_test::SGPRS_PER_WF;
    cfg.vgprs_per_wf = mma_test::VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_swmmac_k64_bench", cfg, &gpu_mem, &l2);
    wf = cu->dispatch_wf(0, 0, mma_test::SGPRS_PER_WF, mma_test::VGPRS_PER_WF);
    if (wf)
      vbase = wf->vgpr_alloc().base;
  }

  void seed(uint32_t off, uint32_t regs, Format fmt, uint32_t seed) {
    mma_test::SmallGen gen(seed);
    for (uint32_t reg = 0; reg < regs; ++reg) {
      for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
        uint32_t word = 0;
        if (fmt == Format::F32) {
          word = std::bit_cast<uint32_t>(gen());
        } else {
          const auto pack = [&](float value) {
            return fmt == Format::F16 ? util::f32_to_f16(value) : util::f32_to_bf16(value);
          };
          word = pack(gen()) | (static_cast<uint32_t>(pack(gen())) << 16);
        }
        cu->write_vgpr(vbase + off + reg, lane, word);
      }
    }
  }

  void fill(uint32_t off, uint32_t regs, uint32_t word) {
    for (uint32_t reg = 0; reg < regs; ++reg)
      for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
        cu->write_vgpr(vbase + off + reg, lane, word);
  }

  std::vector<uint32_t> snapshot(uint32_t off, uint32_t regs) const {
    std::vector<uint32_t> words(static_cast<size_t>(regs) * WF_SIZE);
    for (uint32_t reg = 0; reg < regs; ++reg)
      for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
        words[static_cast<size_t>(reg) * WF_SIZE + lane] = cu->read_vgpr(vbase + off + reg, lane);
    return words;
  }
};

double relative_error(float a, float b) {
  if (!std::isfinite(a) || !std::isfinite(b))
    return a == b ? 0.0 : INFINITY;
  return std::abs(static_cast<double>(a) - b) / (std::abs(a) + 1.0e-6);
}

void expect_close(const BenchCase &test, const std::vector<uint32_t> &scalar,
                  const std::vector<uint32_t> &simd) {
  ASSERT_EQ(scalar.size(), simd.size());
  double max_relative_error = 0.0;
  for (size_t i = 0; i < scalar.size(); ++i) {
    if (test.output == Format::F32) {
      max_relative_error =
          std::max(max_relative_error,
                   relative_error(std::bit_cast<float>(scalar[i]), std::bit_cast<float>(simd[i])));
      continue;
    }
    for (uint32_t half = 0; half < 2; ++half) {
      const uint16_t scalar_raw = static_cast<uint16_t>(scalar[i] >> (half * 16));
      const uint16_t simd_raw = static_cast<uint16_t>(simd[i] >> (half * 16));
      const float scalar_value =
          test.output == Format::F16 ? util::f16_to_f32(scalar_raw) : util::bf16_to_f32(scalar_raw);
      const float simd_value =
          test.output == Format::F16 ? util::f16_to_f32(simd_raw) : util::bf16_to_f32(simd_raw);
      max_relative_error = std::max(max_relative_error, relative_error(scalar_value, simd_value));
    }
  }
  EXPECT_LT(max_relative_error, 5.0e-3) << test.label;
}

void benchmark_case(const BenchCase &test) {
  BenchFixture fx;
  ASSERT_NE(fx.cu, nullptr);
  ASSERT_NE(fx.wf, nullptr);

  fx.seed(A_OFF, 8, test.input, test.seed);
  fx.seed(B_OFF, 16, test.input, test.seed + 1);
  fx.fill(INDEX_OFF, 1, 0xE4E4E4E4u); // alternating legal (0,1)/(2,3) 2:4 pairs

  const auto words = cdna5::build_vop3p(
      test.opcode,
      {.vdst = ACC_OFF, .src0 = 256 + A_OFF, .src1 = 256 + B_OFF, .src2 = 256 + INDEX_OFF});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> instruction(decode_valid(*decoder, words.data()));
  ASSERT_NE(instruction, nullptr);

  auto reset_accumulator = [&] {
    fx.seed(ACC_OFF, test.accumulator_regs, test.accumulator, test.seed + 2);
  };
  bool execution_failed = false;
  auto run = [&] {
    execution_failed |= !fx.cu->execute_instruction(instruction.get(), *fx.wf).succeeded();
  };

  ForceScalarGuard force_scalar_guard;
  reset_accumulator();
  const auto upper_before = fx.snapshot(ACC_OFF + 4, 4);
  util::set_force_scalar_for_testing(true);
  run();
  const auto scalar = fx.snapshot(ACC_OFF, test.output_regs);
  if (test.opcode == cdna5::kVSwmmacBf16f3216x16x64Bf16Vop3p) {
    EXPECT_EQ(fx.snapshot(ACC_OFF + 4, 4), upper_before);
  }

  reset_accumulator();
  util::set_force_scalar_for_testing(false);
  run();
  const auto simd = fx.snapshot(ACC_OFF, test.output_regs);
  if (test.opcode == cdna5::kVSwmmacBf16f3216x16x64Bf16Vop3p) {
    EXPECT_EQ(fx.snapshot(ACC_OFF + 4, 4), upper_before);
  }
  expect_close(test, scalar, simd);

  auto time_block = [&](bool force_scalar, int iterations) {
    util::set_force_scalar_for_testing(force_scalar);
    for (int i = 0; i < 25; ++i) {
      reset_accumulator();
      run();
    }
    std::chrono::nanoseconds elapsed{};
    for (int i = 0; i < iterations; ++i) {
      // Keep reset work outside the timed region and execute every instruction
      // from the same C state.  This avoids measuring a dependent D/C chain;
      // it is required for BF16F32, whose packed D cannot serve as its F32 C.
      reset_accumulator();
      const auto start = Clock::now();
      run();
      elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start);
    }
    return elapsed.count();
  };

  const int first_iterations = mma_test::BENCH_ITERATIONS / 2;
  const int second_iterations = mma_test::BENCH_ITERATIONS - first_iterations;
  const auto scalar_first = time_block(true, first_iterations);
  const auto simd_first = time_block(false, first_iterations);
  const auto simd_second = time_block(false, second_iterations);
  const auto scalar_second = time_block(true, second_iterations);
  util::set_force_scalar_for_testing(false);

  const double scalar_ns =
      static_cast<double>(scalar_first + scalar_second) / mma_test::BENCH_ITERATIONS;
  const double simd_ns = static_cast<double>(simd_first + simd_second) / mma_test::BENCH_ITERATIONS;
  std::printf("\n  === decoded %s (gfx1250, wave32, fresh C) ===\n"
              "  MACs/op: %.0f   scalar: %9.1f ns   simd: %9.1f ns   speedup: %5.2fx\n",
              test.label, SPARSE_MACS, scalar_ns, simd_ns,
              simd_ns > 0.0 ? scalar_ns / simd_ns : 0.0);
  EXPECT_GT(scalar_ns, 0.0);
  EXPECT_GT(simd_ns, 0.0);
  EXPECT_FALSE(execution_failed);
}

} // namespace

TEST(SwmmacK64SimdBenchmark, DecodedF16Bf16Forms) {
  SKIP_IF_NO_SIMD();
  if (util::native<float>::size() != 16)
    GTEST_SKIP() << "the K=64 SWMMAC fast paths require 16-lane native SIMD";

  for (const auto &test : BENCH_CASES) {
    SCOPED_TRACE(test.label);
    benchmark_case(test);
    if (testing::Test::HasFatalFailure())
      return;
  }
}
