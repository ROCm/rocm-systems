// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file swmmac_k128_simd_benchmark.cpp
/// @brief Decoded-instruction A/B benchmark for the eight gfx1250 K=128
/// FP8/BF8 SWMMAC forms.
///
/// The forced-scalar and default paths use the same decoded instruction and
/// must agree bit-for-bit before timing. The default path uses AVX-512 when
/// the fast path is present. Each timed call starts with the same C state;
/// restoring it is outside the timed interval.

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

#include <array>
#include <bit>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace rocjitsu;
using Clock = std::chrono::steady_clock;

constexpr uint32_t WF_SIZE = mma_test::WMMA_WF_SIZE;
constexpr uint32_t A_OFF = 0;
constexpr uint32_t B_OFF = 32;
constexpr uint32_t ACC_OFF = 64;
constexpr uint32_t INDEX_OFF = 96;
constexpr uint32_t A_REGS = 8;
constexpr uint32_t B_REGS = 16;
constexpr uint32_t INDEX_REGS = 2;
constexpr int ITERATIONS = mma_test::BENCH_ITERATIONS / 10;
constexpr double SPARSE_MACS = 16.0 * 16.0 * (128.0 / 2.0);

enum class Format { FP8, BF8, F32, F16 };

struct BenchCase {
  uint16_t opcode;
  const char *name;
  const char *label;
  Format a_format;
  Format b_format;
  Format accumulator_format;
  uint32_t accumulator_regs;
  uint32_t seed;
};

constexpr std::array BENCH_CASES{
    BenchCase{cdna5::kVSwmmacF3216x16x128Fp8Fp8Vop3p, "F32Fp8Fp8", "v_swmmac_f32_16x16x128_fp8_fp8",
              Format::FP8, Format::FP8, Format::F32, 8, 101},
    BenchCase{cdna5::kVSwmmacF3216x16x128Fp8Bf8Vop3p, "F32Fp8Bf8", "v_swmmac_f32_16x16x128_fp8_bf8",
              Format::FP8, Format::BF8, Format::F32, 8, 111},
    BenchCase{cdna5::kVSwmmacF3216x16x128Bf8Fp8Vop3p, "F32Bf8Fp8", "v_swmmac_f32_16x16x128_bf8_fp8",
              Format::BF8, Format::FP8, Format::F32, 8, 121},
    BenchCase{cdna5::kVSwmmacF3216x16x128Bf8Bf8Vop3p, "F32Bf8Bf8", "v_swmmac_f32_16x16x128_bf8_bf8",
              Format::BF8, Format::BF8, Format::F32, 8, 131},
    BenchCase{cdna5::kVSwmmacF1616x16x128Fp8Fp8Vop3p, "F16Fp8Fp8", "v_swmmac_f16_16x16x128_fp8_fp8",
              Format::FP8, Format::FP8, Format::F16, 4, 141},
    BenchCase{cdna5::kVSwmmacF1616x16x128Fp8Bf8Vop3p, "F16Fp8Bf8", "v_swmmac_f16_16x16x128_fp8_bf8",
              Format::FP8, Format::BF8, Format::F16, 4, 151},
    BenchCase{cdna5::kVSwmmacF1616x16x128Bf8Fp8Vop3p, "F16Bf8Fp8", "v_swmmac_f16_16x16x128_bf8_fp8",
              Format::BF8, Format::FP8, Format::F16, 4, 161},
    BenchCase{cdna5::kVSwmmacF1616x16x128Bf8Bf8Vop3p, "F16Bf8Bf8", "v_swmmac_f16_16x16x128_bf8_bf8",
              Format::BF8, Format::BF8, Format::F16, 4, 171},
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
  amdgpu::GpuMemory gpu_mem{"swmmac_k128_bench_mem"};
  amdgpu::L2Cache l2{"swmmac_k128_bench_l2"};
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
    cu = amdgpu::ComputeUnitCore::create("cu_swmmac_k128_bench", cfg, &gpu_mem, &l2);
    if (cu) {
      wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
      if (wf)
        vbase = wf->vgpr_alloc().base;
    }
  }

  void seed(uint32_t off, uint32_t regs, Format fmt, uint32_t seed) {
    mma_test::SmallGen gen(seed);
    for (uint32_t reg = 0; reg < regs; ++reg) {
      for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
        uint32_t word = 0;
        switch (fmt) {
        case Format::F32:
          word = std::bit_cast<uint32_t>(gen());
          break;
        case Format::F16:
          word = util::f32_to_f16(gen());
          word |= static_cast<uint32_t>(util::f32_to_f16(gen())) << 16;
          break;
        case Format::FP8:
        case Format::BF8:
          for (uint32_t byte = 0; byte < 4; ++byte) {
            const uint8_t packed = fmt == Format::FP8 ? util::f32_to_fp8_e4m3_rne(gen())
                                                      : util::f32_to_bf8_e5m2_rne(gen());
            word |= static_cast<uint32_t>(packed) << (8 * byte);
          }
          break;
        }
        cu->write_vgpr(vbase + off + reg, lane, word);
      }
    }
  }

  void seed_indices() {
    // The six legal index0 < index1 pairs, varied across both K=128 words.
    constexpr std::array<uint32_t, 6> pairs{0x4u, 0x8u, 0xCu, 0x9u, 0xDu, 0xEu};
    for (uint32_t reg = 0; reg < INDEX_REGS; ++reg) {
      for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
        uint32_t word = 0;
        for (uint32_t nibble = 0; nibble < 8; ++nibble)
          word |= pairs[(lane + nibble + 3 * reg) % pairs.size()] << (4 * nibble);
        cu->write_vgpr(vbase + INDEX_OFF + reg, lane, word);
      }
    }
  }

  std::vector<uint32_t> snapshot(uint32_t off, uint32_t regs) const {
    std::vector<uint32_t> words(static_cast<size_t>(regs) * WF_SIZE);
    for (uint32_t reg = 0; reg < regs; ++reg)
      for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
        words[static_cast<size_t>(reg) * WF_SIZE + lane] = cu->read_vgpr(vbase + off + reg, lane);
    return words;
  }

  void restore(uint32_t off, const std::vector<uint32_t> &words) {
    for (size_t i = 0; i < words.size(); ++i)
      cu->write_vgpr(vbase + off + static_cast<uint32_t>(i / WF_SIZE),
                     static_cast<uint32_t>(i % WF_SIZE), words[i]);
  }
};

void benchmark_case(const BenchCase &test) {
  BenchFixture fx;
  ASSERT_NE(fx.cu, nullptr);
  ASSERT_NE(fx.wf, nullptr);

  fx.seed(A_OFF, A_REGS, test.a_format, test.seed);
  fx.seed(B_OFF, B_REGS, test.b_format, test.seed + 1);
  fx.seed_indices();
  fx.seed(ACC_OFF, test.accumulator_regs, test.accumulator_format, test.seed + 2);
  const auto initial = fx.snapshot(ACC_OFF, test.accumulator_regs);

  const auto words = cdna5::build_vop3p(
      test.opcode,
      {.vdst = ACC_OFF, .src0 = 256 + A_OFF, .src1 = 256 + B_OFF, .src2 = 256 + INDEX_OFF});
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> instruction(decode_valid(*decoder, words.data()));
  ASSERT_NE(instruction, nullptr);
  EXPECT_EQ(instruction->mnemonic(), std::string_view(test.label));

  bool execution_failed = false;
  const auto run = [&] {
    execution_failed |= !fx.cu->execute_instruction(instruction.get(), *fx.wf).succeeded();
  };
  const auto reset_accumulator = [&] { fx.restore(ACC_OFF, initial); };

  ForceScalarGuard force_scalar_guard;
  util::set_force_scalar_for_testing(true);
  run();
  ASSERT_FALSE(execution_failed) << test.label;
  const auto scalar = fx.snapshot(ACC_OFF, test.accumulator_regs);
  ASSERT_NE(scalar, initial) << test.label << ": instruction did not update D";

  reset_accumulator();
  util::set_force_scalar_for_testing(false);
  run();
  ASSERT_FALSE(execution_failed) << test.label;
  const auto simd = fx.snapshot(ACC_OFF, test.accumulator_regs);
  ASSERT_EQ(scalar.size(), simd.size());
  for (size_t i = 0; i < scalar.size(); ++i)
    ASSERT_EQ(scalar[i], simd[i]) << test.label << ": raw output mismatch at word " << i;

  const auto time_block = [&](bool force_scalar) {
    util::set_force_scalar_for_testing(force_scalar);
    for (int i = 0; i < 12; ++i) {
      reset_accumulator();
      run();
    }
    std::chrono::nanoseconds elapsed{};
    for (int i = 0; i < ITERATIONS / 2; ++i) {
      reset_accumulator();
      const auto start = Clock::now();
      run();
      elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start);
    }
    return elapsed.count();
  };

  // ABBA block order balances clock and thermal drift between the modes.
  const auto scalar_first = time_block(true);
  const auto default_first = time_block(false);
  const auto default_second = time_block(false);
  const auto scalar_second = time_block(true);
  util::set_force_scalar_for_testing(false);

  const double scalar_ns = static_cast<double>(scalar_first + scalar_second) / ITERATIONS;
  const double default_ns = static_cast<double>(default_first + default_second) / ITERATIONS;
  std::printf("\n  === decoded %s (gfx1250, wave32, fresh C) ===\n"
              "  iterations/mode: %d   MACs/op: %.0f\n"
              "  scalar: %9.1f ns   default: %9.1f ns   speedup: %5.2fx\n",
              test.label, ITERATIONS, SPARSE_MACS, scalar_ns, default_ns,
              default_ns > 0.0 ? scalar_ns / default_ns : 0.0);
  EXPECT_GT(scalar_ns, 0.0);
  EXPECT_GT(default_ns, 0.0);
  EXPECT_FALSE(execution_failed) << test.label;
}

class SwmmacK128SimdBenchmark : public testing::TestWithParam<size_t> {};

TEST_P(SwmmacK128SimdBenchmark, DecodedInstruction) {
  SKIP_IF_NO_SIMD();
  if (util::native<float>::size() != 16)
    GTEST_SKIP() << "the K=128 SWMMAC fast paths require 16-lane native SIMD";
  benchmark_case(BENCH_CASES[GetParam()]);
}

INSTANTIATE_TEST_SUITE_P(Forms, SwmmacK128SimdBenchmark,
                         testing::Range<size_t>(0, BENCH_CASES.size()),
                         [](const testing::TestParamInfo<size_t> &info) {
                           return std::string(BENCH_CASES[info.param].name);
                         });

} // namespace
