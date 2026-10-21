// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file smfmac_simd_benchmark.cpp
/// @brief Decoded-instruction A/B benchmark for the F32 sparse MFMA family.
///
/// Covers every F32 SMFMAC helper shape on CDNA3 and CDNA4, including mixed
/// FP8/BF8 inputs. Each opcode is decoded once. Forced-scalar and default
/// execution must produce identical output bits before timings are reported.
/// The default mode selects AVX-512 when the fast path is present. The D
/// register window is restored outside each timed interval so every call
/// starts from the same accumulator state.

#include "decode_test_util.h"
#include "mma_test_util.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
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
#include <vector>

namespace {

using namespace rocjitsu;
using Clock = std::chrono::steady_clock;

constexpr uint32_t WF_SIZE = mma_test::MFMA_WF_SIZE;
constexpr uint32_t A_OFF = 0;
constexpr uint32_t B_OFF = 32;
constexpr uint32_t INDEX_OFF = 64;
constexpr uint32_t D_OFF = 96;
constexpr int ITERATIONS = mma_test::BENCH_ITERATIONS / 20;

enum class Format { F16, BF16, FP8, BF8 };

struct Shape {
  rj_code_arch_t arch;
  uint32_t m, n, k;
  uint32_t a_regs, b_regs, d_regs;
  bool fp8;
  // For 16-bit inputs: F16, BF16. For 8-bit inputs: BF8/BF8,
  // BF8/FP8, FP8/BF8, FP8/FP8.
  std::array<uint16_t, 4> opcodes;
};

constexpr std::array SHAPES{
    Shape{ROCJITSU_CODE_ARCH_CDNA3,
          16,
          16,
          32,
          2,
          4,
          4,
          false,
          {cdna3::kVSmfmacF3216x16x32F16Vop3pMfma, cdna3::kVSmfmacF3216x16x32Bf16Vop3pMfma, 0, 0}},
    Shape{ROCJITSU_CODE_ARCH_CDNA3,
          32,
          32,
          16,
          2,
          4,
          16,
          false,
          {cdna3::kVSmfmacF3232x32x16F16Vop3pMfma, cdna3::kVSmfmacF3232x32x16Bf16Vop3pMfma, 0, 0}},
    Shape{ROCJITSU_CODE_ARCH_CDNA4,
          16,
          16,
          64,
          4,
          8,
          4,
          false,
          {cdna4::kVSmfmacF3216x16x64F16Vop3pMfma, cdna4::kVSmfmacF3216x16x64Bf16Vop3pMfma, 0, 0}},
    Shape{ROCJITSU_CODE_ARCH_CDNA4,
          32,
          32,
          32,
          4,
          8,
          16,
          false,
          {cdna4::kVSmfmacF3232x32x32F16Vop3pMfma, cdna4::kVSmfmacF3232x32x32Bf16Vop3pMfma, 0, 0}},
    Shape{ROCJITSU_CODE_ARCH_CDNA3,
          16,
          16,
          64,
          2,
          4,
          4,
          true,
          {cdna3::kVSmfmacF3216x16x64Bf8Bf8Vop3pMfma, cdna3::kVSmfmacF3216x16x64Bf8Fp8Vop3pMfma,
           cdna3::kVSmfmacF3216x16x64Fp8Bf8Vop3pMfma, cdna3::kVSmfmacF3216x16x64Fp8Fp8Vop3pMfma}},
    Shape{ROCJITSU_CODE_ARCH_CDNA3,
          32,
          32,
          32,
          2,
          4,
          16,
          true,
          {cdna3::kVSmfmacF3232x32x32Bf8Bf8Vop3pMfma, cdna3::kVSmfmacF3232x32x32Bf8Fp8Vop3pMfma,
           cdna3::kVSmfmacF3232x32x32Fp8Bf8Vop3pMfma, cdna3::kVSmfmacF3232x32x32Fp8Fp8Vop3pMfma}},
    Shape{ROCJITSU_CODE_ARCH_CDNA4,
          16,
          16,
          128,
          4,
          8,
          4,
          true,
          {cdna4::kVSmfmacF3216x16x128Bf8Bf8Vop3pMfma, cdna4::kVSmfmacF3216x16x128Bf8Fp8Vop3pMfma,
           cdna4::kVSmfmacF3216x16x128Fp8Bf8Vop3pMfma, cdna4::kVSmfmacF3216x16x128Fp8Fp8Vop3pMfma}},
    Shape{ROCJITSU_CODE_ARCH_CDNA4,
          32,
          32,
          64,
          4,
          8,
          16,
          true,
          {cdna4::kVSmfmacF3232x32x64Bf8Bf8Vop3pMfma, cdna4::kVSmfmacF3232x32x64Bf8Fp8Vop3pMfma,
           cdna4::kVSmfmacF3232x32x64Fp8Bf8Vop3pMfma, cdna4::kVSmfmacF3232x32x64Fp8Fp8Vop3pMfma}},
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
  amdgpu::GpuMemory gpu_mem{"smfmac_simd_bench_mem"};
  amdgpu::L2Cache l2{"smfmac_simd_bench_l2"};
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  amdgpu::Wavefront *wf = nullptr;
  uint32_t vbase = 0;

  explicit BenchFixture(rj_code_arch_t arch) {
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = arch;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = mma_test::SGPRS_PER_WF;
    cfg.vgprs_per_wf = mma_test::VGPRS_PER_WF;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("cu_smfmac_simd_bench", cfg, &gpu_mem, &l2);
    if (cu) {
      wf = cu->dispatch_wf(0, 0, cfg.sgprs_per_wf, cfg.vgprs_per_wf);
      if (wf)
        vbase = wf->vgpr_alloc().base;
    }
  }

  void seed(uint32_t off, uint32_t regs, Format fmt, uint32_t seed, rj_code_arch_t arch) {
    mma_test::SmallGen gen(seed);
    for (uint32_t reg = 0; reg < regs; ++reg) {
      for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
        uint32_t word = 0;
        if (fmt == Format::F16 || fmt == Format::BF16) {
          const auto pack = [&](float value) {
            return fmt == Format::F16 ? util::f32_to_f16(value) : util::f32_to_bf16(value);
          };
          word = pack(gen()) | (static_cast<uint32_t>(pack(gen())) << 16);
        } else {
          const auto pack = [&](float value) {
            if (fmt == Format::FP8)
              return arch == ROCJITSU_CODE_ARCH_CDNA3 ? util::f32_to_fp8_e4m3_fnuz_rne(value)
                                                      : util::f32_to_fp8_e4m3_rne(value);
            return arch == ROCJITSU_CODE_ARCH_CDNA3 ? util::f32_to_bf8_e5m2_fnuz_rne(value)
                                                    : util::f32_to_bf8_e5m2_rne(value);
          };
          for (uint32_t byte = 0; byte < 4; ++byte)
            word |= static_cast<uint32_t>(pack(gen())) << (8 * byte);
        }
        cu->write_vgpr(vbase + off + reg, lane, word);
      }
    }
  }

  void seed_accumulator(uint32_t regs, uint32_t seed) {
    mma_test::SmallGen gen(seed);
    for (uint32_t reg = 0; reg < regs; ++reg)
      for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
        cu->write_vgpr(vbase + D_OFF + reg, lane, std::bit_cast<uint32_t>(gen()));
  }

  void seed_index() {
    // All six legal 2-of-4 selector pairs are used across lanes and nibbles.
    constexpr std::array<uint32_t, 6> pairs{0x4, 0x8, 0xC, 0x9, 0xD, 0xE};
    for (uint32_t lane = 0; lane < WF_SIZE; ++lane) {
      uint32_t word = 0;
      for (uint32_t nibble = 0; nibble < 8; ++nibble)
        word |= pairs[(lane + nibble) % pairs.size()] << (4 * nibble);
      cu->write_vgpr(vbase + INDEX_OFF, lane, word);
    }
  }

  std::vector<uint32_t> snapshot(uint32_t regs) const {
    std::vector<uint32_t> words(static_cast<size_t>(regs) * WF_SIZE);
    for (uint32_t reg = 0; reg < regs; ++reg)
      for (uint32_t lane = 0; lane < WF_SIZE; ++lane)
        words[static_cast<size_t>(reg) * WF_SIZE + lane] = cu->read_vgpr(vbase + D_OFF + reg, lane);
    return words;
  }

  void restore(const std::vector<uint32_t> &words) {
    for (size_t i = 0; i < words.size(); ++i)
      cu->write_vgpr(vbase + D_OFF + static_cast<uint32_t>(i / WF_SIZE),
                     static_cast<uint32_t>(i % WF_SIZE), words[i]);
  }
};

std::array<uint32_t, 2> build_words(const Shape &shape, uint16_t opcode) {
  constexpr uint32_t a = 256 + A_OFF, b = 256 + B_OFF, idx = 256 + INDEX_OFF;
  if (shape.arch == ROCJITSU_CODE_ARCH_CDNA3)
    return cdna3::build_vop3p_mfma(opcode,
                                   {.vdst = D_OFF, .acc_cd = 0, .src0 = a, .src1 = b, .src2 = idx});
  return cdna4::build_vop3p_mfma(opcode,
                                 {.vdst = D_OFF, .acc_cd = 0, .src0 = a, .src1 = b, .src2 = idx});
}

void benchmark_case(const Shape &shape, size_t variant) {
  const Format fmt_a = shape.fp8 ? (variant < 2 ? Format::BF8 : Format::FP8)
                                 : (variant == 0 ? Format::F16 : Format::BF16);
  const Format fmt_b = shape.fp8 ? (variant % 2 == 0 ? Format::BF8 : Format::FP8) : fmt_a;
  const char *fmt_name[] = {"f16", "bf16", "bf8_bf8", "bf8_fp8", "fp8_bf8", "fp8_fp8"};
  const char *suffix = shape.fp8 ? fmt_name[variant + 2] : fmt_name[variant];
  const std::string label = "v_smfmac_f32_" + std::to_string(shape.m) + "x" +
                            std::to_string(shape.n) + "x" + std::to_string(shape.k) + "_" + suffix;
  SCOPED_TRACE(label);

  BenchFixture fx(shape.arch);
  ASSERT_NE(fx.cu, nullptr);
  ASSERT_NE(fx.wf, nullptr);
  const uint32_t seed = 101 + static_cast<uint32_t>(variant) + 17 * shape.k;
  fx.seed(A_OFF, shape.a_regs, fmt_a, seed, shape.arch);
  fx.seed(B_OFF, shape.b_regs, fmt_b, seed + 1, shape.arch);
  fx.seed_index();
  fx.seed_accumulator(shape.d_regs, seed + 2);
  const auto initial = fx.snapshot(shape.d_regs);

  const auto words = build_words(shape, shape.opcodes[variant]);
  auto decoder = Decoder::create(shape.arch);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<Instruction> instruction(decode_valid(*decoder, words.data()));
  ASSERT_NE(instruction, nullptr);

  bool execution_failed = false;
  const auto run = [&] {
    execution_failed |= !fx.cu->execute_instruction(instruction.get(), *fx.wf).succeeded();
  };

  ForceScalarGuard force_scalar_guard;
  util::set_force_scalar_for_testing(true);
  run();
  ASSERT_FALSE(execution_failed) << label;
  const auto scalar = fx.snapshot(shape.d_regs);
  ASSERT_NE(scalar, initial) << label << ": instruction did not update D";

  fx.restore(initial);
  util::set_force_scalar_for_testing(false);
  run();
  ASSERT_FALSE(execution_failed) << label;
  const auto simd = fx.snapshot(shape.d_regs);
  ASSERT_EQ(scalar.size(), simd.size());
  for (size_t i = 0; i < scalar.size(); ++i)
    ASSERT_EQ(scalar[i], simd[i]) << label << ": raw F32 mismatch at word " << i;

  const auto time_block = [&](bool force_scalar) {
    util::set_force_scalar_for_testing(force_scalar);
    for (int i = 0; i < 8; ++i) {
      fx.restore(initial);
      run();
    }
    std::chrono::nanoseconds elapsed{};
    for (int i = 0; i < ITERATIONS / 2; ++i) {
      fx.restore(initial);
      const auto start = Clock::now();
      run();
      elapsed += std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start);
    }
    return elapsed.count();
  };

  // ABBA ordering balances changes in clock rate and thermal state.
  const auto scalar_first = time_block(true);
  const auto simd_first = time_block(false);
  const auto simd_second = time_block(false);
  const auto scalar_second = time_block(true);
  util::set_force_scalar_for_testing(false);

  const double scalar_ns = static_cast<double>(scalar_first + scalar_second) / ITERATIONS;
  const double simd_ns = static_cast<double>(simd_first + simd_second) / ITERATIONS;
  const double macs = static_cast<double>(shape.m) * shape.n * shape.k / 2;
  std::printf("\n  === decoded %s (%s, wave64, fresh D) ===\n"
              "  MACs/op: %.0f   scalar: %9.1f ns   default: %9.1f ns   speedup: %5.2fx\n",
              label.c_str(), shape.arch == ROCJITSU_CODE_ARCH_CDNA3 ? "gfx942" : "gfx950", macs,
              scalar_ns, simd_ns, simd_ns > 0.0 ? scalar_ns / simd_ns : 0.0);
  EXPECT_GT(scalar_ns, 0.0);
  EXPECT_GT(simd_ns, 0.0);
  EXPECT_FALSE(execution_failed) << label;
}

} // namespace

TEST(SmfmacSimdBenchmark, DecodedF32Forms) {
  SKIP_IF_NO_SIMD();
  if (util::native<float>::size() != 16)
    GTEST_SKIP() << "SMFMAC AVX-512 paths require 16-lane native SIMD";

  for (const auto &shape : SHAPES) {
    const size_t variants = shape.fp8 ? 4 : 2;
    for (size_t variant = 0; variant < variants; ++variant) {
      benchmark_case(shape, variant);
      if (testing::Test::HasFatalFailure())
        return;
    }
  }
}
