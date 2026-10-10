// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rdna_wmma_float_test.cpp
/// @brief Raw decoded WMMA and SWMMAC regressions from physical RDNA captures.
/// @details Saved local outputs constrain a best-effort simulator model. These
/// tests do not establish hardware internals or guarantee untested inputs,
/// flags, modes or devices. See fixtures/float_dot/README.md for provenance.

#include "fixtures/float_dot/rdna_swmmac_bf16_bf16_cases.h"
#include "fixtures/float_dot/rdna_swmmac_f16_f16_cases.h"
#include "fixtures/float_dot/rdna_swmmac_f32_bf16_cases.h"
#include "fixtures/float_dot/rdna_swmmac_f32_bf8_bf8_cases.h"
#include "fixtures/float_dot/rdna_swmmac_f32_bf8_fp8_cases.h"
#include "fixtures/float_dot/rdna_swmmac_f32_f16_cases.h"
#include "fixtures/float_dot/rdna_swmmac_f32_fp8_bf8_cases.h"
#include "fixtures/float_dot/rdna_swmmac_f32_fp8_fp8_cases.h"
#include "fixtures/float_dot/rdna_wmma_bf16_bf16_cases.h"
#include "fixtures/float_dot/rdna_wmma_f16_f16_cases.h"
#include "fixtures/float_dot/rdna_wmma_f32_bf16_cases.h"
#include "fixtures/float_dot/rdna_wmma_f32_bf8_bf8_cases.h"
#include "fixtures/float_dot/rdna_wmma_f32_bf8_fp8_cases.h"
#include "fixtures/float_dot/rdna_wmma_f32_f16_cases.h"
#include "fixtures/float_dot/rdna_wmma_f32_fp8_bf8_cases.h"
#include "fixtures/float_dot/rdna_wmma_f32_fp8_fp8_cases.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/simd_test_hooks.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace rocjitsu;

struct WmmaFloatShape {
  uint16_t opcode;
  std::string_view mnemonic;
  uint32_t input_bits;
  bool a_bfloat;
  bool b_bfloat;
  bool packed;
  std::span<const RdnaWmmaFloatMatrixCase> matrices;
  std::span<const RdnaWmmaFloatOperandCase> operands;
};

static_assert(rdna3::kVWmmaF3216x16x16F16Vop3p == rdna4::kVWmmaF3216x16x16F16Vop3p);
static_assert(rdna3::kVWmmaF3216x16x16Bf16Vop3p == rdna4::kVWmmaF3216x16x16Bf16Vop3p);
static_assert(rdna3::kVWmmaF1616x16x16F16Vop3p == rdna4::kVWmmaF1616x16x16F16Vop3p);
static_assert(rdna3::kVWmmaBf1616x16x16Bf16Vop3p == rdna4::kVWmmaBf1616x16x16Bf16Vop3p);

constexpr std::array<WmmaFloatShape, 16> kWmmaFloatShapes{{
    {rdna4::kVWmmaF3216x16x16Fp8Fp8Vop3p, "v_wmma_f32_16x16x16_fp8_fp8", 8, false, false, false,
     kRdnaWmmaF32Fp8Fp8Matrices, kRdnaWmmaF32Fp8Fp8Operands},
    {rdna4::kVWmmaF3216x16x16Fp8Bf8Vop3p, "v_wmma_f32_16x16x16_fp8_bf8", 8, false, true, false,
     kRdnaWmmaF32Fp8Bf8Matrices, kRdnaWmmaF32Fp8Bf8Operands},
    {rdna4::kVWmmaF3216x16x16Bf8Fp8Vop3p, "v_wmma_f32_16x16x16_bf8_fp8", 8, true, false, false,
     kRdnaWmmaF32Bf8Fp8Matrices, kRdnaWmmaF32Bf8Fp8Operands},
    {rdna4::kVWmmaF3216x16x16Bf8Bf8Vop3p, "v_wmma_f32_16x16x16_bf8_bf8", 8, true, true, false,
     kRdnaWmmaF32Bf8Bf8Matrices, kRdnaWmmaF32Bf8Bf8Operands},
    {rdna4::kVWmmaF3216x16x16F16Vop3p, "v_wmma_f32_16x16x16_f16", 16, false, false, false,
     kRdnaWmmaF32F16Matrices, kRdnaWmmaF32F16Operands},
    {rdna4::kVWmmaF3216x16x16Bf16Vop3p, "v_wmma_f32_16x16x16_bf16", 16, true, true, false,
     kRdnaWmmaF32Bf16Matrices, kRdnaWmmaF32Bf16Operands},
    {rdna4::kVWmmaF1616x16x16F16Vop3p, "v_wmma_f16_16x16x16_f16", 16, false, false, true,
     kRdnaWmmaF16F16Matrices, kRdnaWmmaF16F16Operands},
    {rdna4::kVWmmaBf1616x16x16Bf16Vop3p, "v_wmma_bf16_16x16x16_bf16", 16, true, true, true,
     kRdnaWmmaBf16Bf16Matrices, kRdnaWmmaBf16Bf16Operands},
    {rdna4::kVSwmmacF3216x16x32F16Vop3p, "v_swmmac_f32_16x16x32_f16", 16, false, false, false,
     kRdnaSwmmacF32F16Matrices, kRdnaSwmmacF32F16Operands},
    {rdna4::kVSwmmacF3216x16x32Bf16Vop3p, "v_swmmac_f32_16x16x32_bf16", 16, true, true, false,
     kRdnaSwmmacF32Bf16Matrices, kRdnaSwmmacF32Bf16Operands},
    {rdna4::kVSwmmacF1616x16x32F16Vop3p, "v_swmmac_f16_16x16x32_f16", 16, false, false, true,
     kRdnaSwmmacF16F16Matrices, kRdnaSwmmacF16F16Operands},
    {rdna4::kVSwmmacBf1616x16x32Bf16Vop3p, "v_swmmac_bf16_16x16x32_bf16", 16, true, true, true,
     kRdnaSwmmacBf16Bf16Matrices, kRdnaSwmmacBf16Bf16Operands},
    {rdna4::kVSwmmacF3216x16x32Fp8Fp8Vop3p, "v_swmmac_f32_16x16x32_fp8_fp8", 8, false, false, false,
     kRdnaSwmmacF32Fp8Fp8Matrices, kRdnaSwmmacF32Fp8Fp8Operands},
    {rdna4::kVSwmmacF3216x16x32Fp8Bf8Vop3p, "v_swmmac_f32_16x16x32_fp8_bf8", 8, false, true, false,
     kRdnaSwmmacF32Fp8Bf8Matrices, kRdnaSwmmacF32Fp8Bf8Operands},
    {rdna4::kVSwmmacF3216x16x32Bf8Fp8Vop3p, "v_swmmac_f32_16x16x32_bf8_fp8", 8, true, false, false,
     kRdnaSwmmacF32Bf8Fp8Matrices, kRdnaSwmmacF32Bf8Fp8Operands},
    {rdna4::kVSwmmacF3216x16x32Bf8Bf8Vop3p, "v_swmmac_f32_16x16x32_bf8_bf8", 8, true, true, false,
     kRdnaSwmmacF32Bf8Bf8Matrices, kRdnaSwmmacF32Bf8Bf8Operands},
}};

struct WmmaFloatCase {
  rj_code_arch_t arch;
  uint32_t width;
  bool scalar;
  WmmaFloatShape shape;
};

class RdnaWmmaFloat : public testing::TestWithParam<WmmaFloatCase> {
protected:
  void SetUp() override {
    shape = GetParam().shape;
    gfx12 = GetParam().arch == ROCJITSU_CODE_ARCH_RDNA4;
    width = GetParam().width;
    scalar = GetParam().scalar;
    old_scalar = util::force_scalar();
    util::set_force_scalar_for_testing(scalar);
    cache.set_backing_memory(&memory);
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = GetParam().arch;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = 106;
    cfg.vgprs_per_wf = 256;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("wmma_float_test", cfg, &memory, &cache);
    decoder = Decoder::create(cfg.arch);
    wave = cu->dispatch_wf(0, 0, 106, 256, width);
    wave->set_exec(width == 64 ? ~uint64_t{0} : 0xffffffffu);
    base = wave->vgpr_alloc().base;
  }
  void TearDown() override {
    if (wave)
      wave->halt();
    util::set_force_scalar_for_testing(old_scalar);
  }
  void execute(const std::array<uint32_t, 2> &words, uint32_t mode) {
    ASSERT_EQ((words[0] >> 16) & 0x7fu, shape.opcode);
    wave->set_mode_raw(mode);
    for (uint32_t reg = 0; reg < 33; ++reg)
      for (uint32_t lane = 0; lane < width; ++lane)
        cu->write_vgpr(base + reg, lane, input[reg * width + lane]);
    const std::array<uint32_t, 4> code{words[0], words[1], 0, 0};
    auto decoded = decoder->decode(code.data());
    ASSERT_TRUE(decoded.succeeded());
    auto inst = std::move(decoded).value();
    ASSERT_EQ(inst->mnemonic(), shape.mnemonic);
    ASSERT_TRUE(cu->execute_instruction(inst.get(), *wave).succeeded());
  }
  void expect_registers() {
    for (uint32_t reg = 0; reg < 33; ++reg)
      for (uint32_t lane = 0; lane < width; ++lane)
        EXPECT_EQ(cu->read_vgpr(base + reg, lane), expected[reg * width + lane])
            << "reg=" << reg << " lane=" << lane;
  }
  void matrix_input(uint32_t rng, bool finite, bool bounded_bf16) {
    if (bounded_bf16) {
      ASSERT_TRUE(finite);
      ASSERT_EQ(shape.input_bits, 16u);
      ASSERT_TRUE(shape.a_bfloat && shape.b_bfloat);
    }
    // Full-range packed matrices keep F32-shaped C/D for overflow and layout.
    // Packed-C witnesses use F16 operand cases or bounded BF16 matrices.
    constexpr std::array<uint32_t, 8> accumulators{
        0, 1, 0x80000000u, 0x3f800001u, 0xbf800001u, 0x43800001u, 0xc3800001u, 0x80000001u};
    // BF16 C boundary values repeat across both packed halves.
    constexpr std::array<uint32_t, 8> packed_bf16_accumulators{
        0,           0x00010001u, 0x80008000u, 0x3f813f81u,
        0xbf81bf81u, 0x43814381u, 0xc381c381u, 0x80018001u};
    const auto &c_values = bounded_bf16 && shape.packed ? packed_bf16_accumulators : accumulators;
    constexpr std::array<uint32_t, 6> index_pairs{4, 8, 12, 9, 13, 14};
    const uint32_t bits = shape.input_bits;
    for (uint32_t reg = 0; reg < 33; ++reg)
      for (uint32_t lane = 0; lane < width; ++lane) {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        uint32_t value = rng;
        if (finite && reg < 16) {
          const bool bf = reg < 8 ? shape.a_bfloat : shape.b_bfloat;
          const uint32_t max_finite = bits == 16 ? (bf ? 0x7f7f : 0x7bff) : (bf ? 0x7b : 0x7e);
          value = 0;
          for (uint32_t shift = 0; shift < 32; shift += bits) {
            const uint32_t part = (rng >> shift) & ((1u << bits) - 1);
            const uint32_t sign = part & (1u << (bits - 1));
            const uint32_t magnitude = part & ((1u << (bits - 1)) - 1);
            // BF16 magnitudes 0x3c80..0x417f bound factor exponents to [-6, 3].
            const uint32_t finite_magnitude =
                bounded_bf16 ? 0x3c80u + magnitude % 0x500u : magnitude % (max_finite + 1);
            value |= (sign | finite_magnitude) << shift;
          }
        } else if (finite && reg >= 16 && reg < 32)
          value = c_values[(rng >> 24) & 7];
        if (reg == 32) {
          value = 0;
          for (uint32_t i = 0; i < 8; ++i)
            value |= index_pairs[((rng >> (4 * i)) & 15) % 6] << (4 * i);
        }
        input[reg * width + lane] = value;
      }
    if (!gfx12)
      for (uint32_t reg = 0; reg < 16; ++reg)
        for (uint32_t lane = 16; lane < width; ++lane)
          input[reg * width + lane] = input[reg * width + lane % 16];
  }
  WmmaFloatShape shape{};
  bool gfx12 = false;
  bool scalar = false;
  bool old_scalar = false;
  uint32_t width = 0;
  uint32_t base = 0;
  std::array<uint32_t, 33 * 64> input{}, expected{};
  amdgpu::GpuMemory memory{"wmma_float_memory"};
  amdgpu::L2Cache cache{"wmma_float_cache"};
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wave = nullptr;
};

TEST_P(RdnaWmmaFloat, CapturedMatrices) {
  uint32_t tested = 0;
  for (const auto &fixture : shape.matrices) {
    if (fixture.gfx12 != gfx12 || fixture.width != width)
      continue;
    SCOPED_TRACE(testing::Message() << "seed=" << fixture.seed << " finite=" << fixture.finite
                                    << " bounded_bf16=" << fixture.bounded_bf16
                                    << " word0=" << fixture.code[0] << " mode=" << fixture.mode);
    ASSERT_NO_FATAL_FAILURE(matrix_input(fixture.seed, fixture.finite, fixture.bounded_bf16));
    expected = input;
    const uint32_t dst = fixture.code[0] & 255;
    ASSERT_LE(fixture.expected_words, fixture.expected.size());
    ASSERT_LE(dst * width + fixture.expected_words, 33 * width);
    std::copy_n(fixture.expected.begin(), fixture.expected_words, expected.begin() + dst * width);
    ASSERT_NO_FATAL_FAILURE(execute(fixture.code, fixture.mode));
    expect_registers();
    ++tested;
  }
  EXPECT_NE(tested, 0u);
}

TEST_P(RdnaWmmaFloat, CapturedOperands) {
  uint32_t tested = 0;
  for (const auto &fixture : shape.operands) {
    if (fixture.gfx12 != gfx12 || fixture.width != width)
      continue;
    SCOPED_TRACE(testing::Message() << "word0=" << fixture.code[0] << " word1=" << fixture.code[1]);
    for (uint32_t reg = 0; reg < 33; ++reg)
      for (uint32_t lane = 0; lane < width; ++lane)
        input[reg * width + lane] = reg < 8    ? fixture.a
                                    : reg < 16 ? fixture.b
                                    : reg < 24 ? fixture.c
                                    : reg < 32 ? fixture.d
                                               : fixture.index;
    if (fixture.factor_base != RdnaWmmaFloatOperandCase::kUniformFactors) {
      ASSERT_LT(fixture.factor_base, 33u);
      const uint32_t start = fixture.factor_base * width;
      ASSERT_LE(start + 16 * fixture.factor_groups.size(), 33 * width);
      for (uint32_t group = 0; group < fixture.factor_groups.size(); ++group)
        std::fill_n(input.begin() + start + 16 * group, 16, fixture.factor_groups[group]);
    }
    expected = input;
    const uint32_t count = shape.packed && gfx12 ? 128 : 256;
    const uint32_t dst = fixture.code[0] & 255;
    ASSERT_LE(dst * width + count, 33 * width);
    std::fill_n(expected.begin() + dst * width, count, fixture.expected);
    ASSERT_NO_FATAL_FAILURE(execute(fixture.code, fixture.mode));
    expect_registers();
    ++tested;
  }
  EXPECT_NE(tested, 0u);
}

std::vector<WmmaFloatCase> rdna_wmma_float_cases() {
  std::vector<WmmaFloatCase> result;
  for (const auto &shape : kWmmaFloatShapes)
    for (rj_code_arch_t arch :
         {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5, ROCJITSU_CODE_ARCH_RDNA4})
      for (uint32_t width : {32u, 64u})
        for (bool scalar : {false, true})
          if (std::any_of(shape.matrices.begin(), shape.matrices.end(), [&](const auto &f) {
                return f.gfx12 == (arch == ROCJITSU_CODE_ARCH_RDNA4) && f.width == width;
              }))
            result.push_back({arch, width, scalar, shape});
  return result;
}

std::string rdna_wmma_float_case_name(const testing::TestParamInfo<WmmaFloatCase> &info) {
  const auto &[arch, width, scalar, shape] = info.param;
  const char *arch_name = arch == ROCJITSU_CODE_ARCH_RDNA4     ? "Rdna4"
                          : arch == ROCJITSU_CODE_ARCH_RDNA3_5 ? "Rdna35"
                                                               : "Rdna3";
  return std::string(shape.mnemonic) + arch_name + "Wave" + std::to_string(width) +
         (scalar ? "Scalar" : "Default");
}

INSTANTIATE_TEST_SUITE_P(HardwareWidthsAndPaths, RdnaWmmaFloat,
                         testing::ValuesIn(rdna_wmma_float_cases()), rdna_wmma_float_case_name);
} // namespace
