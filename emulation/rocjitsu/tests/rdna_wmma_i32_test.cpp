// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rdna_wmma_i32_test.cpp
/// @brief Decoded integer WMMA regressions from physical RDNA captures.
/// @details Saved local outputs constrain a best-effort simulator model. These
/// tests do not establish hardware internals or guarantee untested inputs,
/// flags, modes or devices. See fixtures/float_dot/README.md for provenance.

#include "fixtures/float_dot/rdna_wmma_iu4_cases.h"
#include "fixtures/float_dot/rdna_wmma_iu4_k32_cases.h"
#include "fixtures/float_dot/rdna_wmma_iu8_cases.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/simd_test_hooks.h"

#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace {
using namespace rocjitsu;

struct WmmaI32Shape {
  uint16_t opcode;
  std::string_view mnemonic;
  uint32_t input_bits;
  uint32_t k_size;
};

constexpr std::array<WmmaI32Shape, 3> kWmmaI32Shapes{{
    {rdna4::kVWmmaI3216x16x16Iu8Vop3p, "v_wmma_i32_16x16x16_iu8", 8, 16},
    {rdna4::kVWmmaI3216x16x16Iu4Vop3p, "v_wmma_i32_16x16x16_iu4", 4, 16},
    {rdna4::kVWmmaI3216x16x32Iu4Vop3p, "v_wmma_i32_16x16x32_iu4", 4, 32},
}};

struct WmmaI32Case {
  rj_code_arch_t arch;
  uint32_t width;
  bool scalar;
  WmmaI32Shape shape;
};

class RdnaWmmaI32 : public testing::TestWithParam<WmmaI32Case> {
protected:
  void SetUp() override {
    const auto [arch, width, scalar, shape] = GetParam();
    old_scalar = util::force_scalar();
    util::set_force_scalar_for_testing(scalar);
    cache.set_backing_memory(&memory);
    amdgpu::ComputeUnitCore::Config cfg{};
    cfg.arch = arch;
    cfg.num_wf_slots = 1;
    cfg.sgprs_per_wf = 106;
    cfg.vgprs_per_wf = 256;
    cfg.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("wmma_i32_test", cfg, &memory, &cache);
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
  void set(uint32_t reg, uint32_t lane, uint32_t value) { cu->write_vgpr(base + reg, lane, value); }
  uint32_t get(uint32_t reg, uint32_t lane) { return cu->read_vgpr(base + reg, lane); }
  uint32_t input_bits() const { return GetParam().shape.input_bits; }
  uint32_t k_size() const { return GetParam().shape.k_size; }
  void execute(uint32_t neg, bool clamp, uint32_t src2 = 272, uint32_t dst = 24) {
    const auto words = GetParam().arch == ROCJITSU_CODE_ARCH_RDNA4
                           ? rdna4::build_vop3p(GetParam().shape.opcode, {.vdst = uint8_t(dst),
                                                                          .opsel_hi_2 = 1,
                                                                          .clamp = uint8_t(clamp),
                                                                          .src0 = 256,
                                                                          .src1 = 264,
                                                                          .src2 = uint16_t(src2),
                                                                          .opsel_hi = 3,
                                                                          .neg = uint8_t(neg)})
                           : rdna3::build_vop3p(GetParam().shape.opcode, {.vdst = uint8_t(dst),
                                                                          .op_sel_hi_2 = 1,
                                                                          .clamp = uint8_t(clamp),
                                                                          .src0 = 256,
                                                                          .src1 = 264,
                                                                          .src2 = uint16_t(src2),
                                                                          .op_sel_hi = 3,
                                                                          .neg = uint8_t(neg)});
    const std::array<uint32_t, 4> code{words[0], words[1], 0, 0};
    auto decoded = decoder->decode(code.data());
    ASSERT_TRUE(decoded.succeeded());
    auto inst = std::move(decoded).value();
    ASSERT_EQ(inst->mnemonic(), GetParam().shape.mnemonic);
    ASSERT_TRUE(cu->execute_instruction(inst.get(), *wave).succeeded());
  }
  void fill(uint32_t a, uint32_t b, uint32_t c) {
    for (uint32_t reg = 0; reg < 33; ++reg)
      for (uint32_t lane = 0; lane < wave->wf_size(); ++lane)
        set(reg, lane, reg < 8 ? a : reg < 16 ? b : reg < 24 ? c : 0x13579bdfu);
  }
  void expect_constant(uint32_t dst, uint32_t expected) {
    for (uint32_t reg = 0; reg < 256 / wave->wf_size(); ++reg)
      for (uint32_t lane = 0; lane < wave->wf_size(); ++lane)
        EXPECT_EQ(get(dst + reg, lane), expected) << "reg=" << reg << " lane=" << lane;
  }
  amdgpu::GpuMemory memory{"wmma_memory"};
  amdgpu::L2Cache cache{"wmma_cache"};
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  std::unique_ptr<Decoder> decoder;
  amdgpu::Wavefront *wave = nullptr;
  uint32_t base = 0;
  bool old_scalar = false;
};

TEST_P(RdnaWmmaI32, HardwareSaturation) {
  const auto [arch, width, scalar, shape] = GetParam();
  const bool gfx12 = arch == ROCJITSU_CODE_ARCH_RDNA4;
  uint32_t rng = 0x12056101u;
  constexpr std::array<uint32_t, 8> c{0,           1,           0xffffffffu, 0x7fffffffu,
                                      0x80000000u, 0x7ffff000u, 0x80001000u, 0x12345678u};
  for (uint32_t reg = 0; reg < 33; ++reg)
    for (uint32_t lane = 0; lane < width; ++lane) {
      rng ^= rng << 13;
      rng ^= rng >> 17;
      rng ^= rng << 5;
      set(reg, lane, reg >= 16 && reg < 32 ? c[(rng >> 24) & 7] : rng);
    }
  if (!gfx12)
    for (uint32_t reg = 0; reg < 16; ++reg)
      for (uint32_t lane = 16; lane < width; ++lane)
        set(reg, lane, get(reg, lane % 16));
  ASSERT_NO_FATAL_FAILURE(execute(3, true));
  const auto &expected =
      k_size() == 32 ? kRdnaWmmaIu4K32Saturated[width == 64]
                     : (input_bits() == 8 ? kRdnaWmmaIu8Saturated
                                          : kRdnaWmmaIu4Saturated)[2 * gfx12 + (width == 64)];
  for (uint32_t reg = 0; reg < 256 / width; ++reg)
    for (uint32_t lane = 0; lane < width; ++lane)
      EXPECT_EQ(get(24 + reg, lane), expected[reg * width + lane])
          << "reg=" << reg << " lane=" << lane;
}

TEST_P(RdnaWmmaI32, SignednessAndOverlap) {
  // Physical captures cover both cards for K16 and gfx1201 for K32, with
  // all interpretations of raw A=255/15, B=128/8, clamp settings and overlaps.
  const bool iu8 = input_bits() == 8;
  const std::array<int32_t, 4> expected =
      k_size() == 32 ? std::array<int32_t, 4>{3847, -249, -3833, 263}
      : iu8          ? std::array<int32_t, 4>{522247, -2041, -522233, 2055}
                     : std::array<int32_t, 4>{1927, -121, -1913, 135};
  for (uint32_t neg = 0; neg < 4; ++neg)
    for (bool clamp : {false, true})
      for (uint32_t dst : {0u, 8u, 16u, 24u}) {
        fill(0xffffffffu, iu8 ? 0x80808080u : 0x88888888u, 7);
        ASSERT_NO_FATAL_FAILURE(execute(neg, clamp, 272, dst));
        expect_constant(dst, static_cast<uint32_t>(expected[neg]));
      }
}

TEST_P(RdnaWmmaI32, InlineMinusOneIsAnAccumulatorValue) {
  for (bool clamp : {false, true}) {
    fill(0xffffffffu, input_bits() == 8 ? 0x01010101u : 0x11111111u, 0x12345678u);
    ASSERT_NO_FATAL_FAILURE(execute(1, clamp, 193));
    expect_constant(24, k_size() == 32 ? 0xffffffdfu : 0xffffffefu); // -1 + K * (-1 * 1).
  }
}

TEST_P(RdnaWmmaI32, OverflowWrapsUnlessClamped) {
  // Both widths: unsigned products cross INT32_MAX on both cards for K16
  // and on gfx1201 for K32.
  for (bool clamp : {false, true}) {
    const bool iu8 = input_bits() == 8;
    fill(0xffffffffu, iu8 ? 0x80808080u : 0x88888888u, 0x7fffffffu);
    ASSERT_NO_FATAL_FAILURE(execute(0, clamp));
    expect_constant(24, clamp            ? 0x7fffffffu
                        : k_size() == 32 ? 0x80000effu
                        : iu8            ? 0x8007f7ffu
                                         : 0x8000077fu);
  }
}

std::vector<WmmaI32Case> rdna_wmma_i32_cases() {
  std::vector<WmmaI32Case> cases;
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5, ROCJITSU_CODE_ARCH_RDNA4})
    for (uint32_t width : {32u, 64u})
      for (bool scalar : {false, true})
        for (const auto &shape : kWmmaI32Shapes)
          if (arch == ROCJITSU_CODE_ARCH_RDNA4 || shape.k_size == 16)
            cases.push_back({arch, width, scalar, shape});
  return cases;
}

std::string rdna_wmma_i32_case_name(const testing::TestParamInfo<WmmaI32Case> &info) {
  const auto &[arch, width, scalar, shape] = info.param;
  const char *arch_name = arch == ROCJITSU_CODE_ARCH_RDNA4   ? "Rdna4"
                          : arch == ROCJITSU_CODE_ARCH_RDNA3 ? "Rdna3"
                                                             : "Rdna35";
  return "Iu" + std::to_string(shape.input_bits) + "K" + std::to_string(shape.k_size) + arch_name +
         "Wave" + std::to_string(width) + (scalar ? "Scalar" : "Default");
}

INSTANTIATE_TEST_SUITE_P(HardwareWidthsAndPaths, RdnaWmmaI32,
                         testing::ValuesIn(rdna_wmma_i32_cases()), rdna_wmma_i32_case_name);
} // namespace
