// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file legacy_minmax_test.cpp
/// @brief Decoded legacy selection, MODE and architecture-specific NaN/zero rules.

#include "decode_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/shared/fp_mode.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/data_types.h"
#include "util/simd_test_hooks.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstdint>
#include <memory>
#include <tuple>

namespace rocjitsu {
namespace {

class LegacyMinMaxInstruction : public testing::TestWithParam<std::tuple<rj_code_arch_t, bool>> {
protected:
  void SetUp() override {
    saved_scalar_ = util::force_scalar();
    util::set_force_scalar_for_testing(std::get<1>(GetParam()));
    amdgpu::ComputeUnitCore::Config config{};
    config.arch = std::get<0>(GetParam());
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 106;
    config.vgprs_per_wf = 256;
    config.lds_size_kb = 64;
    cu_ = amdgpu::ComputeUnitCore::create("minmax_cu", config, &memory_, &cache_);
    decoder_ = Decoder::create(config.arch);
    wave_ = cu_->dispatch_wf(0, 0, 106, 256);
    ASSERT_NE(wave_, nullptr);
    wave_->set_exec(~uint64_t{0});
  }

  void TearDown() override {
    if (wave_)
      wave_->halt();
    util::set_force_scalar_for_testing(saved_scalar_);
  }

  void put(unsigned reg, uint32_t value) {
    for (unsigned lane = 0; lane != wave_->wf_size(); ++lane)
      cu_->write_vgpr(wave_->vgpr_alloc().base + reg, lane, value);
  }

  void check(std::array<uint32_t, 2> words, uint32_t expected) {
    SCOPED_TRACE(testing::Message() << "encoded words=" << std::hex << words[0] << ":" << words[1]);
    put(4, 0xdeadbeef);
    std::unique_ptr<Instruction> instruction(decode_valid(*decoder_, words.data()));
    ASSERT_NE(instruction, nullptr);
    SCOPED_TRACE(testing::Message() << instruction->mnemonic() << " words=" << std::hex << words[0]
                                    << ":" << words[1]);
    ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
    for (unsigned lane = 0; lane != wave_->wf_size(); ++lane)
      EXPECT_EQ(cu_->read_vgpr_storage(wave_->vgpr_alloc().base + 4, lane),
                wave_->exec() & (uint64_t{1} << lane) ? expected : 0xdeadbeefu)
          << "lane=" << lane;
  }

  static std::array<uint32_t, 2> vop3(uint32_t opcode, uint32_t opsel = 0, uint32_t omod = 0) {
    return {0xd0000004u | (opcode << 16) | (opsel << 11),
            256u | (257u << 9) | (258u << 18) | (omod << 27)};
  }

  bool saved_scalar_ = false;
  amdgpu::GpuMemory memory_{"numeric_memory"};
  amdgpu::L2Cache cache_{"numeric_cache"};
  std::unique_ptr<amdgpu::ComputeUnitCore> cu_;
  std::unique_ptr<Decoder> decoder_;
  amdgpu::Wavefront *wave_ = nullptr;
};

TEST_P(LegacyMinMaxInstruction, MinMaxSignedZeroAndSignalingNanSelection) {
  for (bool ieee : {false, true}) {
    wave_->set_mode_raw(ieee ? amdgpu::Wavefront::IEEE_BIT : 0);
    for (bool reverse : {false, true}) {
      put(0, reverse ? 0 : 0x80000000);
      put(1, reverse ? 0x80000000 : 0);
      check({(10u << 25) | (4u << 17) | (1u << 9) | 256u, 0}, 0x80000000);
      check({(11u << 25) | (4u << 17) | (1u << 9) | 256u, 0}, 0);
      check(vop3(cdna4::kVMinF32Vop3), 0x80000000);
      check(vop3(cdna4::kVMaxF32Vop3), 0);
      put(0, reverse ? 0x3f800000 : 0x7f800123);
      put(1, reverse ? 0x7f800123 : 0x3f800000);
      check(vop3(cdna4::kVMinF32Vop3), ieee ? 0x7fc00123 : 0x3f800000);
      check(vop3(cdna4::kVMaxF32Vop3), ieee ? 0x7fc00123 : 0x3f800000);
    }
  }
}

TEST_P(LegacyMinMaxInstruction, MinimumMaximumPropagateNanPayload) {
  if (std::get<0>(GetParam()) != ROCJITSU_CODE_ARCH_CDNA4)
    GTEST_SKIP() << "MINIMUM3/MAXIMUM3 starts with CDNA4";
  for (unsigned nan_source = 0; nan_source != 3; ++nan_source) {
    put(0, 0x3f800000);
    put(1, 0x40000000);
    put(2, 0x40400000);
    put(nan_source, 0xff800123);
    check(vop3(cdna4::kVMinimum3F32Vop3), 0xffc00123);
    check(vop3(cdna4::kVMaximum3F32Vop3), 0xffc00123);
  }
}

TEST_P(LegacyMinMaxInstruction, DoubleSelectionPreservesEncodings) {
  constexpr uint64_t kNegativeZero = 0x8000000000000000ull;
  constexpr uint64_t kOne = 0x3ff0000000000000ull;
  constexpr uint64_t kSignaling = 0x7ff0000000000123ull;
  constexpr uint64_t kQuiet = 0x7ff8000000000123ull;
  for (bool ieee : {false, true}) {
    for (uint32_t mode = 0; mode != 4; ++mode) {
      wave_->set_mode_raw(((mode ^ 1u) << 4) | (mode << 6) |
                          (ieee ? amdgpu::Wavefront::IEEE_BIT : 0));
      for (const auto &[a, b, minimum, maximum] : std::array<std::array<uint64_t, 4>, 6>{{
               {kNegativeZero, 0, kNegativeZero, 0},
               {0, 1, 0, mode & 1u},
               {kOne, kSignaling, ieee ? kQuiet : kOne, ieee ? kQuiet : kOne},
               {kSignaling, kOne, ieee ? kQuiet : kOne, ieee ? kQuiet : kOne},
               {kQuiet, kQuiet + 1, kQuiet + 1, kQuiet + 1},
               {kQuiet + 1, kQuiet, kQuiet, kQuiet},
           }}) {
        put(0, uint32_t(a));
        put(1, uint32_t(a >> 32));
        put(2, uint32_t(b));
        put(3, uint32_t(b >> 32));
        for (bool maximum_op : {false, true}) {
          const auto words =
              cdna4::build_vop3(maximum_op ? cdna4::kVMaxF64Vop3 : cdna4::kVMinF64Vop3,
                                {.vdst = 4, .src0 = 256, .src1 = 258});
          std::unique_ptr<Instruction> inst(decode_valid(*decoder_, words.data()));
          ASSERT_NE(inst, nullptr);
          ASSERT_TRUE(cu_->execute_instruction(inst.get(), *wave_).succeeded());
          const uint64_t expected = maximum_op ? maximum : minimum;
          for (uint32_t lane = 0; lane != wave_->wf_size(); ++lane) {
            EXPECT_EQ(cu_->read_vgpr_storage(wave_->vgpr_alloc().base + 4, lane),
                      uint32_t(expected));
            EXPECT_EQ(cu_->read_vgpr_storage(wave_->vgpr_alloc().base + 5, lane),
                      uint32_t(expected >> 32));
          }
        }
      }
    }
  }
}

TEST_P(LegacyMinMaxInstruction, MedianNanThirdSourceUsesMinimum) {
  for (bool ieee : {false, true}) {
    wave_->set_mode_raw(ieee ? amdgpu::Wavefront::IEEE_BIT : 0);
    put(0, 0x3f800000);
    put(1, 0x40000000);
    put(2, 0x7fc00123);
    check(vop3(cdna4::kVMed3F32Vop3), 0x3f800000);
    put(2, 0x7f800123);
    check(vop3(cdna4::kVMed3F32Vop3), ieee ? 0x7fc00123 : 0x3f800000);
    put(0, 0x3c00);
    put(1, 0x4000);
    put(2, 0x7e01);
    check(vop3(cdna4::kVMed3F16Vop3), 0x3c00);
    put(2, 0x7c01);
    check(vop3(cdna4::kVMed3F16Vop3), ieee ? 0x7e01 : 0x3c00);
  }
}

TEST_P(LegacyMinMaxInstruction, PackedHalfMinMaxSelectNanPayloads) {
  for (bool ieee : {false, true}) {
    wave_->set_mode_raw((3u << 6) | (ieee ? amdgpu::Wavefront::IEEE_BIT : 0));
    for (auto opcode : {cdna4::kVPkMinF16Vop3p, cdna4::kVPkMaxF16Vop3p}) {
      for (const auto &input :
           std::array<std::array<uint16_t, 4>, 4>{{{0x7e01, 0x7e02, 0x7e02, 0x7e02},
                                                   {0x7e01, 0x7c02, 0x7c02, 0x7e02},
                                                   {0x7c01, 0x3c00, 0x3c00, 0x7e01},
                                                   {0x7c01, 0x7c02, 0x7c02, 0x7e01}}}) {
        put(0, uint32_t(input[0]) * 0x10001u);
        put(1, uint32_t(input[1]) * 0x10001u);
        check(cdna4::build_vop3p(opcode, {.vdst = 4, .src0 = 256, .src1 = 257, .op_sel_hi = 3}),
              uint32_t(input[ieee ? 3 : 2]) * 0x10001u);
      }
    }
  }
}

// CDNA1-4, RDNA1/2 and RDNA3/3.5 each share the encodings used here.
class LegacySelectionPolicy : public LegacyMinMaxInstruction {
protected:
  bool rdna12() const {
    return std::get<0>(GetParam()) == ROCJITSU_CODE_ARCH_RDNA1 ||
           std::get<0>(GetParam()) == ROCJITSU_CODE_ARCH_RDNA2;
  }

  bool rdna() const {
    return rdna12() || std::get<0>(GetParam()) == ROCJITSU_CODE_ARCH_RDNA3 ||
           std::get<0>(GetParam()) == ROCJITSU_CODE_ARCH_RDNA3_5;
  }

  void check_vop3(uint32_t cdna_opcode, uint32_t rdna_opcode, uint32_t expected,
                  bool half = false) {
    if (half && rdna())
      expected |= 0xdead0000u; // RDNA half writes preserve the other destination half.
    if (rdna12()) {
      // Binary min/max opcodes agree across RDNA; MED3 moved in RDNA3.
      static_assert(rdna2::kVMinF32Vop3 == rdna3::kVMinF32Vop3 &&
                    rdna2::kVMaxF32Vop3 == rdna3::kVMaxF32Vop3 &&
                    rdna2::kVMinF16Vop3 == rdna3::kVMinF16Vop3 &&
                    rdna2::kVMaxF16Vop3 == rdna3::kVMaxF16Vop3);
      if (rdna_opcode == rdna3::kVMed3F16Vop3)
        rdna_opcode = rdna2::kVMed3F16Vop3;
      else if (rdna_opcode == rdna3::kVMed3F32Vop3)
        rdna_opcode = rdna2::kVMed3F32Vop3;
      check(rdna2::build_vop3(rdna_opcode, {.vdst = 4, .src0 = 256, .src1 = 257, .src2 = 258}),
            expected);
      return;
    }
    check(rdna()
              ? rdna3::build_vop3(rdna_opcode, {.vdst = 4, .src0 = 256, .src1 = 257, .src2 = 258})
              : cdna4::build_vop3(cdna_opcode, {.vdst = 4, .src0 = 256, .src1 = 257, .src2 = 258}),
          expected);
  }
};

TEST_P(LegacySelectionPolicy, InputDenormalModeAcrossEncodings) {
  for (uint32_t mode = 0; mode != 4; ++mode) {
    const uint32_t half_mode = mode ^ 1u;
    wave_->set_mode_raw((mode << 4) | (half_mode << 6));
    SCOPED_TRACE(mode);
    for (bool maximum : {false, true}) {
      // Min(-denorm,+0) and max(+denorm,-0) expose input flushing separately
      // from output flushing, which unpacked selection does not apply.
      const uint32_t sign = maximum ? 0 : 0x80000000u;
      put(0, sign | 1u);
      put(1, sign ^ 0x80000000u);
      const uint32_t expected = sign | (mode & 1u);
      const uint32_t op = (rdna() ? rdna3::kVMinF32Vop2 : 10u) + maximum;
      check({(op << 25) | (4u << 17) | (1u << 9) | 256u, 0}, expected);
      check_vop3(maximum ? cdna4::kVMaxF32Vop3 : cdna4::kVMinF32Vop3,
                 maximum ? rdna3::kVMaxF32Vop3 : rdna3::kVMinF32Vop3, expected);
      if (rdna() && !rdna12()) {
        const uint32_t dual_op = maximum ? 10u : 11u;
        check({(0x32u << 26) | (dual_op << 22) | (8u << 17) | (1u << 9) | 256u,
               (4u << 24) | (2u << 17) | 259u},
              expected);
      }
      put(0, (sign >> 16) | 1u);
      put(1, (sign ^ 0x80000000u) >> 16);
      const uint32_t half_expected = (sign >> 16) | (half_mode & 1u);
      const uint32_t half_opcode = rdna() ? (maximum ? rdna3::kVMaxF16Vop2 : rdna3::kVMinF16Vop2)
                                          : (maximum ? cdna4::kVMaxF16Vop2 : cdna4::kVMinF16Vop2);
      check({(half_opcode << 25) | (4u << 17) | (1u << 9) | 256u, 0},
            half_expected | (rdna() ? 0xdead0000u : 0));
      check_vop3(maximum ? cdna4::kVMaxF16Vop3 : cdna4::kVMinF16Vop3,
                 maximum ? rdna3::kVMaxF16Vop3 : rdna3::kVMinF16Vop3, half_expected, true);
    }
  }
}

TEST_P(LegacySelectionPolicy, HalfVop2NanAndZeroSelection) {
  for (bool ieee : {false, true}) {
    wave_->set_mode_raw((3u << 6) | (ieee ? amdgpu::Wavefront::IEEE_BIT : 0));
    for (bool maximum : {false, true}) {
      const uint32_t opcode = rdna() ? (maximum ? rdna3::kVMaxF16Vop2 : rdna3::kVMinF16Vop2)
                                     : (maximum ? cdna4::kVMaxF16Vop2 : cdna4::kVMinF16Vop2);
      const auto select = [&](uint32_t expected) {
        check({(opcode << 25) | (4u << 17) | (1u << 9) | 256u, 0},
              (rdna() ? 0xdead0000u : 0) | expected);
      };
      // a, b, CDNA/RDNA non-IEEE results, CDNA/RDNA IEEE results.
      for (auto input : std::array<std::array<uint16_t, 6>, 6>{
               {{0x7e01, 0x7e02, 0x7e02, 0x7e01, 0x7e02, 0x7e01},
                {0x7e02, 0x7e01, 0x7e01, 0x7e02, 0x7e01, 0x7e02},
                {0x7e01, 0x7c02, 0x7c02, 0x7e01, 0x7e02, 0x7e02},
                {0x7c01, 0x3c00, 0x3c00, 0x3c00, 0x7e01, 0x7e01},
                {0x3c00, 0x7c01, 0x3c00, 0x3c00, 0x7e01, 0x7e01},
                {0x7c01, 0x7c02, 0x7c02, 0x7c01, 0x7e01, 0x7e01}}}) {
        SCOPED_TRACE(testing::Message() << std::hex << input[0] << ":" << input[1]);
        put(0, input[0]);
        put(1, input[1]);
        select(input[(ieee ? 4 : 2) + rdna()]);
      }
      for (bool reverse : {false, true}) {
        put(0, reverse ? 0 : 0x8000);
        put(1, reverse ? 0x8000 : 0);
        select(maximum ? 0 : 0x8000);
      }
    }
  }
}

TEST_P(LegacySelectionPolicy, PackedSelectionPreservesDenormalResults) {
  for (uint32_t mode = 0; mode != 4; ++mode) {
    wave_->set_mode_raw(mode << 6);
    SCOPED_TRACE(mode);
    for (bool clamp : {false, true})
      for (bool maximum : {false, true}) {
        // Opposite signs in the two halves expose sign preservation independently.
        put(0, 0x80010001u);
        put(1, maximum ? 0xbc00bc00u : 0x3c003c00u);
        const uint32_t expected =
            (clamp ? 0 : 0x80000000u) | ((mode & 1u) ? (clamp ? 1u : 0x00010001u) : 0);
        check(rdna() ? rdna3::build_vop3p(
                           maximum ? rdna3::kVPkMaxF16Vop3p : rdna3::kVPkMinF16Vop3p,
                           {.vdst = 4, .clamp = clamp, .src0 = 256, .src1 = 257, .op_sel_hi = 3})
                     : cdna4::build_vop3p(
                           maximum ? cdna4::kVPkMaxF16Vop3p : cdna4::kVPkMinF16Vop3p,
                           {.vdst = 4, .clamp = clamp, .src0 = 256, .src1 = 257, .op_sel_hi = 3}),
              expected);
      }
  }
}

TEST_P(LegacySelectionPolicy, MedianNanSourceAndPayloadPriority) {
  for (bool half : {false, true}) {
    const uint32_t one = half ? 0x3c00 : 0x3f800000;
    const uint32_t two = half ? 0x4000 : 0x40000000;
    const uint32_t three = half ? 0x4200 : 0x40400000;
    const uint32_t signaling = half ? 0x7c01 : 0x7f800123;
    const uint32_t quiet = half ? 0x7e01 : 0x7fc00123;
    const auto median = [&](uint32_t expected) {
      check_vop3(half ? cdna4::kVMed3F16Vop3 : cdna4::kVMed3F32Vop3,
                 half ? rdna3::kVMed3F16Vop3 : rdna3::kVMed3F32Vop3, expected, half);
    };
    for (bool ieee : {false, true}) {
      wave_->set_mode_raw(ieee ? amdgpu::Wavefront::IEEE_BIT : 0);
      for (unsigned source = 0; source != 3; ++source) {
        SCOPED_TRACE(testing::Message()
                     << "half=" << half << " ieee=" << ieee << " source=" << source);
        put(0, one);
        put(1, two);
        put(2, three);
        put(source, quiet);
        median(source == 0 ? two : one);
        put(source, signaling);
        // MIN3 quiets a signaling NaN in the first pair before selecting
        // against src2. Only a signaling src2 survives the second selection.
        median(ieee ? (source == 2 ? quiet : three) : (source == 0 ? two : one));
      }
      put(0, quiet);
      put(1, quiet + 1);
      put(2, quiet + 2);
      median(rdna() ? quiet : quiet + 2);
    }
  }
}

TEST_P(LegacySelectionPolicy, MedianSignedZeroOrdering) {
  for (bool half : {false, true}) {
    const uint32_t negative_zero = half ? 0x8000 : 0x80000000;
    const uint32_t negative_one = half ? 0xbc00 : 0xbf800000;
    for (bool ieee : {false, true}) {
      wave_->set_mode_raw(ieee ? amdgpu::Wavefront::IEEE_BIT : 0);
      // gfx1010/gfx1030/gfx1100 measurements distinguish these cases from
      // floating == in the manual's MAX3/source comparison. CDNA follows that
      // pseudocode.
      for (auto inputs :
           std::array<std::array<uint32_t, 3>, 5>{{{negative_zero, negative_one, 0},
                                                   {negative_zero, 0, negative_zero},
                                                   {negative_zero, negative_zero, 0},
                                                   {negative_zero, 0, negative_one},
                                                   {negative_one, negative_zero, 0}}}) {
        for (unsigned source = 0; source != 3; ++source)
          put(source, inputs[source]);
        check_vop3(half ? cdna4::kVMed3F16Vop3 : cdna4::kVMed3F32Vop3,
                   half ? rdna3::kVMed3F16Vop3 : rdna3::kVMed3F32Vop3, rdna() ? negative_zero : 0,
                   half);
      }
    }
  }
}

TEST_P(LegacySelectionPolicy, PackedHalfNanPriorityAndClamp) {
  for (bool ieee : {false, true}) {
    wave_->set_mode_raw(ieee ? amdgpu::Wavefront::IEEE_BIT : 0);
    for (bool maximum : {false, true}) {
      for (bool clamp : {false, true}) {
        const auto packed = [&](uint32_t expected) {
          check(rdna() ? rdna3::build_vop3p(
                             maximum ? rdna3::kVPkMaxF16Vop3p : rdna3::kVPkMinF16Vop3p,
                             {.vdst = 4, .clamp = clamp, .src0 = 256, .src1 = 257, .op_sel_hi = 3})
                       : cdna4::build_vop3p(
                             maximum ? cdna4::kVPkMaxF16Vop3p : cdna4::kVPkMinF16Vop3p,
                             {.vdst = 4, .clamp = clamp, .src0 = 256, .src1 = 257, .op_sel_hi = 3}),
                expected);
        };
        put(0, 0x7e017e01);
        put(1, 0x7e027e02);
        packed(rdna() ? 0x7e017e01 : 0x7e027e02);
        put(0, 0x80008000);
        put(1, 0x80008000);
        packed(clamp ? 0 : 0x80008000);
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    Legacy, LegacySelectionPolicy,
    testing::Combine(testing::Values(ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2,
                                     ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4,
                                     ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2,
                                     ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5),
                     testing::Bool()));

class ScalarFloatMinMax : public LegacyMinMaxInstruction {};

TEST_P(ScalarFloatMinMax, HalfAndFloatUseArchitecturePolicy) {
  for (bool half : {false, true}) {
    for (bool ieee : {false, true}) {
      wave_->set_mode_raw(ieee ? amdgpu::Wavefront::IEEE_BIT : 0);
      for (bool maximum : {false, true}) {
        const uint32_t opcode = half ? (maximum ? rdna3_5::kSMaxF16Sop2 : rdna3_5::kSMinF16Sop2)
                                     : (maximum ? rdna3_5::kSMaxF32Sop2 : rdna3_5::kSMinF32Sop2);
        for (const auto &input : std::array<std::array<uint32_t, 4>, 7>{
                 {{0x80000000, 0, 0x80000000, 0},
                  {0x7fc02000, 0x3f800000, 0x3f800000, 0x3f800000},
                  {0x3f800000, 0x7fc02000, 0x3f800000, 0x3f800000},
                  {0x7fc02000, 0x7fc04000, 0x7fc02000, 0x7fc02000},
                  {0x7f802000, 0x3f800000, ieee ? 0x7fc02000u : 0x3f800000u,
                   ieee ? 0x7fc02000u : 0x3f800000u},
                  {0x3f800000, 0x7f802000, ieee ? 0x7fc02000u : 0x3f800000u,
                   ieee ? 0x7fc02000u : 0x3f800000u},
                  {0x7fc02000, 0x7f804000, ieee ? 0x7fc04000u : 0x7fc02000u,
                   ieee ? 0x7fc04000u : 0x7fc02000u}}}) {
          const auto bits = [half](uint32_t value) {
            return half ? uint32_t(util::f32_to_f16(std::bit_cast<float>(value))) : value;
          };
          cu_->write_sgpr(wave_->sgpr_alloc().base, bits(input[0]));
          cu_->write_sgpr(wave_->sgpr_alloc().base + 1, bits(input[1]));
          auto words = rdna3_5::build_sop2(opcode, {.ssrc0 = 0, .ssrc1 = 1, .sdst = 4});
          std::unique_ptr<Instruction> inst(decode_valid(*decoder_, words.data()));
          ASSERT_NE(inst, nullptr);
          ASSERT_TRUE(cu_->execute_instruction(inst.get(), *wave_).succeeded());
          EXPECT_EQ(cu_->read_sgpr(wave_->sgpr_alloc().base + 4), bits(input[maximum ? 3 : 2]));
        }
      }
    }
  }
}
INSTANTIATE_TEST_SUITE_P(Rdna35, ScalarFloatMinMax,
                         testing::Combine(testing::Values(ROCJITSU_CODE_ARCH_RDNA3_5),
                                          testing::Bool()));

class RdnaMinMax : public LegacyMinMaxInstruction {};

TEST_P(RdnaMinMax, NanSelectionAcrossScalarSimdAndDualIssue) {
  const auto arch = std::get<0>(GetParam());
  const bool number = arch == ROCJITSU_CODE_ARCH_RDNA4;
  const bool legacy = arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2;
  // RDNA1-3 hardware prefers src0 when both sources are NaN; RDNA4 *_NUM
  // also quiets the selected NaN. All cases run in both input orders.
  constexpr std::array<std::array<uint32_t, 2>, 5> inputs{{
      {0x7f800123u, 0x3f800000u},
      {0x7fc00123u, 0x7fc00456u},
      {0x7fc00123u, 0x7f800456u},
      {0x7f800123u, 0x7f800456u},
      {0x80000000u, 0u},
  }};
  for (bool ieee : {false, true}) {
    wave_->set_mode_raw(ieee ? amdgpu::Wavefront::IEEE_BIT : 0);
    for (const auto &input : inputs) {
      for (bool reverse : {false, true}) {
        const uint32_t a = input[reverse];
        const uint32_t b = input[!reverse];
        put(0, a);
        put(1, b);
        const bool nan_a = (a & 0x7fffffffu) > 0x7f800000u;
        const bool nan_b = (b & 0x7fffffffu) > 0x7f800000u;
        for (bool maximum : {false, true}) {
          uint32_t expected;
          if (!number && ieee && nan_a && !(a & 0x400000u))
            expected = a | 0x400000u;
          else if (!number && ieee && nan_b && !(b & 0x400000u))
            expected = b | 0x400000u;
          else if (nan_a && nan_b)
            expected = number ? a | 0x400000u : a;
          else if (nan_a)
            expected = b;
          else if (nan_b)
            expected = a;
          else
            expected = maximum ? 0 : 0x80000000u;
          const uint32_t op = (number ? rdna4::kVMinNumF32Vop2 : rdna3::kVMinF32Vop2) + maximum;
          check({(op << 25) | (4u << 17) | (1u << 9) | 256u, 0}, expected);
          check(number   ? rdna4::build_vop3(op + 256, {.vdst = 4, .src0 = 256, .src1 = 257})
                : legacy ? rdna2::build_vop3(op + 256, {.vdst = 4, .src0 = 256, .src1 = 257})
                         : rdna3::build_vop3(op + 256, {.vdst = 4, .src0 = 256, .src1 = 257}),
                expected);
          if (legacy)
            continue; // Dual issue starts with RDNA3.
          // VOPD X is MIN/MAX, Y is MOV. Destinations v4/v5 are disjoint.
          const uint32_t dual_op = maximum ? 10u : 11u;
          check({(0x32u << 26) | (dual_op << 22) | (8u << 17) | (1u << 9) | 256u,
                 (4u << 24) | (2u << 17) | 259u},
                expected);
        }
      }
    }
  }
}

TEST_P(RdnaMinMax, DoubleNanPayloadPriority) {
  const auto arch = std::get<0>(GetParam());
  if (arch == ROCJITSU_CODE_ARCH_RDNA4)
    GTEST_SKIP() << "RDNA4 uses MIN_NUM/MAX_NUM rather than legacy selection";
  const bool rdna12 = arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2;
  constexpr uint64_t kQuiet0 = 0x7ff8000000000000ull;
  constexpr uint64_t kQuiet1 = 0x7ff8000012345678ull;
  constexpr uint64_t kSignaling0 = 0x7ff0000000000001ull;
  constexpr uint64_t kSignaling1 = 0x7ff0000012345678ull;
  constexpr uint64_t kQuietBit = 0x0008000000000000ull;
  constexpr uint64_t kOne = 0x3ff0000000000000ull;
  for (bool ieee : {false, true}) {
    wave_->set_mode_raw(ieee ? amdgpu::Wavefront::IEEE_BIT : 0);
    for (const auto &[a, b, expected_non_ieee, expected_ieee] :
         std::array<std::array<uint64_t, 4>, 6>{{
             {kQuiet0, kQuiet1, kQuiet0, kQuiet0},
             {kQuiet1, kQuiet0, kQuiet1, kQuiet1},
             {kQuiet0, kSignaling1, kQuiet0, kSignaling1 | kQuietBit},
             {kSignaling0, kSignaling1, kSignaling0, kSignaling0 | kQuietBit},
             {kSignaling0, kOne, kOne, kSignaling0 | kQuietBit},
             {kOne, kSignaling0, kOne, kSignaling0 | kQuietBit},
         }}) {
      SCOPED_TRACE(testing::Message() << "ieee=" << ieee << " a=" << std::hex << a << " b=" << b);
      put(0, uint32_t(a));
      put(1, uint32_t(a >> 32));
      put(2, uint32_t(b));
      put(3, uint32_t(b >> 32));
      for (bool maximum : {false, true}) {
        put(4, 0xdeadbeefu);
        put(5, 0xdeadbeefu);
        const auto words =
            rdna12 ? rdna2::build_vop3(maximum ? rdna2::kVMaxF64Vop3 : rdna2::kVMinF64Vop3,
                                       {.vdst = 4, .src0 = 256, .src1 = 258})
                   : rdna3::build_vop3(maximum ? rdna3::kVMaxF64Vop3 : rdna3::kVMinF64Vop3,
                                       {.vdst = 4, .src0 = 256, .src1 = 258});
        std::unique_ptr<Instruction> inst(decode_valid(*decoder_, words.data()));
        ASSERT_NE(inst, nullptr);
        ASSERT_TRUE(cu_->execute_instruction(inst.get(), *wave_).succeeded());
        const uint64_t expected = ieee ? expected_ieee : expected_non_ieee;
        for (uint32_t lane = 0; lane != wave_->wf_size(); ++lane) {
          EXPECT_EQ(cu_->read_vgpr_storage(wave_->vgpr_alloc().base + 4, lane), uint32_t(expected));
          EXPECT_EQ(cu_->read_vgpr_storage(wave_->vgpr_alloc().base + 5, lane),
                    uint32_t(expected >> 32));
        }
      }
    }
  }
}

class PackedNumberSelection : public LegacyMinMaxInstruction {};

TEST_P(PackedNumberSelection, PreservesDenormalResults) {
  for (uint32_t mode = 0; mode != 4; ++mode) {
    wave_->set_mode_raw(mode << 6);
    SCOPED_TRACE(mode);
    for (bool clamp : {false, true})
      for (bool maximum : {false, true}) {
        put(0, 0x80010001u);
        put(1, maximum ? 0xbc00bc00u : 0x3c003c00u);
        check(rdna4::build_vop3p(
                  maximum ? rdna4::kVPkMaxNumF16Vop3p : rdna4::kVPkMinNumF16Vop3p,
                  {.vdst = 4, .clamp = clamp, .src0 = 256, .src1 = 257, .opsel_hi = 3}),
              (clamp ? 0 : 0x80000000u) | ((mode & 1u) ? (clamp ? 1u : 0x00010001u) : 0));
      }
  }
}

TEST_P(PackedNumberSelection, SignalingNanUsesNumberSemantics) {
  for (bool ieee : {false, true}) {
    wave_->set_mode_raw(ieee ? amdgpu::Wavefront::IEEE_BIT : 0);
    for (auto opcode : {rdna4::kVPkMinNumF16Vop3p, rdna4::kVPkMaxNumF16Vop3p}) {
      // Different halves and payloads detect source/half selection mistakes.
      for (const auto &[first, second, expected] : std::array<std::array<uint32_t, 3>, 3>{{
               {0xfc027c01u, 0x7c04fc03u, 0xfe027e01u},
               {0xfc027c01u, 0x3c004000u, 0x3c004000u},
               {0x3c004000u, 0xfc027c01u, 0x3c004000u},
           }}) {
        SCOPED_TRACE(testing::Message() << "ieee=" << ieee << " opcode=" << opcode
                                        << " first=" << std::hex << first << " second=" << second);
        put(0, first);
        put(1, second);
        check(rdna4::build_vop3p(opcode, {.vdst = 4, .src0 = 256, .src1 = 257, .opsel_hi = 3}),
              expected);
      }
    }
  }
}

TEST(LegacyMinMaxPolicyDeathTest, RejectsUnsupportedArchitectures) {
  static_assert(noexcept(amdgpu::fp_mode::legacy_minmax_policy(ROCJITSU_CODE_ARCH_CDNA1, false)));
  for (auto arch : {ROCJITSU_CODE_ARCH_RDNA4, ROCJITSU_CODE_ARCH_CDNA5, ROCJITSU_CODE_ARCH_RV32I,
                    ROCJITSU_CODE_ARCH_RV64I, ROCJITSU_CODE_ARCH_INVALID})
    EXPECT_DEATH(amdgpu::fp_mode::legacy_minmax_policy(arch, false), "");
}

INSTANTIATE_TEST_SUITE_P(Rdna4, PackedNumberSelection,
                         testing::Combine(testing::Values(ROCJITSU_CODE_ARCH_RDNA4),
                                          testing::Bool()));

INSTANTIATE_TEST_SUITE_P(
    Rdna, RdnaMinMax,
    testing::Combine(testing::Values(ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2,
                                     ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                                     ROCJITSU_CODE_ARCH_RDNA4),
                     testing::Bool()));

INSTANTIATE_TEST_SUITE_P(
    Cdna, LegacyMinMaxInstruction,
    testing::Combine(testing::Values(ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2,
                                     ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4),
                     testing::Bool()));

} // namespace
} // namespace rocjitsu
