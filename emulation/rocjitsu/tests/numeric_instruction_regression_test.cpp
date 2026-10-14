// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "decode_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna2/builders.h"
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
#include "util/simd.h"
#include "util/simd_test_hooks.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstdint>
#include <memory>
#include <tuple>

namespace rocjitsu {
namespace {

class NumericInstructionRegression
    : public testing::TestWithParam<std::tuple<rj_code_arch_t, bool>> {
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
    cu_ = amdgpu::ComputeUnitCore::create("numeric_repro", config, &memory_, &cache_);
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

  static std::array<uint32_t, 2> vop1(uint32_t opcode) {
    return {0x7e000100u | (4u << 17) | (opcode << 9), 0};
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

TEST_P(NumericInstructionRegression, HalfFrexpProducesIntegerExponent) {
  wave_->set_mode_raw(3u << 6);
  // AMD CDNA3/4 FREXP returns an integer, including negative exponents.
  constexpr std::array<std::array<uint32_t, 2>, 9> cases{{
      {0x3c00, 1},
      {0xc000, 2},
      {0x3400, 0xffff},
      {0x0400, 0xfff3},
      {0x0001, 0xffe9},
      {0x0000, 0},
      {0x8000, 0},
      {0x7c00, 0},
      {0x7e01, 0},
  }};
  for (const auto &[input, expected] : cases) {
    SCOPED_TRACE(input);
    put(0, input);
    check(vop1(cdna4::kVFrexpExpI16F16Vop1), expected);
    check(vop3(cdna4::kVFrexpExpI16F16Vop3, 0, 2), expected); // Integer destinations ignore OMOD.
    put(0, input << 16);
    check(vop3(cdna4::kVFrexpExpI16F16Vop3, 9, 2), (expected << 16) | 0xbeef);
  }
}

TEST_P(NumericInstructionRegression, DoubleFrexpVop3ProducesIntegerExponent) {
  for (const auto &[high, expected] :
       std::array<std::array<uint32_t, 2>, 5>{{{0x3ff00000, 1},
                                               {0x3fd00000, 0xffffffff},
                                               {0x00000000, 0},
                                               {0x7ff00000, 0},
                                               {0x7ff80000, 0}}}) {
    put(0, 0);
    put(1, high);
    check(vop1(cdna4::kVFrexpExpI32F64Vop1), expected);
    check(vop3(cdna4::kVFrexpExpI32F64Vop3, 0, 2), expected);
  }
}

TEST_P(NumericInstructionRegression, PackedNormalizedHalfAliasesExecute) {
  put(0, 0x3c003800); // High 1, low 0.5.
  put(1, 0xbc00b800); // High -1, low -0.5.
  check(vop3(cdna4::kVCvtPknormI16F16Vop3), 0xc0004000);
  check(vop3(cdna4::kVCvtPknormI16F16Vop3, 3), 0x80017fff);
  check(vop3(cdna4::kVCvtPknormU16F16Vop3), 0x00008000);
  check(vop3(cdna4::kVCvtPknormU16F16Vop3, 3), 0x0000ffff);
}

class NumericLegacyRegression : public NumericInstructionRegression {};

TEST_P(NumericLegacyRegression, LegacyMathDecodesAndExecutesBothForms) {
  put(0, 0x3f800000);
  check(vop1(cdna4::kVExpLegacyF32Vop1), 0x40000000); // exp2(1) = 2.
  check(vop3(cdna4::kVExpLegacyF32Vop3), 0x40000000);
  put(0, 0x40000000);
  check(vop1(cdna4::kVLogLegacyF32Vop1), 0x3f800000); // log2(2) = 1.
  check(vop3(cdna4::kVLogLegacyF32Vop3), 0x3f800000);
  wave_->set_mode_raw(3u << 4);
  put(0, 0xc3020000); // exp2(-130) flushes the output under either guest mode.
  check(vop1(cdna4::kVExpLegacyF32Vop1), 0);
  check(vop3(cdna4::kVExpLegacyF32Vop3), 0);
  put(0, 0xff800000); // exp2(-infinity).
  check(vop1(cdna4::kVExpLegacyF32Vop1), 0);
  for (uint32_t zero : {0u, 0x80000000u}) {
    put(0, zero);
    check(vop1(cdna4::kVLogLegacyF32Vop1), 0xff800000);
    check(vop3(cdna4::kVLogLegacyF32Vop3), 0xff800000);
  }
}

TEST_P(NumericInstructionRegression, PrngUsesFeedbackAndExecMask) {
  if (std::get<0>(GetParam()) != ROCJITSU_CODE_ARCH_CDNA4)
    GTEST_SKIP() << "PRNG starts with CDNA4";
  constexpr std::array<std::array<uint32_t, 2>, 4> cases{{
      {0, 0},
      {1, 2},
      {0x80000000, 197},
      {0xffffffff, 0xffffff3b},
  }};
  for (uint64_t mask : {~uint64_t{0}, uint64_t{0x5555555555555555}}) {
    wave_->set_exec(mask);
    for (const auto &[input, expected] : cases) {
      put(0, input);
      check(vop1(cdna4::kVPrngB32Vop1), expected);
      check(vop3(cdna4::kVPrngB32Vop3), expected);
    }
  }
}

TEST_P(NumericInstructionRegression, MinMaxSignedZeroAndSignalingNanSelection) {
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

TEST_P(NumericInstructionRegression, MinimumMaximumPropagateNanPayload) {
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

TEST_P(NumericInstructionRegression, FrexpDenormModeAppliesToHalfAndDouble) {
  for (uint32_t denorm : {0u, 1u, 2u, 3u}) {
    wave_->set_mode_raw(denorm << 6);
    put(0, 1);
    put(1, 0);
    const uint32_t half_expected = denorm & 1u ? 0xffe9u : 0u;
    check(vop1(cdna4::kVFrexpExpI16F16Vop1), half_expected);
    check(vop3(cdna4::kVFrexpExpI16F16Vop3), half_expected);
    const uint32_t double_expected = denorm & 1u ? uint32_t(-1073) : 0u;
    check(vop1(cdna4::kVFrexpExpI32F64Vop1), double_expected);
    check(vop3(cdna4::kVFrexpExpI32F64Vop3), double_expected);
  }
}

TEST_P(NumericInstructionRegression, MedianNanThirdSourceUsesMinimum) {
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

TEST_P(NumericInstructionRegression, PackedHalfMinMaxSelectNanPayloads) {
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

class ScalarFloatMinMaxRegression : public NumericInstructionRegression {};
TEST_P(ScalarFloatMinMaxRegression, HalfAndFloatUseArchitecturePolicy) {
  for (bool half : {false, true}) {
    for (bool ieee : {false, true}) {
      wave_->set_mode_raw(ieee ? amdgpu::Wavefront::IEEE_BIT : 0);
      for (bool maximum : {false, true}) {
        const uint32_t opcode = half ? (maximum ? rdna3_5::kSMaxF16Sop2 : rdna3_5::kSMinF16Sop2)
                                     : (maximum ? rdna3_5::kSMaxF32Sop2 : rdna3_5::kSMinF32Sop2);
        for (const auto &input : std::array<std::array<uint32_t, 4>, 4>{
                 {{0x80000000, 0, 0x80000000, 0},
                  {0x7fc02000, 0x3f800000, 0x3f800000, 0x3f800000},
                  {0x3f800000, 0x7fc02000, 0x3f800000, 0x3f800000},
                  {0x7fc02000, 0x7fc04000, 0x7fc02000, 0x7fc02000}}}) {
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
INSTANTIATE_TEST_SUITE_P(Rdna35, ScalarFloatMinMaxRegression,
                         testing::Combine(testing::Values(ROCJITSU_CODE_ARCH_RDNA3_5),
                                          testing::Bool()));

class RdnaMinMaxRegression : public NumericInstructionRegression {};

TEST_P(RdnaMinMaxRegression, NanSelectionAcrossScalarSimdAndDualIssue) {
  const auto arch = std::get<0>(GetParam());
  const bool number = arch == ROCJITSU_CODE_ARCH_RDNA4;
  const bool legacy = arch == ROCJITSU_CODE_ARCH_RDNA1 || arch == ROCJITSU_CODE_ARCH_RDNA2;
  // Independent bit-pattern expectations from the RDNA3 MIN/MAX and RDNA4
  // MIN_NUM/MAX_NUM pseudocode; all cases run in both input orders.
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
            expected = number ? a | 0x400000u : (legacy ? b : a);
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

INSTANTIATE_TEST_SUITE_P(
    Rdna, RdnaMinMaxRegression,
    testing::Combine(testing::Values(ROCJITSU_CODE_ARCH_RDNA1, ROCJITSU_CODE_ARCH_RDNA2,
                                     ROCJITSU_CODE_ARCH_RDNA3, ROCJITSU_CODE_ARCH_RDNA3_5,
                                     ROCJITSU_CODE_ARCH_RDNA4),
                     testing::Bool()));

// Golden cases: CDNA non-IEEE, CDNA IEEE, RDNA non-IEEE, RDNA IEEE,
// IEEE-2019 Number, and IEEE-2019 propagating forms, in that order.
template <typename Float> void check_minmax_nan_cases() {
  using Bits = std::conditional_t<sizeof(Float) == 4, uint32_t, uint64_t>;
  const auto encode = [](uint32_t bits) -> Bits {
    if constexpr (sizeof(Float) == 4)
      return bits;
    else {
      if ((bits & 0x7fffffffu) >= 0x7f800000u)
        return (uint64_t(bits & 0x80000000u) << 32) | 0x7ff0000000000000ull |
               (uint64_t(bits & 0x7fffffu) << 29);
      return std::bit_cast<uint64_t>(double(std::bit_cast<float>(bits)));
    }
  };
  constexpr std::array policies{util::MinMaxPolicy::Legacy, util::MinMaxPolicy::LegacyIeee,
                                util::MinMaxPolicy::Rdna3,  util::MinMaxPolicy::Rdna3Ieee,
                                util::MinMaxPolicy::Number, util::MinMaxPolicy::Propagate};
  constexpr std::array<std::array<uint32_t, 8>, 6> cases{{
      {0x7f800123, 0x3f800000, 0x3f800000, 0x7fc00123, 0x3f800000, 0x7fc00123, 0x3f800000,
       0x7fc00123},
      {0x3f800000, 0x7f800123, 0x3f800000, 0x7fc00123, 0x3f800000, 0x7fc00123, 0x3f800000,
       0x7fc00123},
      {0x7fc00123, 0x7fc00456, 0x7fc00456, 0x7fc00456, 0x7fc00123, 0x7fc00123, 0x7fc00123,
       0x7fc00123},
      {0x7fc00456, 0x7fc00123, 0x7fc00123, 0x7fc00123, 0x7fc00456, 0x7fc00456, 0x7fc00456,
       0x7fc00456},
      {0x7fc00123, 0x7f800456, 0x7f800456, 0x7fc00456, 0x7fc00123, 0x7fc00456, 0x7fc00123,
       0x7fc00456},
      {0x7f800123, 0x7f800456, 0x7f800456, 0x7fc00123, 0x7f800123, 0x7fc00123, 0x7fc00123,
       0x7fc00123},
  }};
  for (const auto &row : cases) {
    const Float a = std::bit_cast<Float>(encode(row[0]));
    const Float b = std::bit_cast<Float>(encode(row[1]));
    for (size_t p = 0; p != policies.size(); ++p) {
      SCOPED_TRACE(testing::Message() << "a=" << row[0] << " b=" << row[1] << " policy=" << p);
      EXPECT_EQ(std::bit_cast<Bits>(util::min_num(a, b, policies[p])), encode(row[p + 2]));
      EXPECT_EQ(std::bit_cast<Bits>(util::max_num(a, b, policies[p])), encode(row[p + 2]));
      const auto lo =
          util::min_num_simd(util::native<Float>(a), util::native<Float>(b), policies[p]);
      const auto hi =
          util::max_num_simd(util::native<Float>(a), util::native<Float>(b), policies[p]);
      for (size_t lane = 0; lane != lo.size(); ++lane) {
        EXPECT_EQ(std::bit_cast<Bits>(Float(lo[lane])), encode(row[p + 2]));
        EXPECT_EQ(std::bit_cast<Bits>(Float(hi[lane])), encode(row[p + 2]));
      }
    }
  }
}

TEST(NumericMinMaxHelpers, FloatAndDoubleNanPayloadsMatchGoldenCases) {
  check_minmax_nan_cases<float>();
  check_minmax_nan_cases<double>();
}

TEST(NumericMinMaxHelpers, PackedDoubleNumberPolicyKeepsSelectedNanPayload) {
  for (auto operation :
       {amdgpu::fp_mode::BinaryF64Op::MinimumNumber, amdgpu::fp_mode::BinaryF64Op::MaximumNumber}) {
    EXPECT_EQ(
        amdgpu::fp_mode::binary_f64(0x7ff0000000000123ull, 0x3ff0000000000000ull, operation, 0, 3),
        0x3ff0000000000000ull);
    EXPECT_EQ(
        amdgpu::fp_mode::binary_f64(0x7ff0000000000123ull, 0x7ff8000000000456ull, operation, 0, 3),
        0x7ff8000000000123ull);
  }
}

INSTANTIATE_TEST_SUITE_P(
    Legacy, NumericLegacyRegression,
    testing::Combine(testing::Values(ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2,
                                     ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4),
                     testing::Bool()));

INSTANTIATE_TEST_SUITE_P(Cdna, NumericInstructionRegression,
                         testing::Combine(testing::Values(ROCJITSU_CODE_ARCH_CDNA3,
                                                          ROCJITSU_CODE_ARCH_CDNA4),
                                          testing::Bool()));

} // namespace
} // namespace rocjitsu
