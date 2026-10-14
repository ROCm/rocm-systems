// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "decode_test_util.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/simd_test_hooks.h"

#include <array>
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
    for (unsigned lane = 0; lane != 64; ++lane)
      cu_->write_vgpr(wave_->vgpr_alloc().base + reg, lane, value);
  }

  void check(std::array<uint32_t, 2> words, uint32_t expected) {
    put(4, 0xdeadbeef);
    std::unique_ptr<Instruction> instruction(decode_valid(*decoder_, words.data()));
    ASSERT_NE(instruction, nullptr);
    SCOPED_TRACE(testing::Message() << instruction->mnemonic() << " words=" << std::hex << words[0]
                                    << ":" << words[1]);
    ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
    for (unsigned lane = 0; lane != 64; ++lane)
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
    check(vop1(67), expected);
    check(vop3(387, 0, 2), expected); // Integer destinations ignore OMOD.
    put(0, input << 16);
    check(vop3(387, 9, 2), (expected << 16) | 0xbeef);
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
    check(vop1(48), expected);
    check(vop3(368, 0, 2), expected);
  }
}

TEST_P(NumericInstructionRegression, PackedNormalizedHalfAliasesExecute) {
  put(0, 0x3c003800); // High 1, low 0.5.
  put(1, 0xbc00b800); // High -1, low -0.5.
  check(vop3(665), 0xc0004000);
  check(vop3(665, 3), 0x80017fff);
  check(vop3(666), 0x00008000);
  check(vop3(666, 3), 0x0000ffff);
}

TEST_P(NumericInstructionRegression, LegacyMathDecodesAndExecutesBothForms) {
  put(0, 0x3f800000);
  check(vop1(75), 0x40000000); // exp2(1) = 2.
  check(vop3(395), 0x40000000);
  put(0, 0x40000000);
  check(vop1(76), 0x3f800000); // log2(2) = 1.
  check(vop3(396), 0x3f800000);
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
      check(vop1(88), expected);
      check(vop3(408), expected);
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
      check(vop3(266), 0x80000000);
      check(vop3(267), 0);
      put(0, reverse ? 0x3f800000 : 0x7f800123);
      put(1, reverse ? 0x7f800123 : 0x3f800000);
      check(vop3(266), ieee ? 0x7fc00123 : 0x3f800000);
      check(vop3(267), ieee ? 0x7fc00123 : 0x3f800000);
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
    check(vop3(680), 0xffc00123);
    check(vop3(681), 0xffc00123);
  }
}

INSTANTIATE_TEST_SUITE_P(Cdna, NumericInstructionRegression,
                         testing::Combine(testing::Values(ROCJITSU_CODE_ARCH_CDNA3,
                                                          ROCJITSU_CODE_ARCH_CDNA4),
                                          testing::Bool()));

} // namespace
} // namespace rocjitsu
