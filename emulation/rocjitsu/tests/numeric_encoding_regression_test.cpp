// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "decode_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/shared/instruction_encoding.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "util/simd_test_hooks.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstdint>
#include <memory>
#include <tuple>

namespace rocjitsu {
namespace {

class NumericEncodingRegression : public testing::TestWithParam<std::tuple<rj_code_arch_t, bool>> {
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
    cu_ = amdgpu::ComputeUnitCore::create("numeric_encoding", config, &memory_, &cache_);
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

class LegacyMathRegression : public NumericEncodingRegression {};

TEST_P(LegacyMathRegression, LegacyMathDecodesAndExecutesBothForms) {
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

TEST_P(NumericEncodingRegression, PrngUsesFeedbackAndExecMask) {
  const bool is_cdna5 = std::get<0>(GetParam()) == ROCJITSU_CODE_ARCH_CDNA5;
  constexpr std::array<std::array<uint32_t, 2>, 4> cases{{
      {0, 0},
      {1, 2},
      {0x80000000, 197},
      {0xffffffff, 0xffffff3b},
  }};
  for (uint64_t mask : {~uint64_t{0}, uint64_t{0x3333333333333333}}) {
    wave_->set_exec(mask);
    for (const auto &[input, expected] : cases) {
      put(0, input);
      check(vop1(is_cdna5 ? cdna5::kVPrngB32Vop1 : cdna4::kVPrngB32Vop1), expected);
      check(is_cdna5 ? cdna5::build_vop3(cdna5::kVPrngB32Vop3, {.vdst = 4, .src0 = 256})
                     : vop3(cdna4::kVPrngB32Vop3),
            expected);
    }
    if (!is_cdna5)
      continue;

    // Distinct per-lane inputs distinguish source permutation from ignored DPP.
    // The partial EXEC mask keeps each selected source/destination pair active.
    auto dpp = std::bit_cast<cdna5::Vop1VopDpp16MachineInst>(std::array{
        cdna5::build_vop1(cdna5::kVPrngB32Vop1, {.src0 = amdgpu::SRC_DPP, .vdst = 4})[0], 0u});
    dpp.dpp_ctrl = 0xb1; // quad_perm:[1,0,3,2].
    dpp.bank_mask = dpp.row_mask = 15;
    auto dpp8 = std::bit_cast<cdna5::Vop1VopDpp8MachineInst>(std::array{
        cdna5::build_vop1(cdna5::kVPrngB32Vop1, {.src0 = amdgpu::SRC_DPP8_FI_0, .vdst = 4})[0],
        0u});
    dpp8.lane_sel_0 = 1;
    dpp8.lane_sel_1 = 0;
    dpp8.lane_sel_2 = 3;
    dpp8.lane_sel_3 = 2;
    dpp8.lane_sel_4 = 5;
    dpp8.lane_sel_5 = 4;
    dpp8.lane_sel_6 = 7;
    dpp8.lane_sel_7 = 6;
    for (const auto &words : {std::bit_cast<std::array<uint32_t, 2>>(dpp),
                              std::bit_cast<std::array<uint32_t, 2>>(dpp8)}) {
      put(4, 0xdeadbeef);
      for (unsigned lane = 0; lane < wave_->wf_size(); ++lane)
        cu_->write_vgpr(wave_->vgpr_alloc().base, lane, cases[lane % cases.size()][0]);
      std::unique_ptr<Instruction> instruction(decode_valid(*decoder_, words.data()));
      ASSERT_NE(instruction, nullptr);
      ASSERT_TRUE(cu_->execute_instruction(instruction.get(), *wave_).succeeded());
      for (unsigned lane = 0; lane < wave_->wf_size(); ++lane)
        EXPECT_EQ(cu_->read_vgpr_storage(wave_->vgpr_alloc().base + 4, lane),
                  mask & (uint64_t{1} << lane) ? cases[(lane ^ 1) % cases.size()][1] : 0xdeadbeefu)
            << "lane=" << lane << " selector=" << (words[0] & 0x1ff);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    Legacy, LegacyMathRegression,
    testing::Combine(testing::Values(ROCJITSU_CODE_ARCH_CDNA1, ROCJITSU_CODE_ARCH_CDNA2,
                                     ROCJITSU_CODE_ARCH_CDNA3, ROCJITSU_CODE_ARCH_CDNA4),
                     testing::Bool()));

INSTANTIATE_TEST_SUITE_P(Cdna, NumericEncodingRegression,
                         testing::Combine(testing::Values(ROCJITSU_CODE_ARCH_CDNA4,
                                                          ROCJITSU_CODE_ARCH_CDNA5),
                                          testing::Bool()));

} // namespace
} // namespace rocjitsu
