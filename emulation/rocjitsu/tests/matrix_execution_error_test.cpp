// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "mma_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/vop3.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/vop3p.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/vop3p.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <vector>

namespace {
using namespace rocjitsu;

class MatrixExecutionErrorTest : public ::testing::Test {
protected:
  void initialize(rj_code_arch_t arch, uint32_t wave_size = 0) {
    amdgpu::ComputeUnitCore::Config config{};
    config.arch = arch;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = arch == ROCJITSU_CODE_ARCH_CDNA4 ? 102 : 106;
    config.vgprs_per_wf = 128;
    config.lds_size_kb = 64;
    cu = amdgpu::ComputeUnitCore::create("matrix_error_cu", config, &memory, &l2);
    ASSERT_NE(cu, nullptr);
    wave = cu->dispatch_wf(0, 0, config.sgprs_per_wf, 128, wave_size);
    ASSERT_NE(wave, nullptr);
    wave->set_exec(5);
    wave->set_vcc(0x1234);
    wave->write_scc(1);
    for (uint32_t reg = 0; reg < 128; ++reg)
      for (uint32_t lane = 0; lane < wave->wf_size(); ++lane)
        cu->write_vgpr(wave->vgpr_alloc().base + reg, lane,
                       reg < 64 ? 0 : 0xdead0000u + reg * 64 + lane);
  }

  void TearDown() override {
    if (wave && !wave->is_halted())
      wave->halt();
  }

  void expect_rejected(Instruction &instruction) {
    const auto before = read_vgprs();
    const auto exec = wave->exec(), vcc = wave->vcc();
    const auto scc = wave->read_scc();
    ASSERT_NE(instruction.execute, nullptr);
    EXPECT_NO_THROW({ EXPECT_TRUE(cu->execute_instruction(&instruction, *wave).failed()); });
    EXPECT_EQ(wave->instruction_execution_error(),
              amdgpu::InstructionExecutionError::UnsupportedOperandValue);
    EXPECT_EQ(read_vgprs(), before);
    EXPECT_EQ(wave->exec(), exec);
    EXPECT_EQ(wave->vcc(), vcc);
    EXPECT_EQ(wave->read_scc(), scc);
    EXPECT_FALSE(wave->is_halted());
  }

  std::vector<uint32_t> read_vgprs() const {
    std::vector<uint32_t> words;
    for (uint32_t reg = 0; reg < 128; ++reg)
      for (uint32_t lane = 0; lane < wave->wf_size(); ++lane)
        words.push_back(cu->read_vgpr(wave->vgpr_alloc().base + reg, lane));
    return words;
  }

  amdgpu::GpuMemory memory{"matrix_error_memory"};
  amdgpu::L2Cache l2{"matrix_error_l2"};
  std::unique_ptr<amdgpu::ComputeUnitCore> cu;
  amdgpu::Wavefront *wave = nullptr;
};

TEST_F(MatrixExecutionErrorTest, PackedConversionsRejectRequiredSourcesBeforeWriting) {
  ASSERT_NO_FATAL_FAILURE(initialize(ROCJITSU_CODE_ARCH_CDNA4));
  ScopedIsaExecutionBackend backend(&cdna4::execution_backend());
  const auto sr_words = cdna4::build_vop3(cdna4::kVCvtScalef32SrPkFp4F32Vop3,
                                          {.vdst = 64, .src0 = 128, .src1 = 128, .src2 = 128});
  // Bypass decode to exercise the defensive execution guard for a VGPR-only source.
  cdna4::VCvtScalef32SrPkFp4F32Vop3 sr(sr_words.data());
  sr.src0.opr_type_ = cdna4::OperandType::OPR_SRC_SIMPLE;
  expect_rejected(sr);
  const auto words = cdna4::build_vop3(cdna4::kVCvtScalef322xpk16Fp6F32Vop3,
                                       {.vdst = 64, .src0 = 256, .src1 = 288, .src2 = 128});
  cdna4::VCvtScalef322xpk16Fp6F32Vop3 wide(words.data());
  for (auto *operand : {&wide.src0, &wide.src1}) {
    const auto original = operand->encoding_value_;
    const auto type = operand->opr_type_;
    operand->opr_type_ = cdna4::OperandType::OPR_SRC_SIMPLE;
    operand->encoding_value_ = 128;
    expect_rejected(wide);
    operand->encoding_value_ = original;
    operand->opr_type_ = type;
  }
  EXPECT_TRUE(cu->execute_instruction(&wide, *wave).succeeded());
  EXPECT_EQ(wave->instruction_execution_error(), amdgpu::InstructionExecutionError::None);
}

TEST_F(MatrixExecutionErrorTest, SparseMatrixRejectsRequiredOperandsBeforeWriting) {
  ASSERT_NO_FATAL_FAILURE(initialize(ROCJITSU_CODE_ARCH_CDNA5));
  ScopedIsaExecutionBackend backend(&cdna5::execution_backend());
  const auto words = cdna5::build_vop3p(cdna5::kVSwmmacF3216x16x64F16Vop3p,
                                        {.vdst = 64, .src0 = 256, .src1 = 288, .src2 = 304});
  cdna5::VSwmmacF3216x16x64F16Vop3p instruction(words.data());
  for (auto *operand :
       {&instruction.vdst, &instruction.src0, &instruction.src1, &instruction.src2}) {
    const auto type = operand->opr_type_;
    const auto encoding = operand->encoding_value_;
    operand->opr_type_ = cdna5::OperandType::OPR_SRC_SIMPLE;
    operand->encoding_value_ = 128;
    expect_rejected(instruction);
    operand->opr_type_ = type;
    operand->encoding_value_ = encoding;
  }
  EXPECT_TRUE(cu->execute_instruction(&instruction, *wave).succeeded());
  EXPECT_EQ(wave->instruction_execution_error(), amdgpu::InstructionExecutionError::None);
}

TEST_F(MatrixExecutionErrorTest, MfmaFormatsFailWithoutCommittingResults) {
  ASSERT_NO_FATAL_FAILURE(initialize(ROCJITSU_CODE_ARCH_CDNA4));
  ScopedIsaExecutionBackend backend(&cdna4::execution_backend());
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA4);
  for (const auto formats : {std::array{5u, 4u}, std::array{4u, 7u}}) {
    SCOPED_TRACE(::testing::PrintToString(formats));
    const auto dense_words =
        cdna4::build_vop3p_mfma(45, {.vdst = 64,
                                     .cbsz = static_cast<uint8_t>(formats[0]),
                                     .src0 = 256,
                                     .src1 = 288,
                                     .src2 = 128,
                                     .blgp = static_cast<uint8_t>(formats[1])});
    EXPECT_TRUE(decoder->decode(dense_words.data()).failed());
    cdna4::VMfmaF3216x16x128F8f6f4Vop3pMfma dense(dense_words.data());
    expect_rejected(dense);
    const auto scaled_words = mma_test::make_cdna4_mfma_scale_words(
        45, 1, 240, 240, 0, 0, formats[0], formats[1], 64, 256, 288, 128);
    EXPECT_TRUE(decoder->decode(scaled_words.data()).failed());
    cdna4::VMfmaScaleF3216x16x128F8f6f4Vop3px2 scaled(scaled_words.data());
    expect_rejected(scaled);
  }
}

TEST_F(MatrixExecutionErrorTest, WmmaFormatsAndScaleFormatsFailBeforeWriting) {
  ASSERT_NO_FATAL_FAILURE(initialize(ROCJITSU_CODE_ARCH_CDNA5));
  ScopedIsaExecutionBackend backend(&cdna5::execution_backend());
  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_CDNA5);
  for (const auto formats : {std::array{5u, 4u}, std::array{4u, 7u}}) {
    SCOPED_TRACE(::testing::PrintToString(formats));
    const auto words = cdna5::build_vop3p(cdna5::kVWmmaF3216x16x128F8f6f4Vop3p,
                                          {.vdst = 64,
                                           .opsel = static_cast<uint8_t>(formats[0]),
                                           .opsel_hi_2 = static_cast<uint8_t>(formats[1] >> 2),
                                           .src0 = 256,
                                           .src1 = 288,
                                           .src2 = 128,
                                           .opsel_hi = static_cast<uint8_t>(formats[1] & 3u)});
    cdna5::VWmmaF3216x16x128F8f6f4Vop3p dense(words.data());
    expect_rejected(dense);
    const auto scaled_words =
        mma_test::make_cdna5_wmma_scale_words(formats[0], formats[1], 128, 128, 64, 256, 288, 128);
    EXPECT_TRUE(decoder->decode(scaled_words.data()).failed());
    cdna5::VWmmaScaleF32Vop3px2 scaled(scaled_words.data());
    expect_rejected(scaled);
  }
  for (const auto formats : {std::array{4u, 4u}, std::array{0u, 4u}}) {
    auto words =
        mma_test::make_cdna5_wmma_scale_words(formats[0], formats[1], 128, 128, 64, 256, 288, 128);
    cdna5::VWmmaScaleF32Vop3px2 valid(words.data());
    EXPECT_TRUE(cu->execute_instruction(&valid, *wave).succeeded());
    for (const auto shift : {29u, 8u}) {
      auto invalid = words;
      invalid[shift == 29u ? 1 : 0] |= 3u << shift;
      cdna5::VWmmaScaleF32Vop3px2 instruction(invalid.data());
      expect_rejected(instruction);
    }
  }
  for (const auto formats :
       {std::array{4u, 4u, 1u, 2u}, std::array{0u, 4u, 1u, 0u}, std::array{4u, 0u, 0u, 2u}}) {
    SCOPED_TRACE(::testing::PrintToString(formats));
    auto words =
        mma_test::make_cdna5_wmma_scale_words(formats[0], formats[1], 128, 128, 64, 256, 288, 128);
    words[1] |= formats[2] << 29u;
    words[0] |= formats[3] << 8u;
    EXPECT_TRUE(decoder->decode(words.data()).failed());
    cdna5::VWmmaScaleF32Vop3px2 instruction(words.data());
    expect_rejected(instruction);
  }
}

} // namespace
