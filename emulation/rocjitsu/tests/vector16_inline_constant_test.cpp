// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file vector16_inline_constant_test.cpp
/// @brief Decoded 16-bit source carriers, delegated reads and destination windows.

#include "decode_test_util.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3_5/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/execution_backend.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/operand.h"
#include "rocjitsu/isa/arch/amdgpu/shared/dpp_sdwa_ops.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/gpu_memory.h"
#include "rocjitsu/vm/amdgpu/l2_cache.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <gtest/gtest.h>

#include <array>
#include <bit>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace {

using namespace rocjitsu;

template <typename Fields, auto Build>
std::array<uint32_t, 2> encode_vector16(uint16_t opcode, uint16_t source, uint8_t half) {
  Fields fields{};
  fields.vdst = 4;
  fields.src0 = source;
  fields.src1 = 128;
  fields.src2 = 128;
  if constexpr (requires { fields.opsel; })
    fields.opsel = half;
  else
    fields.op_sel = half;
  return Build(opcode, fields);
}

struct Vector16Target {
  rj_code_arch_t arch;
  const char *name;
  uint16_t add_integer;
  uint16_t add_float;
  uint16_t convert_integer;
  std::array<uint32_t, 2> (*encode)(uint16_t, uint16_t, uint8_t);
};

constexpr std::array kTargets{
    Vector16Target{ROCJITSU_CODE_ARCH_CDNA1, "Gfx908", cdna1::kVAddU16Vop3, cdna1::kVAddF16Vop3,
                   cdna1::kVCvtF16U16Vop3,
                   encode_vector16<cdna1::Vop3BuilderFields, cdna1::build_vop3>},
    Vector16Target{ROCJITSU_CODE_ARCH_CDNA2, "Gfx90a", cdna2::kVAddU16Vop3, cdna2::kVAddF16Vop3,
                   cdna2::kVCvtF16U16Vop3,
                   encode_vector16<cdna2::Vop3BuilderFields, cdna2::build_vop3>},
    Vector16Target{ROCJITSU_CODE_ARCH_CDNA3, "Gfx942", cdna3::kVAddU16Vop3, cdna3::kVAddF16Vop3,
                   cdna3::kVCvtF16U16Vop3,
                   encode_vector16<cdna3::Vop3BuilderFields, cdna3::build_vop3>},
    Vector16Target{ROCJITSU_CODE_ARCH_CDNA4, "Gfx950", cdna4::kVAddU16Vop3, cdna4::kVAddF16Vop3,
                   cdna4::kVCvtF16U16Vop3,
                   encode_vector16<cdna4::Vop3BuilderFields, cdna4::build_vop3>},
    Vector16Target{ROCJITSU_CODE_ARCH_RDNA1, "Gfx1010", rdna1::kVAddNcU16Vop3, rdna1::kVAddF16Vop3,
                   rdna1::kVCvtF16U16Vop3,
                   encode_vector16<rdna1::Vop3BuilderFields, rdna1::build_vop3>},
    Vector16Target{ROCJITSU_CODE_ARCH_RDNA2, "Gfx1030", rdna2::kVAddNcU16Vop3, rdna2::kVAddF16Vop3,
                   rdna2::kVCvtF16U16Vop3,
                   encode_vector16<rdna2::Vop3BuilderFields, rdna2::build_vop3>},
    Vector16Target{ROCJITSU_CODE_ARCH_RDNA3, "Gfx1100", rdna3::kVAddNcU16Vop3, rdna3::kVAddF16Vop3,
                   rdna3::kVCvtF16U16Vop3,
                   encode_vector16<rdna3::Vop3BuilderFields, rdna3::build_vop3>},
    Vector16Target{ROCJITSU_CODE_ARCH_RDNA3_5, "Gfx1150", rdna3_5::kVAddNcU16Vop3,
                   rdna3_5::kVAddF16Vop3, rdna3_5::kVCvtF16U16Vop3,
                   encode_vector16<rdna3_5::Vop3BuilderFields, rdna3_5::build_vop3>},
    Vector16Target{ROCJITSU_CODE_ARCH_RDNA4, "Gfx1201", rdna4::kVAddNcU16Vop3, rdna4::kVAddF16Vop3,
                   rdna4::kVCvtF16U16Vop3,
                   encode_vector16<rdna4::Vop3BuilderFields, rdna4::build_vop3>},
    Vector16Target{ROCJITSU_CODE_ARCH_CDNA5, "Gfx1250", cdna5::kVAddNcU16Vop3, cdna5::kVAddF16Vop3,
                   cdna5::kVCvtF16U16Vop3,
                   encode_vector16<cdna5::Vop3BuilderFields, cdna5::build_vop3>},
};

class Vector16InlineConstantTest : public ::testing::TestWithParam<Vector16Target> {
protected:
  void SetUp() override {
    amdgpu::ComputeUnitCore::Config config{};
    config.arch = GetParam().arch;
    config.num_wf_slots = 1;
    config.sgprs_per_wf = 102;
    config.vgprs_per_wf = 16;
    config.lds_size_kb = 64;
    cu_ = amdgpu::ComputeUnitCore::create("vector16_inline", config, &memory_, &cache_);
    decoder_ = Decoder::create(config.arch);
    ASSERT_NE(cu_, nullptr);
    ASSERT_NE(decoder_, nullptr);
    wave_ = cu_->dispatch_wf(0, 0, config.sgprs_per_wf, config.vgprs_per_wf);
    ASSERT_NE(wave_, nullptr);
    wave_->set_exec(kExec);
  }

  void TearDown() override {
    if (wave_)
      wave_->halt();
  }

  bool legacy_cdna() const {
    return GetParam().arch == ROCJITSU_CODE_ARCH_CDNA1 ||
           GetParam().arch == ROCJITSU_CODE_ARCH_CDNA2 ||
           GetParam().arch == ROCJITSU_CODE_ARCH_CDNA3 ||
           GetParam().arch == ROCJITSU_CODE_ARCH_CDNA4;
  }

  bool legacy_rdna() const {
    return GetParam().arch == ROCJITSU_CODE_ARCH_RDNA1 ||
           GetParam().arch == ROCJITSU_CODE_ARCH_RDNA2;
  }

  std::unique_ptr<Instruction> decode(uint16_t opcode, uint16_t source, uint8_t half = 0,
                                      std::optional<uint32_t> literal = std::nullopt) {
    const std::array<uint32_t, 2> encoded = GetParam().encode(opcode, source, half);
    const std::array<uint32_t, 3> words{encoded[0], encoded[1], literal.value_or(0)};
    return std::unique_ptr<Instruction>(decode_valid(*decoder_, words.data()));
  }

  void expect_reads(const Operand &source, uint32_t expected, uint32_t mask = UINT32_MAX) {
    amdgpu::RegisterAccess regs(*wave_);
    EXPECT_EQ(regs.read_lane(source, 0) & mask, expected & mask);
    std::array<uint32_t, 3> chunk{};
    regs.read_chunk(source, 0, chunk.size(), chunk.data());
    for (uint32_t value : chunk)
      EXPECT_EQ(value & mask, expected & mask);
  }

  void expect_raw_scalar(const Operand &source, uint32_t expected) {
    amdgpu::RegisterAccess regs(*wave_);
    EXPECT_EQ(regs.read_scalar(source), expected);
    ASSERT_TRUE(source.const_value().has_value());
    EXPECT_EQ(*source.const_value(), expected);
  }

  void expect_word_result(Instruction &instruction, uint32_t expected) {
    const uint64_t exec = wave_->exec();
    const uint32_t destination = wave_->vgpr_alloc().base + 4;
    for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane) {
      cu_->write_vgpr(destination, lane, 0xDEADBEEFu);
      cu_->write_vgpr(destination + 1, lane, 0xA5A50000u | lane);
    }
    ASSERT_TRUE(cu_->execute_instruction(&instruction, *wave_).succeeded());
    for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane) {
      EXPECT_EQ(cu_->read_vgpr(destination, lane),
                (exec & (uint64_t{1} << lane)) ? expected : 0xDEADBEEFu)
          << "lane=" << lane;
      EXPECT_EQ(cu_->read_vgpr(destination + 1, lane), 0xA5A50000u | lane);
    }
    EXPECT_EQ(wave_->exec(), exec);
  }

  void expect_result(Instruction &instruction, uint16_t expected) {
    const uint64_t exec = wave_->exec();
    const uint32_t destination = wave_->vgpr_alloc().base + 4;
    auto canary = [](uint32_t lane) { return 0xDEAD0000u | (0xB000u + lane); };
    for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane)
      cu_->write_vgpr(destination, lane, canary(lane));
    ASSERT_TRUE(cu_->execute_instruction(&instruction, *wave_).succeeded());
    for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane) {
      const uint32_t expected_word =
          (exec & (uint64_t{1} << lane))
              ? (legacy_cdna() ? 0u : canary(lane) & 0xffff0000u) | expected
              : canary(lane);
      EXPECT_EQ(cu_->read_vgpr(destination, lane), expected_word) << "lane=" << lane;
    }
    EXPECT_EQ(wave_->exec(), exec);
  }

  static constexpr uint64_t kExec = (uint64_t{1} << 0) | (uint64_t{1} << 2) | (uint64_t{1} << 17);
  amdgpu::GpuMemory memory_{"vector16_inline_memory"};
  amdgpu::L2Cache cache_{"vector16_inline_cache"};
  std::unique_ptr<amdgpu::ComputeUnitCore> cu_;
  std::unique_ptr<Decoder> decoder_;
  amdgpu::Wavefront *wave_ = nullptr;
};

// gfx1100 and gfx1201 hardware agree with LLVM's INT16 source policy: float
// inlines retain FP32 bits for integer/bit consumers. Older targets follow the
// same LLVM operand contract without physical qualification here.
TEST_P(Vector16InlineConstantTest, IntegerAndFloatInputsUseTheirOwnInlineFormats) {
  struct InlineCase {
    uint16_t selector;
    uint32_t fp32;
    uint16_t fp16;
  };
  constexpr std::array cases{
      InlineCase{240, 0x3F000000u, 0x3800}, InlineCase{242, 0x3F800000u, 0x3C00},
      InlineCase{244, 0x40000000u, 0x4000}, InlineCase{248, 0x3E22F983u, 0x3118}};
  for (const auto &test : cases) {
    SCOPED_TRACE(test.selector);
    std::unique_ptr<Instruction> integer = decode(GetParam().add_integer, test.selector);
    std::unique_ptr<Instruction> numeric = decode(GetParam().add_float, test.selector);
    ASSERT_NE(integer, nullptr);
    ASSERT_NE(numeric, nullptr);
    ASSERT_NE(integer->src_operand(0), nullptr);
    ASSERT_NE(numeric->src_operand(0), nullptr);
    EXPECT_EQ(integer->src_operand(0)->lane_inline_float_format(),
              Operand::LaneInlineFloatFormat::F32);
    EXPECT_EQ(numeric->src_operand(0)->lane_inline_float_format(),
              Operand::LaneInlineFloatFormat::F16);
    EXPECT_EQ(integer->dst_operand(0)->preserve_unwritten_bytes(), legacy_rdna());
    EXPECT_EQ(numeric->dst_operand(0)->preserve_unwritten_bytes(), legacy_rdna());
    expect_reads(*integer->src_operand(0), test.fp32);
    expect_reads(*numeric->src_operand(0), test.fp16);
    expect_raw_scalar(*integer->src_operand(0), test.fp32);
    expect_raw_scalar(*numeric->src_operand(0), test.fp32);
    expect_result(*integer, static_cast<uint16_t>(test.fp32));
    expect_result(*numeric, test.fp16);
  }
}

TEST_P(Vector16InlineConstantTest, IntegerToFloatConversionKeepsIntegerSourceFormat) {
  std::unique_ptr<Instruction> conversion = decode(GetParam().convert_integer, 242);
  ASSERT_NE(conversion, nullptr);
  expect_reads(*conversion->src_operand(0), 0x3F800000u);
  expect_result(*conversion, 0);
}

TEST_P(Vector16InlineConstantTest, RegisterAndLiteralCarriersKeepTheirStoredBits) {
  for (const uint16_t opcode : {GetParam().add_integer, GetParam().add_float}) {
    SCOPED_TRACE(opcode);
    cu_->write_sgpr(wave_->sgpr_alloc().base, 0x40003C00u);
    std::unique_ptr<Instruction> reg = decode(opcode, 0);
    ASSERT_NE(reg, nullptr);
    expect_reads(*reg->src_operand(0), 0x40003C00u);
    expect_result(*reg, 0x3C00u);
    if (!legacy_rdna()) {
      std::unique_ptr<Instruction> high = decode(opcode, 0, 1);
      ASSERT_NE(high, nullptr);
      expect_result(*high, 0x4000u);
    }
    if (legacy_cdna())
      continue; // GFX9 VOP3 does not encode an appended literal.
    for (uint32_t literal : {0x3F800000u, 0x40003C00u}) {
      SCOPED_TRACE(literal);
      std::unique_ptr<Instruction> instruction = decode(opcode, 255, 0, literal);
      ASSERT_NE(instruction, nullptr);
      EXPECT_EQ(instruction->size(), 12u);
      expect_reads(*instruction->src_operand(0), literal, legacy_rdna() ? 0xffffu : UINT32_MAX);
      expect_result(*instruction, static_cast<uint16_t>(literal));
    }
  }
}

TEST_P(Vector16InlineConstantTest, DelegatesKeepStagedLaneValues) {
  for (const uint16_t opcode : {GetParam().add_integer, GetParam().add_float}) {
    std::unique_ptr<Instruction> instruction = decode(opcode, 242);
    ASSERT_NE(instruction, nullptr);
    Operand &source = *const_cast<Operand *>(instruction->src_operand(0));
    const bool integer = opcode == GetParam().add_integer;
    std::optional<StagedOperand> staged;
    amdgpu::sdwa::stage_source(source, amdgpu::sdwa::WORD_0, false, false, false,
                               amdgpu::sdwa::SourceModifierFormat::NONE, staged, *wave_);
    ASSERT_TRUE(staged.has_value());
    {
      ScopedOperandDelegate delegate(source, &*staged);
      amdgpu::RegisterAccess regs(*wave_);
      EXPECT_EQ(regs.read_lane(source, 0), integer ? 0u : 0x3C00u);
      std::array<uint32_t, 2> values{};
      regs.read_chunk(source, 0, values.size(), values.data());
      EXPECT_EQ(values[0], integer ? 0u : 0x3C00u);
      EXPECT_EQ(values[1], 0u);
    }
  }
}

TEST_P(Vector16InlineConstantTest, HighHalfSelectionUsesTheFormattedCarrier) {
  // CDNA1-4 retain the model choice from the manual's generic VOP3A OPSEL
  // description. LLVM rejects OPSEL on these ADD16 forms; the disagreement
  // has no physical qualification here. gfx1100/gfx1201 controls are separate.
  if (legacy_rdna())
    GTEST_SKIP() << "RDNA1/2 numeric ADD has no OPSEL form; integer ADD OPSEL is not modeled.";
  for (const auto &[selector, high] : {std::pair{242u, 0x3F80u}, std::pair{248u, 0x3E22u}}) {
    SCOPED_TRACE(selector);
    std::unique_ptr<Instruction> integer = decode(GetParam().add_integer, selector, 1);
    std::unique_ptr<Instruction> numeric = decode(GetParam().add_float, selector, 1);
    ASSERT_NE(integer, nullptr);
    ASSERT_NE(numeric, nullptr);
    expect_result(*integer, high);
    expect_result(*numeric, 0u);
  }
}

class PackedSatDestinationTest : public Vector16InlineConstantTest {};

TEST_P(PackedSatDestinationTest, Vop1PackedResultClearsTheWholeDestinationWord) {
  const uint16_t opcode = GetParam().arch == ROCJITSU_CODE_ARCH_RDNA1 ? rdna1::kVSatPkU8I16Vop1
                                                                      : rdna2::kVSatPkU8I16Vop1;
  const std::array<uint32_t, 1> encoded = GetParam().arch == ROCJITSU_CODE_ARCH_RDNA1
                                              ? rdna1::build_vop1(opcode, {.src0 = 0, .vdst = 4})
                                              : rdna2::build_vop1(opcode, {.src0 = 0, .vdst = 4});
  const std::array<uint32_t, 2> words{encoded[0], 0};
  std::unique_ptr<Instruction> instruction(decode_valid(*decoder_, words.data()));
  ASSERT_NE(instruction, nullptr);
  ASSERT_EQ(instruction->size(), 4u);
  ASSERT_NE(instruction->dst_operand(0), nullptr);
  EXPECT_FALSE(instruction->dst_operand(0)->preserve_unwritten_bytes());
  cu_->write_sgpr(wave_->sgpr_alloc().base, 0xFFFF012Cu);
  expect_word_result(*instruction, 0x000000FFu);
}

TEST_P(PackedSatDestinationTest, Vop3PackedResultClearsTheWholeDestinationWord) {
  const uint16_t opcode = GetParam().arch == ROCJITSU_CODE_ARCH_RDNA1 ? rdna1::kVSatPkU8I16Vop3
                                                                      : rdna2::kVSatPkU8I16Vop3;
  std::unique_ptr<Instruction> instruction = decode(opcode, 0);
  ASSERT_NE(instruction, nullptr);
  ASSERT_EQ(instruction->size(), 8u);
  ASSERT_NE(instruction->dst_operand(0), nullptr);
  EXPECT_FALSE(instruction->dst_operand(0)->preserve_unwritten_bytes());
  cu_->write_sgpr(wave_->sgpr_alloc().base, 0xFFFF012Cu);
  expect_word_result(*instruction, 0x000000FFu);
}

class Bf16InlineConstantTest : public Vector16InlineConstantTest {
protected:
  bool dot() const {
    return GetParam().arch == ROCJITSU_CODE_ARCH_RDNA3 ||
           GetParam().arch == ROCJITSU_CODE_ARCH_RDNA4;
  }

  bool fp32_carrier() const { return GetParam().arch == ROCJITSU_CODE_ARCH_CDNA5; }

  std::unique_ptr<Instruction> numeric(uint16_t source,
                                       std::optional<uint32_t> literal = std::nullopt) {
    if (!dot()) {
      const uint16_t opcode = fp32_carrier() ? cdna5::kVCvtF32Bf16Vop3 : cdna4::kVCvtF32Bf16Vop3;
      return decode(opcode, source, fp32_carrier() ? 1 : 0, literal);
    }
    std::array<uint32_t, 2> encoded{};
    if (GetParam().arch == ROCJITSU_CODE_ARCH_RDNA3) {
      encoded = rdna3::build_vop3(rdna3::kVDot2Bf16Bf16Vop3,
                                  {.vdst = 4, .src0 = 128, .src1 = 128, .src2 = source});
    } else {
      encoded = rdna4::build_vop3(rdna4::kVDot2Bf16Bf16Vop3,
                                  {.vdst = 4, .src0 = 128, .src1 = 128, .src2 = source});
    }
    const std::array<uint32_t, 3> words{encoded[0], encoded[1], literal.value_or(0)};
    return std::unique_ptr<Instruction>(decode_valid(*decoder_, words.data()));
  }

  const Operand &numeric_source(const Instruction &instruction) const {
    return *instruction.src_operand(dot() ? 2 : 0);
  }

  void expect_numeric_result(Instruction &instruction, uint32_t fp32, uint16_t bf16) {
    if (dot()) {
      expect_result(instruction, bf16);
    } else {
      expect_word_result(instruction, fp32);
    }
  }
};

TEST_P(Bf16InlineConstantTest, NumericInlineSourceKeepsLaneChunkAndRawScalarContracts) {
  // RDNA3/4 DOT accumulator controls are measured on the local cards. The
  // CDNA4 low and CDNA5 high conversion forms remain source-backed checks.
  struct InlineCase {
    uint16_t selector;
    uint32_t fp32;
    uint16_t bf16;
  };
  constexpr std::array cases{
      InlineCase{240, 0x3F000000u, 0x3F00}, InlineCase{241, 0xBF000000u, 0xBF00},
      InlineCase{242, 0x3F800000u, 0x3F80}, InlineCase{243, 0xBF800000u, 0xBF80},
      InlineCase{244, 0x40000000u, 0x4000}, InlineCase{245, 0xC0000000u, 0xC000},
      InlineCase{246, 0x40800000u, 0x4080}, InlineCase{247, 0xC0800000u, 0xC080},
      InlineCase{248, 0x3E22F983u, 0x3E22}};
  for (const auto &test : cases) {
    SCOPED_TRACE(test.selector);
    std::unique_ptr<Instruction> instruction = numeric(test.selector);
    ASSERT_NE(instruction, nullptr);
    const Operand &source = numeric_source(*instruction);
    EXPECT_EQ(source.lane_inline_float_format(), fp32_carrier()
                                                     ? Operand::LaneInlineFloatFormat::F32
                                                     : Operand::LaneInlineFloatFormat::BF16);
    expect_reads(source, fp32_carrier() ? test.fp32 : test.bf16);
    expect_raw_scalar(source, test.fp32);
    // BF16's 1/(2*pi) entry is 0x3e22, not nearest-rounded 0x3e23.
    expect_numeric_result(*instruction, uint32_t{test.bf16} << 16, test.bf16);
  }
}

TEST_P(Bf16InlineConstantTest, Cdna5Vop1AndVop3SelectRawInlineCarrierHalves) {
  if (!fp32_carrier())
    GTEST_SKIP() << "CDNA5 raw BF16 carrier policy";

  // CDNA5 manual sections 4.1 and 7.2.2 disagree on no-OPSEL inline inputs.
  // Follow LLVM's raw low-half VOP1 interpretation until hardware qualifies it;
  // older assembler acceptance alone does not resolve that disagreement.
  constexpr std::array<uint32_t, 9> carriers{0x3F000000u, 0xBF000000u, 0x3F800000u,
                                             0xBF800000u, 0x40000000u, 0xC0000000u,
                                             0x40800000u, 0xC0800000u, 0x3E22F983u};
  const uint64_t full_exec =
      wave_->wf_size() == 64 ? UINT64_MAX : (uint64_t{1} << wave_->wf_size()) - 1;
  for (uint64_t exec : {uint64_t{1}, full_exec}) {
    SCOPED_TRACE(exec);
    wave_->set_exec(exec);
    for (uint16_t source = 240; source <= 248; ++source) {
      SCOPED_TRACE(source);
      const std::array<uint32_t, 1> words =
          cdna5::build_vop1(cdna5::kVCvtF32Bf16Vop1, {.src0 = source, .vdst = 4});
      std::unique_ptr<Instruction> vop1(decode_valid(*decoder_, words.data()));
      ASSERT_NE(vop1, nullptr);
      const uint32_t carrier = carriers[source - 240];
      expect_reads(*vop1->src_operand(0), carrier);
      expect_raw_scalar(*vop1->src_operand(0), carrier);
      expect_word_result(*vop1, carrier << 16);
      for (uint8_t half : {uint8_t{0}, uint8_t{1}}) {
        SCOPED_TRACE(half);
        std::unique_ptr<Instruction> vop3 = decode(cdna5::kVCvtF32Bf16Vop3, source, half);
        ASSERT_NE(vop3, nullptr);
        expect_word_result(*vop3, half ? carrier & 0xFFFF0000u : carrier << 16);
      }
    }
    // The low half of F32 inline 1.0 is BF16 zero, so VOP1 reciprocal is infinity.
    const std::array<uint32_t, 1> words =
        cdna5::build_vop1(cdna5::kVRcpBf16Vop1, {.src0 = 242, .vdst = 4});
    std::unique_ptr<Instruction> reciprocal(decode_valid(*decoder_, words.data()));
    ASSERT_NE(reciprocal, nullptr);
    expect_result(*reciprocal, 0x7F80u);
  }
}

TEST_P(Bf16InlineConstantTest, RegisterAndLegalLiteralSourcesRetainTheirCarriers) {
  const uint32_t carrier = fp32_carrier() ? 0x3F800000u : 0x00003F80u;
  cu_->write_sgpr(wave_->sgpr_alloc().base, carrier);
  std::unique_ptr<Instruction> reg = numeric(0);
  ASSERT_NE(reg, nullptr);
  expect_reads(numeric_source(*reg), carrier);
  expect_numeric_result(*reg, 0x3F800000u, 0x3F80u);
  if (dot()) {
    std::unique_ptr<Instruction> literal = numeric(255, carrier);
    ASSERT_NE(literal, nullptr);
    EXPECT_EQ(literal->size(), 12u);
    expect_reads(numeric_source(*literal), carrier);
    expect_numeric_result(*literal, 0x3F800000u, 0x3F80u);
  }
}

TEST_P(Bf16InlineConstantTest, DelegatesKeepStagedValuesRaw) {
  std::unique_ptr<Instruction> instruction = numeric(242);
  ASSERT_NE(instruction, nullptr);
  Operand &source = const_cast<Operand &>(numeric_source(*instruction));
  const uint32_t carrier = fp32_carrier() ? 0x3F800000u : 0x00003F80u;
  StagedOperand staged(source, static_cast<int>(wave_->wf_size()));
  for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane) {
    staged.set_lane(lane, (kExec & (uint64_t{1} << lane)) ? carrier : 0);
  }
  ScopedOperandDelegate delegate(source, &staged);
  amdgpu::RegisterAccess regs(*wave_);
  EXPECT_EQ(regs.read_lane(source, 0), carrier);
  std::array<uint32_t, 3> values{};
  regs.read_chunk(source, 0, values.size(), values.data());
  EXPECT_EQ(values, (std::array<uint32_t, 3>{carrier, 0, carrier}));
  expect_numeric_result(*instruction, 0x3F800000u, 0x3F80u);
}

class Dot2InlineConstantTest : public Vector16InlineConstantTest {
protected:
  std::unique_ptr<Instruction> accumulator(uint16_t opcode, uint16_t source, uint8_t opsel,
                                           std::optional<uint32_t> literal = std::nullopt,
                                           bool dpp = false) {
    std::array<uint32_t, 2> encoded{};
    if (GetParam().arch == ROCJITSU_CODE_ARCH_RDNA3) {
      encoded = rdna3::build_vop3(
          opcode, {.vdst = 4, .op_sel = opsel, .src0 = 128, .src1 = 128, .src2 = source});
    } else if (GetParam().arch == ROCJITSU_CODE_ARCH_RDNA3_5) {
      encoded = rdna3_5::build_vop3(
          opcode, {.vdst = 4, .op_sel = opsel, .src0 = 128, .src1 = 128, .src2 = source});
    } else {
      encoded = rdna4::build_vop3(
          opcode, {.vdst = 4, .opsel = opsel, .src0 = 128, .src1 = 128, .src2 = source});
    }
    const std::array<uint32_t, 3> words{encoded[0], encoded[1], literal.value_or(0)};
    if (dpp) {
      auto decode_dpp = [&]<typename Raw>() {
        Raw raw = std::bit_cast<Raw>(words);
        raw.src0 = amdgpu::SRC_DPP;
        raw.vsrc0 = 1;
        raw.dpp_ctrl = amdgpu::dpp::ROW_MIRROR;
        raw.bound_ctrl = 1;
        raw.bank_mask = 0xF;
        raw.row_mask = 0xF;
        return std::unique_ptr<Instruction>(
            decode_valid(*decoder_, reinterpret_cast<const uint32_t *>(&raw)));
      };
      if (GetParam().arch == ROCJITSU_CODE_ARCH_RDNA3)
        return decode_dpp.template operator()<rdna3::Vop3VopDpp16MachineInst>();
      if (GetParam().arch == ROCJITSU_CODE_ARCH_RDNA3_5)
        return decode_dpp.template operator()<rdna3_5::Vop3VopDpp16MachineInst>();
      return decode_dpp.template operator()<rdna4::Vop3VopDpp16MachineInst>();
    }
    return std::unique_ptr<Instruction>(decode_valid(*decoder_, words.data()));
  }
};

TEST_P(Dot2InlineConstantTest, FloatAccumulatorReplicatesBeforeOpSel) {
  struct InlineCase {
    uint16_t selector;
    uint16_t f16;
    uint16_t bf16;
  };
  constexpr std::array cases{InlineCase{240, 0x3800, 0x3F00}, InlineCase{241, 0xB800, 0xBF00},
                             InlineCase{242, 0x3C00, 0x3F80}, InlineCase{243, 0xBC00, 0xBF80},
                             InlineCase{244, 0x4000, 0x4000}, InlineCase{245, 0xC000, 0xC000},
                             InlineCase{246, 0x4400, 0x4080}, InlineCase{247, 0xC400, 0xC080},
                             InlineCase{248, 0x3118, 0x3E22}};
  const uint64_t full = wave_->wf_size() == 64 ? UINT64_MAX : UINT32_MAX;
  for (bool bf16 : {false, true}) {
    for (const auto &test : cases) {
      for (uint8_t opsel : {0u, 4u}) {
        SCOPED_TRACE(::testing::Message() << "bf16=" << bf16 << " selector=" << test.selector
                                          << " opsel=" << unsigned(opsel));
        std::unique_ptr<Instruction> instruction = accumulator(
            bf16 ? rdna4::kVDot2Bf16Bf16Vop3 : rdna4::kVDot2F16F16Vop3, test.selector, opsel);
        ASSERT_NE(instruction, nullptr);
        for (uint64_t exec : {uint64_t{0}, kExec, full}) {
          wave_->set_exec(exec);
          expect_result(*instruction, bf16 ? test.bf16 : test.f16);
        }
      }
    }
  }
}

TEST_P(Dot2InlineConstantTest, AccumulatorRegistersAndLiteralsKeepIndependentHalves) {
  for (bool bf16 : {false, true}) {
    const uint16_t low = bf16 ? 0x3F80u : 0x3C00u;
    const uint32_t carrier = 0x40000000u | low;
    cu_->write_sgpr(wave_->sgpr_alloc().base, carrier);
    for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane)
      cu_->write_vgpr(wave_->vgpr_alloc().base, lane, carrier);
    for (uint16_t source : {0u, 256u, 255u}) {
      for (uint8_t opsel : {0u, 4u}) {
        SCOPED_TRACE(::testing::Message()
                     << "bf16=" << bf16 << " source=" << source << " opsel=" << unsigned(opsel));
        std::unique_ptr<Instruction> instruction = accumulator(
            bf16 ? rdna4::kVDot2Bf16Bf16Vop3 : rdna4::kVDot2F16F16Vop3, source, opsel, carrier);
        ASSERT_NE(instruction, nullptr);
        expect_result(*instruction, opsel ? 0x4000u : low);
      }
    }
  }
  // A literal whose value resembles a floating selector remains a literal.
  std::unique_ptr<Instruction> literal = accumulator(rdna4::kVDot2Bf16Bf16Vop3, 255, 4, 242);
  ASSERT_NE(literal, nullptr);
  expect_result(*literal, 0);
}

TEST_P(Dot2InlineConstantTest, IntegerAccumulatorReplicatesRawHalfBits) {
  // Inline integer 64 is the raw half pattern 0x0040, not numeric 64.0.
  // gfx1100 and gfx1201 retain this F16 accumulator subnormal. GFX11 flushes
  // the corresponding BF16 subnormal, while gfx1201 retains it.
  for (bool bf16 : {false, true}) {
    const uint16_t expected = bf16 && GetParam().arch != ROCJITSU_CODE_ARCH_RDNA4 ? 0u : 64u;
    for (uint8_t opsel : {0u, 4u}) {
      SCOPED_TRACE(::testing::Message() << "bf16=" << bf16 << " opsel=" << unsigned(opsel));
      std::unique_ptr<Instruction> instruction =
          accumulator(bf16 ? rdna4::kVDot2Bf16Bf16Vop3 : rdna4::kVDot2F16F16Vop3, 192, opsel);
      ASSERT_NE(instruction, nullptr);
      expect_reads(*instruction->src_operand(2), 64);
      expect_result(*instruction, expected);
    }
  }
}

TEST_P(Dot2InlineConstantTest, DppAccumulatorReplicatesInlineAndPreservesRegisterHalves) {
  wave_->set_exec(wave_->wf_size() == 64 ? UINT64_MAX : UINT32_MAX);
  for (uint32_t lane = 0; lane < wave_->wf_size(); ++lane)
    cu_->write_vgpr(wave_->vgpr_alloc().base + 1, lane, 0);
  for (bool bf16 : {false, true}) {
    const uint16_t low = bf16 ? 0x3F80u : 0x3C00u;
    cu_->write_sgpr(wave_->sgpr_alloc().base, 0x40000000u | low);
    for (uint16_t source : {242u, 0u}) {
      SCOPED_TRACE(::testing::Message() << "bf16=" << bf16 << " source=" << source);
      std::unique_ptr<Instruction> instruction =
          accumulator(bf16 ? rdna4::kVDot2Bf16Bf16Vop3 : rdna4::kVDot2F16F16Vop3, source, 4,
                      std::nullopt, true);
      ASSERT_NE(instruction, nullptr);
      ASSERT_EQ(instruction->size(), 12u);
      expect_result(*instruction, source == 242 ? low : 0x4000u);
    }
  }
}

TEST_P(Dot2InlineConstantTest, FmaKeepsOrdinaryHighHalfSelection) {
  std::unique_ptr<Instruction> instruction = accumulator(rdna4::kVFmaF16Vop3, 242, 4);
  ASSERT_NE(instruction, nullptr);
  expect_result(*instruction, 0);
}

INSTANTIATE_TEST_SUITE_P(Dot2Targets, Dot2InlineConstantTest,
                         ::testing::Values(kTargets[6], kTargets[7], kTargets[8]),
                         [](const auto &info) { return std::string(info.param.name); });

INSTANTIATE_TEST_SUITE_P(Targets, Vector16InlineConstantTest, ::testing::ValuesIn(kTargets),
                         [](const auto &info) { return std::string(info.param.name); });
INSTANTIATE_TEST_SUITE_P(LegacyRdna, PackedSatDestinationTest,
                         ::testing::Values(kTargets[4], kTargets[5]),
                         [](const auto &info) { return std::string(info.param.name); });
INSTANTIATE_TEST_SUITE_P(Bf16Targets, Bf16InlineConstantTest,
                         ::testing::Values(kTargets[3], kTargets[6], kTargets[8], kTargets[9]),
                         [](const auto &info) { return std::string(info.param.name); });

} // namespace
