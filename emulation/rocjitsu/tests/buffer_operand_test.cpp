// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file buffer_operand_test.cpp
/// @brief Regression tests for semantic legacy buffer operands.

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/operand.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <memory>
#include <string>

namespace {

using namespace rocjitsu;

struct BufferOperandCase {
  rj_code_arch_t arch;
  const char *label;
  uint32_t encoding;
  const char *mnemonic;
  bool address_mode_in_hi_word;
};

class BufferOperandTest : public ::testing::TestWithParam<BufferOperandCase> {};

TEST_P(BufferOperandTest, AddressModeAndResourceUsePhysicalRegisters) {
  const auto &tc = GetParam();
  auto decoder = Decoder::create(tc.arch);
  ASSERT_NE(decoder, nullptr);

  struct AddressMode {
    uint32_t fields;
    uint8_t width;
  };
  constexpr std::array<AddressMode, 4> kModes = {AddressMode{0, 0}, AddressMode{1, 1},
                                                 AddressMode{2, 1}, AddressMode{3, 2}};

  for (const auto &mode : kModes) {
    // VADDR=63, VDATA=68, raw SRSRC=1 (descriptor s[4:7]), SOFFSET=128 (zero).
    const uint32_t words[] = {tc.encoding | (tc.address_mode_in_hi_word ? 0u : mode.fields << 12),
                              0x8001443fu | (tc.address_mode_in_hi_word ? mode.fields << 22 : 0u)};
    std::unique_ptr<Instruction> inst(decoder->decode(words));
    ASSERT_NE(inst, nullptr) << tc.label;
    ASSERT_EQ(inst->mnemonic(), tc.mnemonic) << tc.label;
    ASSERT_GE(inst->num_src_operands(), 2) << tc.label;

    const auto vaddr = inst->src_operand(0)->to_register_ref();
    if (mode.width == 0) {
      EXPECT_FALSE(vaddr.has_value()) << tc.label;
    } else {
      ASSERT_TRUE(vaddr.has_value()) << tc.label;
      EXPECT_EQ(*vaddr, (RegisterRef{RegClass::VGPR, 63, mode.width})) << tc.label;
    }

    const auto srsrc = inst->src_operand(1)->to_register_ref();
    ASSERT_TRUE(srsrc.has_value()) << tc.label;
    EXPECT_EQ(*srsrc, (RegisterRef{RegClass::SGPR, 4, 4})) << tc.label;

    const auto vdata = inst->dst_operand(0)->to_register_ref();
    ASSERT_TRUE(vdata.has_value()) << tc.label;
    EXPECT_EQ(*vdata, (RegisterRef{RegClass::VGPR, 68, 1})) << tc.label;

    if (mode.width == 1) {
      const std::string disasm = inst->disassemble();
      EXPECT_NE(disasm.find("v63"), std::string::npos) << disasm;
      EXPECT_EQ(disasm.find("v[63:64]"), std::string::npos) << disasm;
      EXPECT_NE(disasm.find("s[4:7]"), std::string::npos) << disasm;
      EXPECT_EQ(disasm.find("s[1:4]"), std::string::npos) << disasm;
    }
  }

  const bool has_acc = tc.arch == ROCJITSU_CODE_ARCH_CDNA2 || tc.arch == ROCJITSU_CODE_ARCH_CDNA3 ||
                       tc.arch == ROCJITSU_CODE_ARCH_CDNA4;
  if (has_acc) {
    // Word 1 bit 23 selects the AccVGPR bank for VDATA on CDNA2+. Keep both
    // address components enabled to cover the acc fold together with VADDR
    // width and SRSRC scaling.
    const uint32_t words[] = {tc.encoding | (3u << 12), 0x8081443fu};
    std::unique_ptr<Instruction> inst(decoder->decode(words));
    ASSERT_NE(inst, nullptr) << tc.label;

    const auto vaddr = inst->src_operand(0)->to_register_ref();
    ASSERT_TRUE(vaddr.has_value()) << tc.label;
    EXPECT_EQ(*vaddr, (RegisterRef{RegClass::VGPR, 63, 2})) << tc.label;

    const auto srsrc = inst->src_operand(1)->to_register_ref();
    ASSERT_TRUE(srsrc.has_value()) << tc.label;
    EXPECT_EQ(*srsrc, (RegisterRef{RegClass::SGPR, 4, 4})) << tc.label;

    const auto vdata = inst->dst_operand(0)->to_register_ref();
    ASSERT_TRUE(vdata.has_value()) << tc.label;
    EXPECT_EQ(*vdata, (RegisterRef{RegClass::ACC_VGPR, 68, 1})) << tc.label;
  }
}

INSTANTIATE_TEST_SUITE_P(
    LegacyBufferFamilies, BufferOperandTest,
    ::testing::Values(BufferOperandCase{ROCJITSU_CODE_ARCH_CDNA1, "cdna1_mubuf", 0xE0500000u,
                                        "buffer_load_dword", false},
                      BufferOperandCase{ROCJITSU_CODE_ARCH_CDNA2, "cdna2_mubuf", 0xE0500000u,
                                        "buffer_load_dword", false},
                      BufferOperandCase{ROCJITSU_CODE_ARCH_CDNA3, "cdna3_mubuf", 0xE0500000u,
                                        "buffer_load_dword", false},
                      BufferOperandCase{ROCJITSU_CODE_ARCH_CDNA4, "cdna4_mubuf", 0xE0500000u,
                                        "buffer_load_dword", false},
                      BufferOperandCase{ROCJITSU_CODE_ARCH_RDNA1, "rdna1_mubuf", 0xE0300000u,
                                        "buffer_load_dword", false},
                      BufferOperandCase{ROCJITSU_CODE_ARCH_RDNA2, "rdna2_mubuf", 0xE0300000u,
                                        "buffer_load_dword", false},
                      BufferOperandCase{ROCJITSU_CODE_ARCH_RDNA3, "rdna3_mubuf", 0xE0500000u,
                                        "buffer_load_b32", true},
                      BufferOperandCase{ROCJITSU_CODE_ARCH_RDNA3_5, "rdna3_5_mubuf", 0xE0500000u,
                                        "buffer_load_b32", true},
                      BufferOperandCase{ROCJITSU_CODE_ARCH_CDNA1, "cdna1_mtbuf", 0xE8000000u,
                                        "tbuffer_load_format_x", false},
                      BufferOperandCase{ROCJITSU_CODE_ARCH_CDNA2, "cdna2_mtbuf", 0xE8000000u,
                                        "tbuffer_load_format_x", false},
                      BufferOperandCase{ROCJITSU_CODE_ARCH_CDNA3, "cdna3_mtbuf", 0xE8000000u,
                                        "tbuffer_load_format_x", false},
                      BufferOperandCase{ROCJITSU_CODE_ARCH_CDNA4, "cdna4_mtbuf", 0xE8000000u,
                                        "tbuffer_load_format_x", false},
                      BufferOperandCase{ROCJITSU_CODE_ARCH_RDNA1, "rdna1_mtbuf", 0xE8000000u,
                                        "tbuffer_load_format_x", false},
                      BufferOperandCase{ROCJITSU_CODE_ARCH_RDNA2, "rdna2_mtbuf", 0xE8000000u,
                                        "tbuffer_load_format_x", false},
                      BufferOperandCase{ROCJITSU_CODE_ARCH_RDNA3, "rdna3_mtbuf", 0xE8000000u,
                                        "tbuffer_load_format_x", true},
                      BufferOperandCase{ROCJITSU_CODE_ARCH_RDNA3_5, "rdna3_5_mtbuf", 0xE8000000u,
                                        "tbuffer_load_format_x", true}),
    [](const ::testing::TestParamInfo<BufferOperandCase> &info) {
      return std::string(info.param.label);
    });

} // namespace
