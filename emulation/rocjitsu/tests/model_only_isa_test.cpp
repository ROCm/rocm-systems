// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file model_only_isa_test.cpp
/// @brief Link and decode smoke test for the narrow model-only ISA boundary.

#include "rocjitsu/isa/arch/amdgpu/gfx1250/isa.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/operand.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <stdexcept>

namespace rocjitsu {

// A fixed-profile DBT frontend provides its own narrow factory instead of
// linking the all-architecture decoder registry.
std::unique_ptr<Decoder> Decoder::create(rj_code_arch_t arch) {
  if (arch != ROCJITSU_CODE_ARCH_GFX1250)
    return nullptr;
  return std::make_unique<IsaDecoder<gfx1250::Isa>>();
}

namespace {

TEST(ModelOnlyIsaTest, DecodesWithoutExecutionCallback) {
  constexpr uint32_t kSNop = 0xBF800000u;
  constexpr uint32_t kVMovB32V0V1 = 0x7E000301u;

  auto decoder = Decoder::create(ROCJITSU_CODE_ARCH_GFX1250);
  ASSERT_NE(decoder, nullptr);
  EXPECT_EQ(Decoder::create(ROCJITSU_CODE_ARCH_RDNA4), nullptr);
  EXPECT_THROW(gfx1250::Operand::require_execution_backend(), std::logic_error);

  std::unique_ptr<Instruction> inst(decoder->decode(&kSNop));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "s_nop");
  EXPECT_EQ(inst->execute, nullptr);

  inst.reset(decoder->decode(&kVMovB32V0V1));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "v_mov_b32_e32");
  ASSERT_EQ(inst->num_src_operands(), 1);
  ASSERT_EQ(inst->num_dst_operands(), 1);
  EXPECT_EQ(inst->src_operand(0)->name(), "v1");
  EXPECT_EQ(inst->dst_operand(0)->name(), "v0");
  EXPECT_EQ(inst->execute, nullptr);
}

} // namespace
} // namespace rocjitsu
