// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file model_only_isa_test.cpp
/// @brief Link and decode smoke test for the narrow model-only ISA boundary.

#include "decode_test_util.h"
#include "rocjitsu/isa/arch/amdgpu/cdna5/isa.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/target_registry.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>

namespace rocjitsu {
namespace {

TEST(ModelOnlyIsaTest, DecodesWithoutExecutionCallback) {
  constexpr uint32_t kSNop = 0xBF800000u;
  constexpr uint32_t kVMovB32V0V1 = 0x7E000301u;

  const IsaTargetRegistry &registry = default_isa_target_registry();
  ASSERT_EQ(registry.targets().size(), 1u);
  EXPECT_EQ(registry.targets()[0].id, "cdna5");
  EXPECT_FALSE(registry.targets()[0].supports_execution);
  const IsaGpuTargetDescription *gfx1250 = registry.find_gpu_target(ROCJITSU_CODE_TARGET_GFX1250);
  const IsaGpuTargetDescription *gfx1251 = registry.find_gpu_target(ROCJITSU_CODE_TARGET_GFX1251);
  ASSERT_NE(gfx1250, nullptr);
  ASSERT_NE(gfx1251, nullptr);
  EXPECT_FALSE(gfx1250->capabilities.execution_implemented);
  EXPECT_FALSE(gfx1251->capabilities.execution_implemented);

  std::unique_ptr<Decoder> decoder = Decoder::create(registry, ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  EXPECT_EQ(Decoder::create(registry, ROCJITSU_CODE_ARCH_RDNA4), nullptr);

  std::unique_ptr<Instruction> inst(decode_valid(*decoder, &kSNop));
  ASSERT_NE(inst, nullptr);
  EXPECT_EQ(inst->mnemonic(), "s_nop");
  EXPECT_EQ(inst->execute, nullptr);

  inst.reset(decode_valid(*decoder, &kVMovB32V0V1));
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
