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

// Derive the expected mode independently so the ELF check catches a broken production gate.
#if defined(__ELF__) && defined(__x86_64__) && defined(__LP64__) &&                                \
    defined(__code_model_small__) && (defined(__GNUC__) || defined(__clang__))
#define RJ_TEST_S_NOP rj_test_expect_decode_rel32
#else
#define RJ_TEST_S_NOP rj_test_expect_decode_switch
#endif
// The decode below must read the symbol even under LTO and section garbage collection.
extern "C" const volatile uint32_t RJ_TEST_S_NOP = 0xBF800000u;

namespace rocjitsu {
namespace {

TEST(ModelOnlyIsaTest, DecodesPrimaryAliasesAndReservedOpcodes) {
  auto decoder = Decoder::create(default_isa_target_registry(), ROCJITSU_CODE_ARCH_CDNA5);
  ASSERT_NE(decoder, nullptr);
  // Three fallthrough cases and their common destination in the primary map.
  for (uint32_t encoding = 4; encoding <= 7; ++encoding) {
    const uint32_t opcode = encoding << 23;
    std::unique_ptr<Instruction> inst(decode_valid(*decoder, &opcode));
    ASSERT_NE(inst, nullptr) << encoding;
    EXPECT_EQ(inst->mnemonic(), "v_cndmask_b32_e32");
  }

  constexpr uint32_t kReservedPrimary = 0;
  constexpr uint32_t kReservedSopp = 0xBF890000u;
  EXPECT_TRUE(decode_fails(*decoder, &kReservedPrimary));
  EXPECT_TRUE(decode_fails(*decoder, &kReservedSopp));
}

TEST(ModelOnlyIsaTest, DecodesWithoutExecutionCallback) {
  const uint32_t s_nop = RJ_TEST_S_NOP;
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

  std::unique_ptr<Instruction> inst(decode_valid(*decoder, &s_nop));
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
  EXPECT_FALSE(inst->src_operand(0)->execution_backend_available());
  EXPECT_FALSE(inst->dst_operand(0)->execution_backend_available());
  EXPECT_EQ(inst->execute, nullptr);
}

} // namespace
} // namespace rocjitsu
