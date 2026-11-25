// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/input.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/target_registry.h"

#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <span>
#include <vector>

namespace rocjitsu::consan {
namespace {

TEST(ConSanApplicationInstruction, AdmitsOnlyTheQualifiedReplayModeWriteAndClusterRead) {
  std::unique_ptr<Decoder> decoder =
      Decoder::create(default_isa_target_registry(), target_profile("gfx1250")->public_id);
  ASSERT_TRUE(decoder);
  struct Case {
    std::array<uint32_t, 2> words;
    bool supported;
  };
  const std::array<Case, 5> cases = {{
      {.words = {0xb9800641, 1}, .supported = true},
      {.words = {0xb9800641, 0}, .supported = false},
      // A write to MODE.DST_VGPR_MSB must not inherit replay-mode admission.
      {.words = {0xb9800b01, 1}, .supported = false},
      {.words = {0xb884199c, 0}, .supported = true},
      {.words = {0xb8840001, 0}, .supported = false},
  }};
  for (const Case &test : cases) {
    SCOPED_TRACE(test.words[0]);
    const FailureOr<std::unique_ptr<Instruction>> instruction = decoder->decode_window(test.words);
    ASSERT_TRUE(instruction.succeeded());
    std::vector<NativeLdsAccess> accesses;
    InputError error;
    EXPECT_EQ(
        classify_lds_race_instruction(*instruction.value(), IsaFamily::Cdna5, 128, accesses, error)
            .succeeded(),
        test.supported)
        << error.message;
    EXPECT_TRUE(accesses.empty());
    if (test.supported) {
      EXPECT_TRUE(classify_lds_race_instruction(*instruction.value(), IsaFamily::Rdna4, 128,
                                                accesses, error)
                      .failed());
    }
  }
}

TEST(ConSanApplicationInstruction, RejectsGlobalTransfersThatModifyUnobservedLds) {
  std::unique_ptr<Decoder> decoder =
      Decoder::create(default_isa_target_registry(), target_profile("gfx1250")->public_id);
  ASSERT_TRUE(decoder);
  const std::array<uint32_t, 3> words = cdna5::build_vglobal(cdna5::kGlobalLoadAsyncToLdsB8Vglobal,
                                                             {.saddr = 124, .vdst = 0, .vaddr = 2});
  const FailureOr<std::unique_ptr<Instruction>> instruction = decoder->decode_window(words);
  ASSERT_TRUE(instruction.succeeded());
  ASSERT_TRUE(instruction.value()->transfers_to_lds());
  ASSERT_EQ(instruction.value()->amdgpu_memory_addressing(), amdgpu::MemoryAddressing::Global);
  std::vector<NativeLdsAccess> accesses;
  InputError error;
  EXPECT_TRUE(
      classify_lds_race_instruction(*instruction.value(), IsaFamily::Cdna5, 128, accesses, error)
          .failed());
  EXPECT_EQ(error.code, InputErrorCode::UnsupportedAccess);
  EXPECT_NE(error.message.find("unmodeled memory/LDS transfer"), std::string::npos);
  EXPECT_TRUE(accesses.empty());
}

TEST(ConSanApplicationInstruction, ValidatesBothOperationsOfDualIntegerInstructions) {
  std::unique_ptr<Decoder> decoder =
      Decoder::create(default_isa_target_registry(), target_profile("gfx1250")->public_id);
  ASSERT_TRUE(decoder);
  // Genuine VOPD3 address arithmetic from the original prefill. The second
  // encoding changes only Y's opcode to FMAC: qualifying X alone is insufficient.
  const std::array<uint32_t, 3> integer = {0xcf552082, 0x000000af, 0x0a400008};
  std::array<uint32_t, 3> floating_y = integer;
  floating_y[0] &= ~(0x3fu << 12);
  for (const bool supported : {true, false}) {
    const std::array<uint32_t, 3> &words = supported ? integer : floating_y;
    const FailureOr<std::unique_ptr<Instruction>> instruction = decoder->decode_window(words);
    ASSERT_TRUE(instruction.succeeded());
    ASSERT_EQ(instruction.value()->flags() & INVALID_VOPD_OPERANDS, 0u);
    EXPECT_EQ(instruction.value()->amdgpu_operation(), amdgpu::Operation::V_DUAL_LSHRREV_B32);
    EXPECT_EQ(instruction.value()->amdgpu_secondary_operation(),
              supported ? amdgpu::Operation::V_DUAL_BITOP2_B32
                        : amdgpu::Operation::V_DUAL_FMAC_F32);
    std::vector<NativeLdsAccess> accesses;
    InputError error;
    EXPECT_EQ(
        classify_lds_race_instruction(*instruction.value(), IsaFamily::Cdna5, 128, accesses, error)
            .succeeded(),
        supported)
        << error.message;
    EXPECT_TRUE(accesses.empty());
  }
}

} // namespace
} // namespace rocjitsu::consan
