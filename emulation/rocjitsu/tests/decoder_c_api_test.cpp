// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "decode_test_util.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <type_traits>

namespace {

// s_endpgm in the GFX9/CDNA SOPP encoding, which gfx1250 rejects.
constexpr rj_code_binary_inst_t kCdnaSEndpgm = 0xBF810000u;

static_assert(std::is_same_v<decltype(&rj_code_inst_destroy), void (*)(rj_code_inst_t *)>,
              "standalone destruction must not accept borrowed const instructions");
static_assert(
    std::is_same_v<decltype(rj_code_basic_block_first_inst(nullptr)), const rj_code_inst_t *>,
    "borrowed instructions must remain const-qualified");
static_assert(std::is_same_v<decltype(rj_code_inst_next(nullptr)), const rj_code_inst_t *>,
              "instruction traversal must preserve the borrowed const qualification");

TEST(DecoderCApiTest, InvalidInstructionReturnsErrorAndClearsOutput) {
  rj_code_decoder_t *decoder = nullptr;
  ASSERT_EQ(rj_code_decoder_create(ROCJITSU_CODE_ARCH_CDNA5, &decoder), ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(decoder, nullptr);

  auto *instruction = reinterpret_cast<rj_code_inst_t *>(static_cast<uintptr_t>(1));
  EXPECT_EQ(rj_code_decoder_decode(decoder, &kCdnaSEndpgm, &instruction), ROCJITSU_STATUS_ERROR);
  EXPECT_EQ(instruction, nullptr);

  rj_code_decoder_destroy(decoder);
}

TEST(DecoderCApiTest, InvalidArgumentsClearWritableOutputs) {
  auto *instruction = reinterpret_cast<rj_code_inst_t *>(static_cast<uintptr_t>(1));
  EXPECT_EQ(rj_code_decoder_decode(nullptr, &kCdnaSEndpgm, &instruction),
            ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(instruction, nullptr);

  auto *decoder = reinterpret_cast<rj_code_decoder_t *>(static_cast<uintptr_t>(1));
  EXPECT_EQ(rj_code_decoder_create(ROCJITSU_CODE_ARCH_INVALID, &decoder),
            ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(decoder, nullptr);

  decoder = reinterpret_cast<rj_code_decoder_t *>(static_cast<uintptr_t>(1));
  EXPECT_EQ(rj_code_decoder_create_for_target(nullptr, &decoder), ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(decoder, nullptr);

  decoder = reinterpret_cast<rj_code_decoder_t *>(static_cast<uintptr_t>(1));
  EXPECT_EQ(rj_code_decoder_create_for_target("", &decoder), ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(decoder, nullptr);
}

TEST(DecoderCApiTest, StandaloneInstructionsAreCallerOwned) {
  rj_code_decoder_t *decoder = nullptr;
  ASSERT_EQ(rj_code_decoder_create(ROCJITSU_CODE_ARCH_CDNA3, &decoder), ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(decoder, nullptr);

  rj_code_inst_t *instruction = nullptr;
  ASSERT_EQ(rj_code_decoder_decode(decoder, &kCdnaSEndpgm, &instruction), ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(instruction, nullptr);
  EXPECT_STREQ(rj_code_inst_mnemonic(instruction), "s_endpgm");
  rj_code_inst_destroy(instruction);

  rj_code_inst_t *survivor = nullptr;
  ASSERT_EQ(rj_code_decoder_decode(decoder, &kCdnaSEndpgm, &survivor), ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(survivor, nullptr);
  rj_code_decoder_destroy(decoder);

  EXPECT_STREQ(rj_code_inst_mnemonic(survivor), "s_endpgm");
  EXPECT_GT(rj_code_inst_size(survivor), 0u);
  char disassembly[64]{};
  EXPECT_EQ(rj_code_inst_disassemble(survivor, disassembly, sizeof(disassembly)),
            ROCJITSU_STATUS_SUCCESS);
  EXPECT_NE(std::strstr(disassembly, "s_endpgm"), nullptr);
  EXPECT_EQ(rj_code_inst_next(survivor), nullptr);

  rj_code_inst_destroy(survivor);
  rj_code_inst_destroy(nullptr);
}

TEST(DecoderCApiTest, StandaloneInstructionIgnoresAmbientDecoderPool) {
  auto pooled_decoder = rocjitsu::Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_NE(pooled_decoder, nullptr);
  pooled_decoder->enable_pool();
  const auto *active_pool =
      static_cast<const rocjitsu::Decoder::Pool *>(rocjitsu::Instruction::alloc_pool_);
  ASSERT_NE(active_pool, nullptr);

  rj_code_decoder_t *decoder = nullptr;
  ASSERT_EQ(rj_code_decoder_create(ROCJITSU_CODE_ARCH_CDNA3, &decoder), ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(decoder, nullptr);

  rj_code_inst_t *instruction = nullptr;
  ASSERT_EQ(rj_code_decoder_decode(decoder, &kCdnaSEndpgm, &instruction), ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(instruction, nullptr);
  EXPECT_FALSE(active_pool->owns(instruction));
  EXPECT_EQ(rocjitsu::Instruction::alloc_pool_, active_pool);
  rj_code_inst_destroy(instruction);
  EXPECT_EQ(rocjitsu::Instruction::alloc_pool_, active_pool);

  instruction = nullptr;
  ASSERT_EQ(rj_code_decoder_decode(decoder, &kCdnaSEndpgm, &instruction), ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(instruction, nullptr);
  EXPECT_FALSE(active_pool->owns(instruction));

  rj_code_decoder_destroy(decoder);
  pooled_decoder.reset();
  EXPECT_EQ(rocjitsu::Instruction::alloc_pool_, nullptr);

  EXPECT_STREQ(rj_code_inst_mnemonic(instruction), "s_endpgm");
  rj_code_inst_destroy(instruction);
}

TEST(DecoderCApiTest, HeapAllocationScopeForgetsDestroyedPool) {
  auto pooled_decoder = rocjitsu::Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_NE(pooled_decoder, nullptr);
  pooled_decoder->enable_pool();
  ASSERT_NE(rocjitsu::Instruction::alloc_pool_, nullptr);

  {
    rocjitsu::Instruction::ScopedHeapAllocation heap_allocation;
    pooled_decoder.reset();
  }
  EXPECT_EQ(rocjitsu::Instruction::alloc_pool_, nullptr);

  auto decoder = rocjitsu::Decoder::create(ROCJITSU_CODE_ARCH_CDNA3);
  ASSERT_NE(decoder, nullptr);
  std::unique_ptr<rocjitsu::Instruction> instruction(decode_valid(*decoder, &kCdnaSEndpgm));
  ASSERT_NE(instruction, nullptr);
  EXPECT_EQ(instruction->mnemonic(), "s_endpgm");
}

} // namespace
