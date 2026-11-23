// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "dbt/support/gfx1250_test_code_object.h"
#include "rocjitsu/code/rj_code_internal.h"

#include <gtest/gtest.h>

#include <array>
#include <cstring>
#include <memory>
#include <span>

namespace rocjitsu {
namespace {

constexpr std::array<uint32_t, 3> kGfx1251Text{
    0xCC4B4004u, 0x1A021908u, // v_pk_add_f64, unavailable on gfx1250.
    0xBFB00000u,              // s_endpgm.
};

void initialize_source(rj_code_object_t &source, uint32_t machine,
                       std::span<const uint32_t> words = kGfx1251Text) {
  auto image = test_support::make_gfx1250_code_object(words);
  Elf64_Ehdr header{};
  std::memcpy(&header, image.data(), sizeof(header));
  header.e_flags = machine;
  std::memcpy(image.data(), &header, sizeof(header));
  source.owned_co = std::make_unique<AmdGpuCodeObject>(image.data(), image.size());
  source.co = source.owned_co.get();
}

TEST(CodeTargetCApi, EveryBuiltinTargetHasEnumAndStringDecoderSelection) {
  constexpr std::array targets{
      std::pair{ROCJITSU_CODE_TARGET_GFX90A, "gfx90a"},
      std::pair{ROCJITSU_CODE_TARGET_GFX942, "gfx942"},
      std::pair{ROCJITSU_CODE_TARGET_GFX950, "gfx950"},
      std::pair{ROCJITSU_CODE_TARGET_GFX1200, "gfx1200"},
      std::pair{ROCJITSU_CODE_TARGET_GFX1201, "gfx1201"},
      std::pair{ROCJITSU_CODE_TARGET_GFX1250, "gfx1250"},
      std::pair{ROCJITSU_CODE_TARGET_GFX1251, "gfx1251"},
      std::pair{ROCJITSU_CODE_TARGET_GFX1100, "gfx1100"},
      std::pair{ROCJITSU_CODE_TARGET_GFX908, "gfx908"},
      std::pair{ROCJITSU_CODE_TARGET_GFX1010, "gfx1010"},
      std::pair{ROCJITSU_CODE_TARGET_GFX1030, "gfx1030"},
      std::pair{ROCJITSU_CODE_TARGET_GFX1150, "gfx1150"},
      std::pair{ROCJITSU_CODE_TARGET_RV64I, "rv64i"},
  };
  for (const auto &[target, name] : targets) {
    SCOPED_TRACE(name);
    rj_code_decoder_t *decoder = nullptr;
    ASSERT_EQ(rj_code_decoder_create(target, &decoder), ROCJITSU_STATUS_SUCCESS);
    ASSERT_NE(decoder, nullptr);
    rj_code_decoder_destroy(decoder);
    decoder = nullptr;
    ASSERT_EQ(rj_code_decoder_create_for_target(name, &decoder), ROCJITSU_STATUS_SUCCESS);
    ASSERT_NE(decoder, nullptr);
    rj_code_decoder_destroy(decoder);
  }
}

TEST(CodeTargetCApi, EnumDecoderPreservesVariantInstructionLegality) {
  for (rj_code_target_id_t target : {ROCJITSU_CODE_TARGET_GFX1250, ROCJITSU_CODE_TARGET_GFX1251}) {
    rj_code_decoder_t *decoder = nullptr;
    ASSERT_EQ(rj_code_decoder_create(target, &decoder), ROCJITSU_STATUS_SUCCESS);
    rj_code_inst_t *instruction = nullptr;
    const rj_status_t status = rj_code_decoder_decode(decoder, kGfx1251Text.data(), &instruction);
    EXPECT_EQ(status, target == ROCJITSU_CODE_TARGET_GFX1251 ? ROCJITSU_STATUS_SUCCESS
                                                             : ROCJITSU_STATUS_ERROR);
    EXPECT_EQ(instruction != nullptr, target == ROCJITSU_CODE_TARGET_GFX1251);
    rj_code_inst_destroy(instruction);
    rj_code_decoder_destroy(decoder);
  }
}

TEST(CodeTargetCApi, TranslatePreservesExplicitGfx1251WithAndWithoutSourceMetadata) {
  for (uint32_t machine : {uint32_t{EF_AMDGPU_MACH_AMDGCN_GFX1251}, uint32_t{0x1234}}) {
    SCOPED_TRACE(machine);
    rj_code_object_t source{};
    initialize_source(source, machine);
    ASSERT_TRUE(source.co->is_valid());
    const rj_code_dbt_options_t options{ROCJITSU_CODE_TARGET_GFX1251, ROCJITSU_CODE_TARGET_GFX1251};
    rj_code_object_t *output = nullptr;
    ASSERT_EQ(rj_code_translate(&source, &options, &output), ROCJITSU_STATUS_SUCCESS);
    ASSERT_NE(output, nullptr);
    EXPECT_EQ(output->co->target_id(), ROCJITSU_CODE_TARGET_GFX1251);
    rj_code_inst_list_t *instructions = nullptr;
    EXPECT_EQ(rj_code_inst_list_create(output, ROCJITSU_CODE_TARGET_GFX1251, &instructions),
              ROCJITSU_STATUS_SUCCESS);
    rj_code_inst_list_destroy(instructions);
    rj_code_object_destroy(output);
  }
}

TEST(CodeTargetCApi, TranslatePreservesNonDefaultRdna4Target) {
  constexpr std::array<uint32_t, 1> words{0xBFB00000u};
  rj_code_object_t source{};
  initialize_source(source, EF_AMDGPU_MACH_AMDGCN_GFX1201, words);
  ASSERT_TRUE(source.co->is_valid());
  const rj_code_dbt_options_t options{ROCJITSU_CODE_TARGET_GFX1201, ROCJITSU_CODE_TARGET_GFX1201};
  rj_code_object_t *output = nullptr;
  ASSERT_EQ(rj_code_translate(&source, &options, &output), ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(output->co->target_id(), ROCJITSU_CODE_TARGET_GFX1201);
  rj_code_object_destroy(output);
}

TEST(CodeTargetCApi, TranslateRejectsSourceMismatchAndInvalidTargets) {
  rj_code_object_t source{};
  initialize_source(source, EF_AMDGPU_MACH_AMDGCN_GFX1251);
  constexpr std::array options{
      rj_code_dbt_options_t{ROCJITSU_CODE_TARGET_GFX1250, ROCJITSU_CODE_TARGET_GFX1251},
      rj_code_dbt_options_t{ROCJITSU_CODE_TARGET_INVALID, ROCJITSU_CODE_TARGET_GFX1251},
      rj_code_dbt_options_t{ROCJITSU_CODE_TARGET_GFX1251, ROCJITSU_CODE_TARGET_INVALID},
      rj_code_dbt_options_t{ROCJITSU_CODE_TARGET_GFX1251, ROCJITSU_CODE_TARGET_RV64I},
      rj_code_dbt_options_t{ROCJITSU_CODE_TARGET_RV64I, ROCJITSU_CODE_TARGET_GFX1251},
      rj_code_dbt_options_t{static_cast<rj_code_target_id_t>(-1), ROCJITSU_CODE_TARGET_GFX1251},
      rj_code_dbt_options_t{ROCJITSU_CODE_TARGET_GFX1251, ROCJITSU_CODE_TARGET_NUM_TARGETS},
  };
  for (const auto &option : options) {
    rj_code_object_t *output = &source;
    EXPECT_EQ(rj_code_translate(&source, &option, &output), ROCJITSU_STATUS_INVALID_ARGUMENT);
    EXPECT_EQ(output, nullptr);
  }
  rj_code_object_t *output = &source;
  EXPECT_EQ(rj_code_translate(&source, nullptr, &output), ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(output, nullptr);
  output = &source;
  EXPECT_EQ(rj_code_translate(nullptr, &options.front(), &output),
            ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(output, nullptr);
  EXPECT_EQ(rj_code_translate(&source, &options.front(), nullptr),
            ROCJITSU_STATUS_INVALID_ARGUMENT);
}

TEST(CodeTargetCApi, TranslateStillRejectsUnsupportedCrossTargetCdna5) {
  constexpr std::array<uint32_t, 1> words{0xBFB00000u};
  for (bool reverse : {false, true}) {
    for (bool recognizable : {false, true}) {
      rj_code_object_t source{};
      const uint32_t machine =
          recognizable ? (reverse ? EF_AMDGPU_MACH_AMDGCN_GFX1250 : EF_AMDGPU_MACH_AMDGCN_GFX1251)
                       : 0x1234;
      initialize_source(source, machine,
                        reverse ? std::span<const uint32_t>(words)
                                : std::span<const uint32_t>(kGfx1251Text));
      const rj_code_dbt_options_t options{
          reverse ? ROCJITSU_CODE_TARGET_GFX1250 : ROCJITSU_CODE_TARGET_GFX1251,
          reverse ? ROCJITSU_CODE_TARGET_GFX1251 : ROCJITSU_CODE_TARGET_GFX1250};
      rj_code_object_t *output = &source;
      EXPECT_EQ(rj_code_translate(&source, &options, &output), ROCJITSU_STATUS_ERROR);
      EXPECT_EQ(output, nullptr);
    }
  }
}

} // namespace
} // namespace rocjitsu
