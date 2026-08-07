// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gfx1250_b0_to_a0_library_test.cpp
/// @brief Tests the fixed-profile gfx1250 B0-to-A0 shared-library API.

#include "rocjitsu/code/dbt/gfx1250_b0_to_a0_diagnostics.h"
#include "rocjitsu/code/rj_gfx1250_b0_to_a0.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/builders.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/opcodes.h"
#include "support/gfx1250_test_code_object.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

struct CapturedDiagnostic {
  std::string severity;
  std::string kind;
  bool has_guest_offset = false;
  uint64_t guest_offset = 0;
  std::string mnemonic;
  std::string message;
  bool required_work = false;
};

void capture_diagnostic(const rj_gfx1250_b0_to_a0_diagnostic_t *diagnostic, void *user_data) {
  auto *captured = static_cast<std::vector<CapturedDiagnostic> *>(user_data);
  captured->push_back(
      {diagnostic->severity != nullptr ? diagnostic->severity : "",
       diagnostic->kind != nullptr ? diagnostic->kind : "", diagnostic->has_guest_offset != 0,
       diagnostic->guest_offset, diagnostic->mnemonic != nullptr ? diagnostic->mnemonic : "",
       diagnostic->message != nullptr ? diagnostic->message : "", diagnostic->required_work != 0});
}

#ifdef GFX1250_B0_TO_A0_FIXTURE
uint64_t source_identity(const std::vector<uint8_t> &bytes) {
  constexpr uint64_t kOffsetBasis = 14695981039346656037ULL;
  constexpr uint64_t kPrime = 1099511628211ULL;
  uint64_t identity = kOffsetBasis;
  for (uint8_t byte : bytes) {
    identity ^= byte;
    identity *= kPrime;
  }
  return identity;
}
#endif

TEST(Gfx1250B0ToA0Library, RejectsInvalidArgumentsAndClearsOutputs) {
  auto *output = reinterpret_cast<uint8_t *>(0x1);
  size_t output_size = 1;
  rj_gfx1250_b0_to_a0_translation_info_t info{1, 1};
  EXPECT_EQ(
      rj_gfx1250_b0_to_a0_translate(nullptr, 0, &output, &output_size, &info, nullptr, nullptr),
      ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(output, nullptr);
  EXPECT_EQ(output_size, 0u);
  EXPECT_EQ(info.source_code_object_id, 0u);
  EXPECT_EQ(info.changed_instruction_count, 0u);

  constexpr std::array<uint8_t, 64> kNotElf = {'N', 'O', 'T', 'E', 'L', 'F'};
  EXPECT_EQ(rj_gfx1250_b0_to_a0_translate(kNotElf.data(), kNotElf.size(), &output, &output_size,
                                          &info, nullptr, nullptr),
            ROCJITSU_STATUS_INVALID_CODE_OBJECT);
  EXPECT_EQ(output, nullptr);
  EXPECT_EQ(output_size, 0u);
  EXPECT_NE(info.source_code_object_id, 0u);
  EXPECT_EQ(info.changed_instruction_count, 0u);

  rj_gfx1250_b0_to_a0_free(nullptr);
}

TEST(Gfx1250B0ToA0Library, ReportsInvalidCodeObjectDiagnostic) {
  constexpr std::array<uint8_t, 64> kNotElf = {'N', 'O', 'T', 'E', 'L', 'F'};
  uint8_t *output = nullptr;
  size_t output_size = 0;
  rj_gfx1250_b0_to_a0_translation_info_t info{};
  std::vector<CapturedDiagnostic> diagnostics;

  EXPECT_EQ(rj_gfx1250_b0_to_a0_translate(kNotElf.data(), kNotElf.size(), &output, &output_size,
                                          &info, capture_diagnostic, &diagnostics),
            ROCJITSU_STATUS_INVALID_CODE_OBJECT);
  ASSERT_FALSE(diagnostics.empty());
  const auto matching = std::find_if(diagnostics.begin(), diagnostics.end(), [](const auto &item) {
    return item.severity == "error" && item.kind == "input-invalid-code-object" &&
           item.message.find("valid gfx1250") != std::string::npos;
  });
  EXPECT_NE(matching, diagnostics.end());
}

TEST(Gfx1250B0ToA0Library, ReportsTranslatorDiagnostics) {
  rocjitsu::gfx1250::Vop3VopDpp16MachineInst dpp{};
  dpp.vdst = 30;
  dpp.clamp = 1;
  dpp.op = rocjitsu::gfx1250::kVCvtPkFp8F32Vop3;
  dpp.encoding = 0x35;
  dpp.src0 = 250;
  dpp.src1 = 256 + 2;
  dpp.vsrc0 = 22;
  dpp.fi = 1;
  dpp.bank_mask = 0xf;
  dpp.row_mask = 0xf;
  std::array<uint32_t, 3> conversion{};
  std::memcpy(conversion.data(), &dpp, sizeof(dpp));
  constexpr uint32_t kEndpgm = 0xBFB00000u;
  const std::array<uint32_t, 4> text = {conversion[0], conversion[1], conversion[2], kEndpgm};
  const auto source = rocjitsu::test_support::make_gfx1250_code_object(text);
  uint8_t *output = nullptr;
  size_t output_size = 0;
  rj_gfx1250_b0_to_a0_translation_info_t info{};
  std::vector<CapturedDiagnostic> diagnostics;

  EXPECT_EQ(rj_gfx1250_b0_to_a0_translate(source.data(), source.size(), &output, &output_size,
                                          &info, capture_diagnostic, &diagnostics),
            ROCJITSU_STATUS_INVALID_CODE_OBJECT);
  EXPECT_EQ(output, nullptr);
  EXPECT_EQ(output_size, 0u);
  ASSERT_FALSE(diagnostics.empty());

  const auto primary = std::find_if(diagnostics.begin(), diagnostics.end(), [](const auto &item) {
    return !item.required_work && item.severity == "error" &&
           item.kind == "translator-expand-failed";
  });
  ASSERT_NE(primary, diagnostics.end());
  EXPECT_TRUE(primary->has_guest_offset);
  EXPECT_EQ(primary->guest_offset, 0u);
  EXPECT_EQ(primary->mnemonic, "v_cvt_pk_fp8_f32");
  EXPECT_NE(primary->message.find("does not support DPP"), std::string::npos);
}

// The diagnostic above carries no required work. This one does, so it covers
// the fan-out through the public C entry point rather than the emit helper
// exercised by FansOutRequiredWorkAsCallbackViews.
TEST(Gfx1250B0ToA0Library, ReportsTranslatorExpandFailedAndRequiredWork) {
  constexpr auto conversion =
      rocjitsu::gfx1250::build_sop1(rocjitsu::gfx1250::kSBarrierSignalIsfirstSop1, {.ssrc0 = 195});
  constexpr uint32_t kEndpgm = 0xBFB00000u;
  const std::array<uint32_t, 2> text = {conversion[0], kEndpgm};
  const auto source = rocjitsu::test_support::make_gfx1250_code_object(text);
  uint8_t *output = nullptr;
  size_t output_size = 0;
  rj_gfx1250_b0_to_a0_translation_info_t info{};
  std::vector<CapturedDiagnostic> diagnostics;

  EXPECT_EQ(rj_gfx1250_b0_to_a0_translate(source.data(), source.size(), &output, &output_size,
                                          &info, capture_diagnostic, &diagnostics),
            ROCJITSU_STATUS_INVALID_CODE_OBJECT);
  EXPECT_EQ(output, nullptr);
  EXPECT_EQ(output_size, 0u);
  ASSERT_FALSE(diagnostics.empty());

  const auto primary = std::find_if(diagnostics.begin(), diagnostics.end(), [](const auto &item) {
    return !item.required_work && item.severity == "error" &&
           item.kind == "translator-expand-failed";
  });
  ASSERT_NE(primary, diagnostics.end());
  EXPECT_TRUE(primary->has_guest_offset);
  EXPECT_EQ(primary->guest_offset, 0u);
  EXPECT_EQ(primary->mnemonic, "s_barrier_signal_isfirst");

  const auto required = std::find_if(diagnostics.begin(), diagnostics.end(), [](const auto &item) {
    return item.required_work && item.kind == "translator-expand-failed";
  });
  ASSERT_NE(required, diagnostics.end());
  EXPECT_TRUE(required->has_guest_offset);
  EXPECT_EQ(required->guest_offset, 0u);
  EXPECT_EQ(required->mnemonic, "s_barrier_signal_isfirst");
  EXPECT_NE(required->message.find("different barrier id"), std::string::npos);
}

TEST(Gfx1250B0ToA0Library, FansOutRequiredWorkAsCallbackViews) {
  const std::vector<rocjitsu::TranslationDiagnostic> source = {{
      .severity = rocjitsu::DiagnosticSeverity::Error,
      .kind = rocjitsu::DiagnosticKind::ExpandMissing,
      .guest_offset = 8,
      .mnemonic = "v_test",
      .message = "primary diagnostic",
      .required_work = {"first required step", "second required step"},
  }};
  std::vector<CapturedDiagnostic> captured;

  rocjitsu::emit_gfx1250_b0_to_a0_diagnostics(capture_diagnostic, &captured, source);

  ASSERT_EQ(captured.size(), 3u);
  EXPECT_FALSE(captured[0].required_work);
  EXPECT_EQ(captured[0].kind, "translator-expand-missing");
  EXPECT_EQ(captured[0].message, "primary diagnostic");
  for (size_t index = 1; index < captured.size(); ++index) {
    EXPECT_TRUE(captured[index].required_work);
    EXPECT_EQ(captured[index].severity, captured[0].severity);
    EXPECT_EQ(captured[index].kind, captured[0].kind);
    EXPECT_EQ(captured[index].has_guest_offset, captured[0].has_guest_offset);
    EXPECT_EQ(captured[index].guest_offset, captured[0].guest_offset);
    EXPECT_EQ(captured[index].mnemonic, captured[0].mnemonic);
  }
  EXPECT_EQ(captured[1].message, "first required step");
  EXPECT_EQ(captured[2].message, "second required step");
}

#ifdef GFX1250_B0_TO_A0_FIXTURE
TEST(Gfx1250B0ToA0Library, TranslatesRealGfx1250CodeObject) {
  std::ifstream input(GFX1250_B0_TO_A0_FIXTURE, std::ios::binary);
  ASSERT_TRUE(input) << GFX1250_B0_TO_A0_FIXTURE;
  const std::vector<uint8_t> source((std::istreambuf_iterator<char>(input)),
                                    std::istreambuf_iterator<char>());
  ASSERT_GE(source.size(), 4u);

  uint8_t *output = nullptr;
  size_t output_size = 0;
  rj_gfx1250_b0_to_a0_translation_info_t info{};
  ASSERT_EQ(rj_gfx1250_b0_to_a0_translate(source.data(), source.size(), &output, &output_size,
                                          &info, nullptr, nullptr),
            ROCJITSU_STATUS_SUCCESS);
  ASSERT_NE(output, nullptr);
  EXPECT_EQ(info.source_code_object_id, source_identity(source));
  EXPECT_GT(info.changed_instruction_count, 0u);
  constexpr std::array<uint8_t, 4> kElfMagic = {0x7f, 'E', 'L', 'F'};
  ASSERT_GE(output_size, kElfMagic.size());
  EXPECT_TRUE(std::equal(kElfMagic.begin(), kElfMagic.end(), output));

  rj_gfx1250_b0_to_a0_free(output);
}
#endif

} // namespace
