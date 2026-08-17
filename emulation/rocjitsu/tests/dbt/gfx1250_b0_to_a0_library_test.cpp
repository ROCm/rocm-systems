// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gfx1250_b0_to_a0_library_test.cpp
/// @brief Tests the fixed-profile gfx1250 B0-to-A0 shared-library API.

#include "rocjitsu/code/dbt/gfx1250_b0_to_a0_diagnostics.h"
#include "rocjitsu/code/rj_gfx1250_b0_to_a0.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/builders.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/opcodes.h"
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

// This fixture covers the rule-refused path (`translator-expand-failed`): a
// rule exists for v_cvt_pk_fp8_f32 but declines the DPP operand form.
//
// The sibling `translator-expand-missing` kind -- classified as needing an
// expansion with no rule registered at all -- has no end-to-end gfx1250 fixture
// because no gfx1250 mnemonic can currently reach it. Every branch of
// requires_b0_to_a0_expansion() has a matching entry in
// semantic_expand_rules_gfx1250_b0_to_a0(), and the operand-driven
// flat-scratch-base path answers Success or Failed but never NotHandled.
//
// It is still covered on both sides of that gap: the translator behavior by
// BinaryTranslatorE2E.ExpandLegalizationWithoutSemanticRuleFails (cdna4-to-cdna3,
// tests/dbt/translate_test.cpp), where an Expand legalization without a rule is
// reachable, and the reported kind and its required-work field by
// FansOutRequiredWorkAsCallbackViews below and
// HsaHotswapHookTest.RendersRequiredWorkDiagnostic, both of which feed a
// synthetic diagnostic straight through the reporting seam. Add a fixture here
// if a gfx1250 mnemonic is ever classified fail-closed ahead of its rule.
TEST(Gfx1250B0ToA0Library, ReportsTranslatorDiagnostics) {
  rocjitsu::cdna5::Vop3VopDpp16MachineInst dpp{};
  dpp.vdst = 30;
  dpp.clamp = 1;
  dpp.op = rocjitsu::cdna5::kVCvtPkFp8F32Vop3;
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
      rocjitsu::cdna5::build_sop1(rocjitsu::cdna5::kSBarrierSignalIsfirstSop1, {.ssrc0 = 195});
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
      .output_offset = std::nullopt,
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

// Diagnostics are not only a failure channel. A translation can succeed and
// still have something to report -- here a family passed through unchanged
// because its A0 handling is not implemented yet -- and reporting only on the
// undispatchable path would drop it. That gap is what someone triaging a
// misbehaving kernel reads these diagnostics for, so it has to reach the
// callback on the success path too.
TEST(Gfx1250B0ToA0Library, ReportsDeferredFamilyDiagnosticOnSuccessfulTranslation) {
  constexpr auto deferred =
      rocjitsu::cdna5::build_sopp(rocjitsu::cdna5::kSMonitorSleepSopp, {.simm16 = 1});
  constexpr uint32_t kEndpgm = 0xBFB00000u;
  const std::array<uint32_t, 3> text = {deferred[0], deferred[0], kEndpgm};
  const auto source = rocjitsu::test_support::make_gfx1250_code_object(text);
  uint8_t *output = nullptr;
  size_t output_size = 0;
  rj_gfx1250_b0_to_a0_translation_info_t info{};
  std::vector<CapturedDiagnostic> diagnostics;

  ASSERT_EQ(rj_gfx1250_b0_to_a0_translate(source.data(), source.size(), &output, &output_size,
                                          &info, capture_diagnostic, &diagnostics),
            ROCJITSU_STATUS_SUCCESS);
  EXPECT_NE(output, nullptr);
  rj_gfx1250_b0_to_a0_free(output);

  const auto reported = std::find_if(diagnostics.begin(), diagnostics.end(), [](const auto &item) {
    return item.severity == "warning" && item.kind == "translator-legalization" &&
           item.mnemonic == "s_monitor_sleep";
  });
  ASSERT_NE(reported, diagnostics.end())
      << "a successful translation must still report the pass-through gap";
  EXPECT_NE(reported->message.find("not yet implemented"), std::string::npos);

  // Two instructions, one report: the gap is a property of the mnemonic.
  const auto count = std::count_if(diagnostics.begin(), diagnostics.end(), [](const auto &item) {
    return item.mnemonic == "s_monitor_sleep";
  });
  EXPECT_EQ(count, 1);
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
