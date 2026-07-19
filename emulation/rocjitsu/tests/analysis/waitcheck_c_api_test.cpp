// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "../tools/waitcheck_fixture.h"
#include "rocjitsu/analysis/rj_waitcheck.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace {

struct OwnedDiagnostic {
  rj_waitcheck_counter_t counter = ROCJITSU_WAITCHECK_COUNTER_INVALID;
  rj_waitcheck_access_t access = ROCJITSU_WAITCHECK_ACCESS_INVALID;
  rj_waitcheck_register_t reg{};
  std::string section_name;
  uint64_t section_offset = 0;
  uint64_t producer_section_offset = 0;
  std::string instruction;
  std::string producer_instruction;
  uint32_t required_count = 0;
  std::string message;
};

struct CallbackState {
  std::vector<OwnedDiagnostic> diagnostics;
  std::vector<std::string> errors;
};

void capture_diagnostic(const rj_waitcheck_diagnostic_t *diagnostic, void *user_data) {
  auto &state = *static_cast<CallbackState *>(user_data);
  state.diagnostics.push_back(OwnedDiagnostic{
      .counter = diagnostic->counter,
      .access = diagnostic->access,
      .reg = diagnostic->reg,
      .section_name = diagnostic->section_name,
      .section_offset = diagnostic->section_offset,
      .producer_section_offset = diagnostic->producer_section_offset,
      .instruction = diagnostic->instruction,
      .producer_instruction = diagnostic->producer_instruction,
      .required_count = diagnostic->required_count,
      .message = diagnostic->message,
  });
}

void capture_error(const char *message, void *user_data) {
  static_cast<CallbackState *>(user_data)->errors.emplace_back(message);
}

[[nodiscard]] rj_waitcheck_options_t callback_options(CallbackState &state) {
  rj_waitcheck_options_t options;
  rj_waitcheck_options_init(&options);
  options.diagnostic_callback = capture_diagnostic;
  options.error_callback = capture_error;
  options.user_data = &state;
  return options;
}

TEST(WaitcheckCApiTest, DefaultOptionsReportEveryDiagnosticWithoutStoppingEarly) {
  rj_waitcheck_options_t options{};
  rj_waitcheck_options_init(&options);

  EXPECT_EQ(options.max_diagnostics, 0u);
  EXPECT_EQ(options.max_reachability_cache_bytes, 0u);
  EXPECT_EQ(options.stop_after_first_diagnostic, 0u);
  EXPECT_EQ(options.diagnostic_callback, nullptr);
  EXPECT_EQ(options.error_callback, nullptr);
  EXPECT_EQ(options.user_data, nullptr);

  rj_waitcheck_options_init(nullptr);
}

TEST(WaitcheckCApiTest, ReportsStructuredDiagnosticFromHazardousBuffer) {
  const auto image = rocjitsu::waitcheck_test::make_gfx1200_missing_wait_code_object();
  CallbackState state;
  const rj_waitcheck_options_t options = callback_options(state);
  rj_waitcheck_result_t result{};

  ASSERT_EQ(rj_waitcheck_analyze(image.data(), image.size(), &options, &result),
            ROCJITSU_STATUS_SUCCESS);
  EXPECT_EQ(result.passed, 0u);
  EXPECT_GT(result.instructions_analyzed, 0u);
  EXPECT_EQ(result.kernels_discovered, 1u);
  EXPECT_EQ(result.kernels_analyzed, 1u);
  EXPECT_EQ(result.diagnostics_observed, 1u);
  EXPECT_EQ(result.diagnostics_reported, 1u);
  EXPECT_EQ(result.diagnostics_truncated, 0u);
  EXPECT_EQ(result.stopped_early, 0u);
  ASSERT_EQ(state.diagnostics.size(), 1u);
  EXPECT_TRUE(state.errors.empty());

  const OwnedDiagnostic &diagnostic = state.diagnostics.front();
  EXPECT_EQ(diagnostic.counter, ROCJITSU_WAITCHECK_COUNTER_LOAD);
  EXPECT_EQ(diagnostic.access, ROCJITSU_WAITCHECK_ACCESS_USE);
  EXPECT_EQ(diagnostic.reg.register_class, ROCJITSU_WAITCHECK_REGISTER_VGPR);
  EXPECT_EQ(diagnostic.reg.index, 0u);
  EXPECT_EQ(diagnostic.reg.width, 1u);
  EXPECT_EQ(diagnostic.section_name, ".text");
  EXPECT_GT(diagnostic.section_offset, diagnostic.producer_section_offset);
  EXPECT_FALSE(diagnostic.instruction.empty());
  EXPECT_FALSE(diagnostic.producer_instruction.empty());
  EXPECT_EQ(diagnostic.required_count, 0u);
  EXPECT_NE(diagnostic.message.find("missing s_wait_loadcnt"), std::string::npos);
}

TEST(WaitcheckCApiTest, CleanBufferPassesWithNullOptions) {
  const auto image = rocjitsu::waitcheck_test::make_gfx1200_correct_wait_code_object();
  rj_waitcheck_result_t result{};

  ASSERT_EQ(rj_waitcheck_analyze(image.data(), image.size(), nullptr, &result),
            ROCJITSU_STATUS_SUCCESS);
  EXPECT_EQ(result.passed, 1u);
  EXPECT_EQ(result.diagnostics_observed, 0u);
  EXPECT_EQ(result.diagnostics_reported, 0u);
}

TEST(WaitcheckCApiTest, HazardsStillFailWithoutDiagnosticCallback) {
  const auto image = rocjitsu::waitcheck_test::make_gfx1200_missing_wait_code_object();
  rj_waitcheck_result_t result{};

  ASSERT_EQ(rj_waitcheck_analyze(image.data(), image.size(), nullptr, &result),
            ROCJITSU_STATUS_SUCCESS);
  EXPECT_EQ(result.passed, 0u);
  EXPECT_EQ(result.diagnostics_observed, 1u);
  EXPECT_EQ(result.diagnostics_reported, 0u);
  EXPECT_EQ(result.diagnostics_truncated, 1u);
}

TEST(WaitcheckCApiTest, SelectsOneKernelByTextEntryOffset) {
  std::vector<uint32_t> hazardous;
  rocjitsu::waitcheck_test::append_inst(hazardous, rocjitsu::waitcheck_test::global_load_b32(0));
  rocjitsu::waitcheck_test::append_inst(hazardous, rocjitsu::waitcheck_test::v_mov_b32(1, 0));
  hazardous.push_back(0xBFB00000U); // s_endpgm
  const std::vector<uint32_t> clean{0xBFB00000U};
  const auto image = rocjitsu::waitcheck_test::make_gfx_multi_kernel_code_object(
      {{"hazardous", hazardous}, {"clean", clean}}, rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1200);

  CallbackState state;
  rj_waitcheck_options_t options = callback_options(state);
  rj_waitcheck_result_t result{};

  ASSERT_EQ(rj_waitcheck_analyze_kernel(image.data(), image.size(),
                                        hazardous.size() * sizeof(uint32_t), &options, &result),
            ROCJITSU_STATUS_SUCCESS);
  EXPECT_EQ(result.passed, 1u);
  EXPECT_EQ(result.kernels_discovered, 2u);
  EXPECT_EQ(result.kernels_analyzed, 1u);
  EXPECT_EQ(result.diagnostics_observed, 0u);
  EXPECT_TRUE(state.diagnostics.empty());
  EXPECT_TRUE(state.errors.empty());

  ASSERT_EQ(rj_waitcheck_analyze_kernel(image.data(), image.size(), 0, &options, &result),
            ROCJITSU_STATUS_SUCCESS);
  EXPECT_EQ(result.passed, 0u);
  EXPECT_EQ(result.kernels_analyzed, 1u);
  EXPECT_EQ(result.diagnostics_observed, 1u);
  ASSERT_EQ(state.diagnostics.size(), 1u);
}

TEST(WaitcheckCApiTest, DiagnosticLimitReportsTruncationAndPreservesFailure) {
  std::vector<uint32_t> hazardous;
  rocjitsu::waitcheck_test::append_inst(hazardous, rocjitsu::waitcheck_test::global_load_b32(0));
  rocjitsu::waitcheck_test::append_inst(hazardous, rocjitsu::waitcheck_test::v_mov_b32(1, 0));
  hazardous.push_back(0xBFB00000U); // s_endpgm
  const auto image = rocjitsu::waitcheck_test::make_gfx_multi_kernel_code_object(
      {{"first", hazardous}, {"second", hazardous}}, rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1200);
  CallbackState state;
  rj_waitcheck_options_t options = callback_options(state);
  options.max_diagnostics = 1;
  rj_waitcheck_result_t result{};

  ASSERT_EQ(rj_waitcheck_analyze(image.data(), image.size(), &options, &result),
            ROCJITSU_STATUS_SUCCESS);
  EXPECT_EQ(result.passed, 0u);
  EXPECT_EQ(result.kernels_discovered, 2u);
  EXPECT_EQ(result.kernels_analyzed, 2u);
  EXPECT_EQ(result.diagnostics_observed, 1u);
  EXPECT_EQ(result.diagnostics_reported, 1u);
  EXPECT_EQ(result.diagnostics_truncated, 1u);
  EXPECT_EQ(state.diagnostics.size(), 1u);
}

TEST(WaitcheckCApiTest, RejectsMalformedBufferAndUnknownKernelOffset) {
  const uint8_t malformed[] = {0x7f, 'E', 'L', 'F'};
  CallbackState state;
  rj_waitcheck_options_t options = callback_options(state);
  rj_waitcheck_result_t result{};

  EXPECT_EQ(rj_waitcheck_analyze(malformed, sizeof(malformed), &options, &result),
            ROCJITSU_STATUS_INVALID_CODE_OBJECT);
  ASSERT_EQ(state.errors.size(), 1u);
  EXPECT_NE(state.errors.back().find("valid AMDGPU HSA code object"), std::string::npos);

  const auto image = rocjitsu::waitcheck_test::make_gfx1200_correct_wait_code_object();
  EXPECT_EQ(rj_waitcheck_analyze_kernel(image.data(), image.size(), 0x12345678, &options, &result),
            ROCJITSU_STATUS_INVALID_ARGUMENT);
  ASSERT_EQ(state.errors.size(), 2u);
  EXPECT_NE(state.errors.back().find("kernel entry offset"), std::string::npos);
}

TEST(WaitcheckCApiTest, RejectsMissingRequiredArguments) {
  rj_waitcheck_result_t result{};
  EXPECT_EQ(rj_waitcheck_analyze(nullptr, 1, nullptr, &result), ROCJITSU_STATUS_INVALID_ARGUMENT);
  const uint8_t byte = 0;
  EXPECT_EQ(rj_waitcheck_analyze(&byte, 0, nullptr, &result), ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(rj_waitcheck_analyze(&byte, 1, nullptr, nullptr), ROCJITSU_STATUS_INVALID_ARGUMENT);
}

} // namespace
