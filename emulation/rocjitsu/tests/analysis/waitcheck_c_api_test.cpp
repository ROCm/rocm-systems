// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "../tools/waitcheck_fixture.h"
#include "rocjitsu/analysis/rj_waitcheck.h"
#include "rocjitsu/code/builders/instruction_builder.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

namespace {

struct OwnedDiagnostic {
  size_t struct_size = 0;
  uint32_t abi_version = 0;
  bool has_kernel = false;
  std::string kernel_name;
  uint64_t kernel_entry_offset = 0;
  rj_waitcheck_diagnostic_code_t code = ROCJITSU_WAITCHECK_DIAGNOSTIC_UNKNOWN;
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

struct OwnedCounterParityDiagnostic {
  size_t struct_size = 0;
  uint32_t abi_version = 0;
  rj_waitcheck_counter_parity_kind_t kind = ROCJITSU_WAITCHECK_COUNTER_PARITY_UNKNOWN;
  bool has_kernel = false;
  std::string kernel_name;
  uint64_t kernel_entry_offset = 0;
  rj_waitcheck_counter_t counter = ROCJITSU_WAITCHECK_COUNTER_INVALID;
  uint32_t emitted_count = 0;
  uint32_t required_count = 0;
  bool has_required_dependency = false;
  rj_waitcheck_access_t access = ROCJITSU_WAITCHECK_ACCESS_INVALID;
  rj_waitcheck_register_t reg{};
  std::string section_name;
  uint64_t wait_section_offset = 0;
  std::string wait_instruction;
  uint64_t consumer_section_offset = 0;
  std::string consumer_instruction;
  uint64_t producer_section_offset = 0;
  std::string producer_instruction;
  std::string message;
};

struct CallbackState {
  std::vector<OwnedDiagnostic> diagnostics;
  std::vector<OwnedCounterParityDiagnostic> counter_parity_diagnostics;
  std::vector<std::string> errors;
};

void capture_diagnostic(const rj_waitcheck_diagnostic_t *diagnostic, void *user_data) {
  auto &state = *static_cast<CallbackState *>(user_data);
  state.diagnostics.push_back(OwnedDiagnostic{
      .struct_size = diagnostic->struct_size,
      .abi_version = diagnostic->abi_version,
      .has_kernel = diagnostic->has_kernel != 0,
      .kernel_name = diagnostic->kernel_name ? diagnostic->kernel_name : "",
      .kernel_entry_offset = diagnostic->kernel_entry_offset,
      .code = diagnostic->code,
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

void capture_counter_parity(const rj_waitcheck_counter_parity_diagnostic_t *diagnostic,
                            void *user_data) {
  auto &state = *static_cast<CallbackState *>(user_data);
  state.counter_parity_diagnostics.push_back(OwnedCounterParityDiagnostic{
      .struct_size = diagnostic->struct_size,
      .abi_version = diagnostic->abi_version,
      .kind = diagnostic->kind,
      .has_kernel = diagnostic->has_kernel != 0,
      .kernel_name = diagnostic->kernel_name ? diagnostic->kernel_name : "",
      .kernel_entry_offset = diagnostic->kernel_entry_offset,
      .counter = diagnostic->counter,
      .emitted_count = diagnostic->emitted_count,
      .required_count = diagnostic->required_count,
      .has_required_dependency = diagnostic->has_required_dependency != 0,
      .access = diagnostic->access,
      .reg = diagnostic->reg,
      .section_name = diagnostic->section_name,
      .wait_section_offset = diagnostic->wait_section_offset,
      .wait_instruction = diagnostic->wait_instruction,
      .consumer_section_offset = diagnostic->consumer_section_offset,
      .consumer_instruction = diagnostic->consumer_instruction,
      .producer_section_offset = diagnostic->producer_section_offset,
      .producer_instruction =
          diagnostic->producer_instruction ? diagnostic->producer_instruction : "",
      .message = diagnostic->message,
  });
}

void throw_from_diagnostic(const rj_waitcheck_diagnostic_t *, void *) {
  throw std::runtime_error("consumer diagnostic callback failed");
}

void throw_from_counter_parity(const rj_waitcheck_counter_parity_diagnostic_t *, void *) {
  throw std::runtime_error("consumer counter-parity callback failed");
}

void capture_error(const char *message, void *user_data) {
  static_cast<CallbackState *>(user_data)->errors.emplace_back(message);
}

[[nodiscard]] rj_waitcheck_options_t callback_options(CallbackState &state) {
  rj_waitcheck_options_t options;
  EXPECT_EQ(rj_waitcheck_options_init(&options, sizeof(options)), ROCJITSU_STATUS_SUCCESS);
  options.diagnostic_callback = capture_diagnostic;
  options.error_callback = capture_error;
  options.user_data = &state;
  return options;
}

[[nodiscard]] rj_waitcheck_result_t initialized_result() {
  rj_waitcheck_result_t result;
  EXPECT_EQ(rj_waitcheck_result_init(&result, sizeof(result)), ROCJITSU_STATUS_SUCCESS);
  return result;
}

TEST(WaitcheckCApiTest, DefaultOptionsReportEveryDiagnosticWithoutStoppingEarly) {
  rj_waitcheck_options_t options{};
  ASSERT_EQ(rj_waitcheck_options_init(&options, sizeof(options)), ROCJITSU_STATUS_SUCCESS);

  EXPECT_EQ(options.struct_size, sizeof(options));
  EXPECT_EQ(options.abi_version, ROCJITSU_WAITCHECK_ABI_VERSION);
  EXPECT_EQ(options.max_diagnostics, 0u);
  EXPECT_EQ(options.max_reachability_cache_bytes, 0u);
  EXPECT_EQ(options.stop_after_first_diagnostic, 0u);
  EXPECT_EQ(options.diagnostic_callback, nullptr);
  EXPECT_EQ(options.error_callback, nullptr);
  EXPECT_EQ(options.user_data, nullptr);
  EXPECT_EQ(options.check_counter_parity, 0u);
  EXPECT_EQ(options.max_counter_parity_diagnostics, 0u);
  EXPECT_EQ(options.counter_parity_callback, nullptr);

  EXPECT_EQ(rj_waitcheck_options_init(nullptr, sizeof(options)), ROCJITSU_STATUS_INVALID_ARGUMENT);
}

TEST(WaitcheckCApiTest, ReportsGfx950CounterParityWithoutChangingPassStatus) {
  const auto image = rocjitsu::waitcheck_test::make_gfx_multi_kernel_code_object(
      {{"strong_wait",
        {0xE0501000u, 0x80000008u, // buffer_load_dword v0, v8, s[0:3] offen
         0xE05D1000u, 0x80100008u, // buffer_load_dwordx4 ... lds
         0xBF8C0F70u,              // s_waitcnt vmcnt(0); vmcnt(1) is sufficient
         0x7E020300u,              // v_mov_b32 v1, v0
         0xBF810000u}}},           // s_endpgm
      rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX950);
  CallbackState state;
  rj_waitcheck_options_t options = callback_options(state);
  options.check_counter_parity = 1;
  options.counter_parity_callback = capture_counter_parity;
  rj_waitcheck_result_t result = initialized_result();

  ASSERT_EQ(rj_waitcheck_analyze(image.data(), image.size(), &options, &result),
            ROCJITSU_STATUS_SUCCESS);
  EXPECT_EQ(result.target, ROCJITSU_WAITCHECK_TARGET_GFX950);
  EXPECT_EQ(result.passed, 1u);
  EXPECT_EQ(result.diagnostics_observed, 0u);
  EXPECT_EQ(result.counter_parity_wait_groups, 1u);
  EXPECT_EQ(result.counter_parity_fields_checked, 1u);
  EXPECT_EQ(result.counter_parity_exact, 0u);
  EXPECT_EQ(result.counter_underaccounting_observed, 1u);
  EXPECT_EQ(result.counter_unmodeled_wait_observed, 0u);
  EXPECT_EQ(result.counter_parity_indeterminate_groups, 0u);
  EXPECT_EQ(result.counter_parity_diagnostics_reported, 1u);
  EXPECT_EQ(result.counter_parity_diagnostics_truncated, 0u);
  EXPECT_EQ(result.counter_parity_evaluated, 1u);
  ASSERT_EQ(state.counter_parity_diagnostics.size(), 1u);
  EXPECT_TRUE(state.diagnostics.empty());
  EXPECT_TRUE(state.errors.empty());

  const auto &diagnostic = state.counter_parity_diagnostics.front();
  EXPECT_EQ(diagnostic.struct_size, sizeof(rj_waitcheck_counter_parity_diagnostic_t));
  EXPECT_EQ(diagnostic.abi_version, ROCJITSU_WAITCHECK_ABI_VERSION);
  EXPECT_EQ(diagnostic.kind, ROCJITSU_WAITCHECK_COUNTER_PARITY_MODELED_UNDERACCOUNTING);
  EXPECT_STREQ(rj_waitcheck_counter_parity_kind_name(diagnostic.kind),
               "modeled-counter-underaccounting");
  EXPECT_TRUE(diagnostic.has_kernel);
  EXPECT_EQ(diagnostic.kernel_name, "strong_wait");
  EXPECT_EQ(diagnostic.kernel_entry_offset, 0u);
  EXPECT_EQ(diagnostic.counter, ROCJITSU_WAITCHECK_COUNTER_LOAD);
  EXPECT_EQ(diagnostic.emitted_count, 0u);
  EXPECT_EQ(diagnostic.required_count, 1u);
  EXPECT_TRUE(diagnostic.has_required_dependency);
  EXPECT_EQ(diagnostic.access, ROCJITSU_WAITCHECK_ACCESS_USE);
  EXPECT_EQ(diagnostic.reg.register_class, ROCJITSU_WAITCHECK_REGISTER_VGPR);
  EXPECT_EQ(diagnostic.reg.index, 0u);
  EXPECT_EQ(diagnostic.reg.width, 1u);
  EXPECT_EQ(diagnostic.section_name, ".text");
  EXPECT_EQ(diagnostic.wait_section_offset, 16u);
  EXPECT_EQ(diagnostic.consumer_section_offset, 20u);
  EXPECT_EQ(diagnostic.producer_section_offset, 0u);
  EXPECT_FALSE(diagnostic.wait_instruction.empty());
  EXPECT_FALSE(diagnostic.consumer_instruction.empty());
  EXPECT_FALSE(diagnostic.producer_instruction.empty());
}

TEST(WaitcheckCApiTest, CounterParityReportsWhenTheRequestedModelDidNotRun) {
  const auto image = rocjitsu::waitcheck_test::make_gfx1200_correct_wait_code_object();
  rj_waitcheck_options_t options{};
  ASSERT_EQ(rj_waitcheck_options_init(&options, sizeof(options)), ROCJITSU_STATUS_SUCCESS);
  options.check_counter_parity = 1;
  rj_waitcheck_result_t result = initialized_result();

  ASSERT_EQ(rj_waitcheck_analyze(image.data(), image.size(), &options, &result),
            ROCJITSU_STATUS_SUCCESS);
  EXPECT_EQ(result.target, ROCJITSU_WAITCHECK_TARGET_GFX1200);
  EXPECT_EQ(result.counter_parity_evaluated, 0u);
}

TEST(WaitcheckCApiTest, ReportsStructuredDiagnosticFromHazardousBuffer) {
  const auto image = rocjitsu::waitcheck_test::make_gfx1200_missing_wait_code_object();
  CallbackState state;
  const rj_waitcheck_options_t options = callback_options(state);
  rj_waitcheck_result_t result = initialized_result();

  ASSERT_EQ(rj_waitcheck_analyze(image.data(), image.size(), &options, &result),
            ROCJITSU_STATUS_SUCCESS);
  EXPECT_EQ(result.passed, 0u);
  EXPECT_EQ(result.struct_size, sizeof(result));
  EXPECT_EQ(result.abi_version, ROCJITSU_WAITCHECK_ABI_VERSION);
  EXPECT_EQ(result.target, ROCJITSU_WAITCHECK_TARGET_GFX1200);
  EXPECT_STREQ(rj_waitcheck_target_name(result.target), "gfx1200");
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
  EXPECT_EQ(diagnostic.struct_size, sizeof(rj_waitcheck_diagnostic_t));
  EXPECT_EQ(diagnostic.abi_version, ROCJITSU_WAITCHECK_ABI_VERSION);
  EXPECT_TRUE(diagnostic.has_kernel);
  EXPECT_EQ(diagnostic.kernel_name, "waitcheck");
  EXPECT_EQ(diagnostic.kernel_entry_offset, 0u);
  EXPECT_EQ(diagnostic.code, ROCJITSU_WAITCHECK_DIAGNOSTIC_WAIT_COUNTER);
  EXPECT_STREQ(rj_waitcheck_diagnostic_code_name(diagnostic.code), "wait-counter");
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

TEST(WaitcheckCApiTest, ReportsControlTransferHazardsThroughDedicatedAccess) {
  const auto image = rocjitsu::waitcheck_test::make_gfx1201_code_object(
      {rocjitsu::waitcheck_test::v_add_f32_e32_word(/*vdst=*/0, /*src0=*/2, /*vsrc1=*/0),
       rocjitsu::build_s_mov_b32(/*sdst=*/2, /*ssrc0=*/128, ROCJITSU_CODE_ARCH_RDNA4),
       rocjitsu::build_s_setpc_b64(/*ssrc0=*/8, ROCJITSU_CODE_ARCH_RDNA4)});
  CallbackState state;
  const rj_waitcheck_options_t options = callback_options(state);
  rj_waitcheck_result_t result = initialized_result();

  ASSERT_EQ(rj_waitcheck_analyze(image.data(), image.size(), &options, &result),
            ROCJITSU_STATUS_SUCCESS);
  EXPECT_EQ(result.target, ROCJITSU_WAITCHECK_TARGET_GFX1201);
  EXPECT_EQ(result.instructions_analyzed, 3u);
  EXPECT_EQ(result.passed, 0u);
  ASSERT_EQ(state.diagnostics.size(), 1u);

  const OwnedDiagnostic &diagnostic = state.diagnostics.front();
  EXPECT_EQ(diagnostic.code, ROCJITSU_WAITCHECK_DIAGNOSTIC_SGPR_DEPCTR);
  EXPECT_EQ(diagnostic.counter, ROCJITSU_WAITCHECK_COUNTER_DEPCTR);
  EXPECT_EQ(diagnostic.access, ROCJITSU_WAITCHECK_ACCESS_CONTROL_TRANSFER);
  EXPECT_EQ(diagnostic.reg.register_class, ROCJITSU_WAITCHECK_REGISTER_SGPR);
  EXPECT_EQ(diagnostic.reg.index, 2u);
  EXPECT_NE(diagnostic.instruction.find("s_setpc_b64"), std::string::npos);
  EXPECT_NE(diagnostic.message.find("control transfer"), std::string::npos);
  EXPECT_TRUE(state.errors.empty());
}

TEST(WaitcheckCApiTest, DiagnosticCodesAndNamesAreStable) {
  static_assert(ROCJITSU_WAITCHECK_DIAGNOSTIC_UNKNOWN == 0);
  static_assert(ROCJITSU_WAITCHECK_DIAGNOSTIC_WAIT_COUNTER == 1);
  static_assert(ROCJITSU_WAITCHECK_DIAGNOSTIC_SGPR_DEPCTR == 2);
  static_assert(ROCJITSU_WAITCHECK_DIAGNOSTIC_ASYNC_BARRIER_PRE_WAIT == 3);
  static_assert(ROCJITSU_WAITCHECK_DIAGNOSTIC_ASYNC_BARRIER_POST_WAIT == 4);
  static_assert(ROCJITSU_WAITCHECK_DIAGNOSTIC_VA_VDST == 5);
  static_assert(ROCJITSU_WAITCHECK_ACCESS_USE == 0);
  static_assert(ROCJITSU_WAITCHECK_ACCESS_DEF == 1);
  static_assert(ROCJITSU_WAITCHECK_ACCESS_MEMORY_ORDER == 2);
  static_assert(ROCJITSU_WAITCHECK_ACCESS_PROGRAM_END == 3);
  static_assert(ROCJITSU_WAITCHECK_ACCESS_INVALID == 4);
  static_assert(ROCJITSU_WAITCHECK_ACCESS_CONTROL_TRANSFER == 5);

  EXPECT_STREQ(rj_waitcheck_diagnostic_code_name(ROCJITSU_WAITCHECK_DIAGNOSTIC_UNKNOWN), "unknown");
  EXPECT_STREQ(rj_waitcheck_diagnostic_code_name(ROCJITSU_WAITCHECK_DIAGNOSTIC_WAIT_COUNTER),
               "wait-counter");
  EXPECT_STREQ(rj_waitcheck_diagnostic_code_name(ROCJITSU_WAITCHECK_DIAGNOSTIC_SGPR_DEPCTR),
               "sgpr-depctr");
  EXPECT_STREQ(
      rj_waitcheck_diagnostic_code_name(ROCJITSU_WAITCHECK_DIAGNOSTIC_ASYNC_BARRIER_PRE_WAIT),
      "async-barrier-pre-wait");
  EXPECT_STREQ(
      rj_waitcheck_diagnostic_code_name(ROCJITSU_WAITCHECK_DIAGNOSTIC_ASYNC_BARRIER_POST_WAIT),
      "async-barrier-post-wait");
  EXPECT_STREQ(rj_waitcheck_diagnostic_code_name(ROCJITSU_WAITCHECK_DIAGNOSTIC_VA_VDST), "va-vdst");
  EXPECT_STREQ(rj_waitcheck_diagnostic_code_name(static_cast<rj_waitcheck_diagnostic_code_t>(1234)),
               "unknown");

  static_assert(ROCJITSU_WAITCHECK_COUNTER_PARITY_UNKNOWN == 0);
  static_assert(ROCJITSU_WAITCHECK_COUNTER_PARITY_MODELED_UNDERACCOUNTING == 1);
  static_assert(ROCJITSU_WAITCHECK_COUNTER_PARITY_UNMODELED_EMITTED_WAIT == 2);
  EXPECT_STREQ(rj_waitcheck_counter_parity_kind_name(ROCJITSU_WAITCHECK_COUNTER_PARITY_UNKNOWN),
               "unknown");
  EXPECT_STREQ(rj_waitcheck_counter_parity_kind_name(
                   ROCJITSU_WAITCHECK_COUNTER_PARITY_MODELED_UNDERACCOUNTING),
               "modeled-counter-underaccounting");
  EXPECT_STREQ(rj_waitcheck_counter_parity_kind_name(
                   ROCJITSU_WAITCHECK_COUNTER_PARITY_UNMODELED_EMITTED_WAIT),
               "unmodeled-emitted-wait");
  EXPECT_STREQ(
      rj_waitcheck_counter_parity_kind_name(static_cast<rj_waitcheck_counter_parity_kind_t>(1234)),
      "unknown");
}

TEST(WaitcheckCApiTest, SupportsRdna35Targets) {
  for (const auto &[image, expected_target, expected_name] :
       std::array{std::tuple{rocjitsu::waitcheck_test::make_gfx1150_missing_wait_code_object(),
                             ROCJITSU_WAITCHECK_TARGET_GFX1150, "gfx1150"},
                  std::tuple{rocjitsu::waitcheck_test::make_gfx1151_missing_wait_code_object(),
                             ROCJITSU_WAITCHECK_TARGET_GFX1151, "gfx1151"}}) {
    CallbackState state;
    const rj_waitcheck_options_t options = callback_options(state);
    rj_waitcheck_result_t result = initialized_result();

    ASSERT_EQ(rj_waitcheck_analyze(image.data(), image.size(), &options, &result),
              ROCJITSU_STATUS_SUCCESS);
    EXPECT_EQ(result.target, expected_target);
    EXPECT_STREQ(rj_waitcheck_target_name(result.target), expected_name);
    EXPECT_EQ(result.passed, 0u);
    ASSERT_EQ(state.diagnostics.size(), 1u);
    EXPECT_EQ(state.diagnostics[0].code, ROCJITSU_WAITCHECK_DIAGNOSTIC_WAIT_COUNTER);
  }
}

TEST(WaitcheckCApiTest, CleanBufferPassesWithNullOptions) {
  const auto image = rocjitsu::waitcheck_test::make_gfx1200_correct_wait_code_object();
  rj_waitcheck_result_t result = initialized_result();

  ASSERT_EQ(rj_waitcheck_analyze(image.data(), image.size(), nullptr, &result),
            ROCJITSU_STATUS_SUCCESS);
  EXPECT_EQ(result.analysis_complete, 1u);
  EXPECT_EQ(result.incomplete_observations, 0u);
  EXPECT_EQ(result.passed, 1u);
  EXPECT_EQ(result.diagnostics_observed, 0u);
  EXPECT_EQ(result.diagnostics_reported, 0u);
}

TEST(WaitcheckCApiTest, HazardsStillFailWithoutDiagnosticCallback) {
  const auto image = rocjitsu::waitcheck_test::make_gfx1200_missing_wait_code_object();
  rj_waitcheck_result_t result = initialized_result();

  ASSERT_EQ(rj_waitcheck_analyze(image.data(), image.size(), nullptr, &result),
            ROCJITSU_STATUS_SUCCESS);
  EXPECT_EQ(result.analysis_complete, 1u);
  EXPECT_EQ(result.incomplete_observations, 0u);
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
  rj_waitcheck_result_t result = initialized_result();

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
  EXPECT_EQ(state.diagnostics.front().kernel_name, "hazardous");
  EXPECT_EQ(state.diagnostics.front().kernel_entry_offset, 0u);
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
  rj_waitcheck_result_t result = initialized_result();

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

TEST(WaitcheckCApiTest, AttributesWholeObjectDiagnosticsToEachKernel) {
  std::vector<uint32_t> hazardous;
  rocjitsu::waitcheck_test::append_inst(hazardous, rocjitsu::waitcheck_test::global_load_b32(0));
  rocjitsu::waitcheck_test::append_inst(hazardous, rocjitsu::waitcheck_test::v_mov_b32(1, 0));
  hazardous.push_back(0xBFB00000U); // s_endpgm
  const auto image = rocjitsu::waitcheck_test::make_gfx_multi_kernel_code_object(
      {{"first", hazardous}, {"second", hazardous}}, rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX1200);
  CallbackState state;
  const rj_waitcheck_options_t options = callback_options(state);
  rj_waitcheck_result_t result = initialized_result();

  ASSERT_EQ(rj_waitcheck_analyze(image.data(), image.size(), &options, &result),
            ROCJITSU_STATUS_SUCCESS);
  ASSERT_EQ(state.diagnostics.size(), 2u);
  EXPECT_EQ(state.diagnostics[0].kernel_name, "first");
  EXPECT_EQ(state.diagnostics[0].kernel_entry_offset, 0u);
  EXPECT_EQ(state.diagnostics[1].kernel_name, "second");
  EXPECT_EQ(state.diagnostics[1].kernel_entry_offset, hazardous.size() * sizeof(uint32_t));
}

TEST(WaitcheckCApiTest, SizedStructuresPreserveCallerExtensions) {
  struct ExtendedOptions {
    rj_waitcheck_options_t value;
    std::array<uint8_t, 32> extension;
  } options;
  std::memset(&options, 0xa5, sizeof(options));
  ASSERT_EQ(rj_waitcheck_options_init(&options.value, sizeof(options)), ROCJITSU_STATUS_SUCCESS);
  EXPECT_EQ(options.value.struct_size, sizeof(options));
  EXPECT_TRUE(std::ranges::all_of(options.extension, [](uint8_t byte) { return byte == 0; }));
  options.extension.fill(0xa5);

  struct ExtendedResult {
    rj_waitcheck_result_t value;
    std::array<uint8_t, 32> extension;
  } result;
  std::memset(&result, 0x5a, sizeof(result));
  ASSERT_EQ(rj_waitcheck_result_init(&result.value, sizeof(result)), ROCJITSU_STATUS_SUCCESS);
  EXPECT_EQ(result.value.struct_size, sizeof(result));
  EXPECT_TRUE(std::ranges::all_of(result.extension, [](uint8_t byte) { return byte == 0; }));
  result.extension.fill(0x5a);

  const auto image = rocjitsu::waitcheck_test::make_gfx1200_correct_wait_code_object();
  ASSERT_EQ(rj_waitcheck_analyze(image.data(), image.size(), &options.value, &result.value),
            ROCJITSU_STATUS_SUCCESS);
  EXPECT_TRUE(std::ranges::all_of(options.extension, [](uint8_t byte) { return byte == 0xa5; }));
  EXPECT_TRUE(std::ranges::all_of(result.extension, [](uint8_t byte) { return byte == 0x5a; }));
}

TEST(WaitcheckCApiTest, ConsumerCallbackExceptionsDoNotChangeAnalysisStatus) {
  const auto hazardous = rocjitsu::waitcheck_test::make_gfx1200_missing_wait_code_object();
  rj_waitcheck_options_t diagnostic_options{};
  ASSERT_EQ(rj_waitcheck_options_init(&diagnostic_options, sizeof(diagnostic_options)),
            ROCJITSU_STATUS_SUCCESS);
  diagnostic_options.diagnostic_callback = throw_from_diagnostic;
  rj_waitcheck_result_t diagnostic_result = initialized_result();
  ASSERT_EQ(rj_waitcheck_analyze(hazardous.data(), hazardous.size(), &diagnostic_options,
                                 &diagnostic_result),
            ROCJITSU_STATUS_SUCCESS);
  EXPECT_EQ(diagnostic_result.diagnostics_reported, 1u);
  EXPECT_EQ(diagnostic_result.passed, 0u);

  const auto parity = rocjitsu::waitcheck_test::make_gfx_multi_kernel_code_object(
      {{"strong_wait",
        {0xE0501000u, 0x80000008u, 0xE05D1000u, 0x80100008u, 0xBF8C0F70u, 0x7E020300u,
         0xBF810000u}}},
      rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX950);
  rj_waitcheck_options_t parity_options{};
  ASSERT_EQ(rj_waitcheck_options_init(&parity_options, sizeof(parity_options)),
            ROCJITSU_STATUS_SUCCESS);
  parity_options.check_counter_parity = 1;
  parity_options.counter_parity_callback = throw_from_counter_parity;
  rj_waitcheck_result_t parity_result = initialized_result();
  ASSERT_EQ(rj_waitcheck_analyze(parity.data(), parity.size(), &parity_options, &parity_result),
            ROCJITSU_STATUS_SUCCESS);
  EXPECT_EQ(parity_result.counter_parity_diagnostics_reported, 1u);
  EXPECT_EQ(parity_result.counter_parity_evaluated, 1u);
}

TEST(WaitcheckCApiTest, RejectsUninitializedOrWrongVersionStructures) {
  rj_waitcheck_options_t options{};
  rj_waitcheck_result_t result{};
  const auto image = rocjitsu::waitcheck_test::make_gfx1200_correct_wait_code_object();

  EXPECT_EQ(rj_waitcheck_options_init(&options, offsetof(rj_waitcheck_options_t, user_data)),
            ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(rj_waitcheck_result_init(&result, offsetof(rj_waitcheck_result_t, stopped_early)),
            ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(rj_waitcheck_analyze(image.data(), image.size(), nullptr, &result),
            ROCJITSU_STATUS_INVALID_ARGUMENT);

  ASSERT_EQ(rj_waitcheck_options_init(&options, sizeof(options)), ROCJITSU_STATUS_SUCCESS);
  ASSERT_EQ(rj_waitcheck_result_init(&result, sizeof(result)), ROCJITSU_STATUS_SUCCESS);
  options.abi_version = ROCJITSU_WAITCHECK_ABI_VERSION + 1;
  EXPECT_EQ(rj_waitcheck_analyze(image.data(), image.size(), &options, &result),
            ROCJITSU_STATUS_INVALID_ARGUMENT);
}

TEST(WaitcheckCApiTest, RetainsInferredTargetWhenTargetIsUnsupported) {
  const auto image = rocjitsu::waitcheck_test::make_gfx_code_object(
      {0xBFB00000U}, rocjitsu::EF_AMDGPU_MACH_AMDGCN_GFX90A);
  rj_waitcheck_result_t result = initialized_result();

  EXPECT_EQ(rj_waitcheck_analyze(image.data(), image.size(), nullptr, &result),
            ROCJITSU_STATUS_INVALID_CODE_OBJECT);
  EXPECT_EQ(result.target, ROCJITSU_WAITCHECK_TARGET_GFX90A);
  EXPECT_STREQ(rj_waitcheck_target_name(result.target), "gfx90a");
  EXPECT_STREQ(rj_waitcheck_target_name(static_cast<rj_waitcheck_target_t>(1234)), "unknown");
}

TEST(WaitcheckCApiTest, ReturnsUnsupportedWithPopulatedResultWhenAnalysisIsIncomplete) {
  const auto image = rocjitsu::waitcheck_test::make_gfx950_incomplete_direct_to_lds_code_object(
      /*include_definite_wait_hazard=*/true);
  CallbackState state;
  rj_waitcheck_options_t options = callback_options(state);
  rj_waitcheck_result_t result = initialized_result();

  const rj_status_t status = rj_waitcheck_analyze(image.data(), image.size(), &options, &result);

  EXPECT_EQ(status, ROCJITSU_STATUS_UNSUPPORTED);
  EXPECT_EQ(result.target, ROCJITSU_WAITCHECK_TARGET_GFX950);
  EXPECT_EQ(result.analysis_complete, 0u);
  EXPECT_EQ(result.incomplete_observations, 1u);
  EXPECT_EQ(result.passed, 0u);
  EXPECT_EQ(result.diagnostics_observed, 1u);
  EXPECT_EQ(result.diagnostics_reported, 1u);
  ASSERT_EQ(state.diagnostics.size(), 1u);
  EXPECT_EQ(state.diagnostics.front().code, ROCJITSU_WAITCHECK_DIAGNOSTIC_WAIT_COUNTER);
  ASSERT_EQ(state.errors.size(), 1u);
  EXPECT_NE(state.errors.front().find("direct-to-LDS producer range is not statically known"),
            std::string::npos);
}

TEST(WaitcheckCApiTest, AcceptsLegacyResultAllocationBeforeCompletenessFields) {
  constexpr size_t legacy_result_size = offsetof(rj_waitcheck_result_t, analysis_complete);
  rj_waitcheck_result_t result{};
  ASSERT_EQ(rj_waitcheck_result_init(&result, legacy_result_size), ROCJITSU_STATUS_SUCCESS);
  const auto image = rocjitsu::waitcheck_test::make_gfx1200_correct_wait_code_object();

  EXPECT_EQ(rj_waitcheck_analyze(image.data(), image.size(), nullptr, &result),
            ROCJITSU_STATUS_SUCCESS);
  EXPECT_EQ(result.target, ROCJITSU_WAITCHECK_TARGET_GFX1200);
  EXPECT_EQ(result.passed, 1u);
}

TEST(WaitcheckCApiTest, IndependentCallsAreConcurrentAndReentrant) {
  constexpr size_t kThreadCount = 8;
  constexpr size_t kIterations = 8;
  const auto image = rocjitsu::waitcheck_test::make_gfx1200_missing_wait_code_object();
  std::atomic<size_t> failures = 0;
  std::array<std::thread, kThreadCount> threads;

  for (std::thread &thread : threads) {
    thread = std::thread([&] {
      for (size_t iteration = 0; iteration < kIterations; ++iteration) {
        rj_waitcheck_result_t result;
        if (rj_waitcheck_result_init(&result, sizeof(result)) != ROCJITSU_STATUS_SUCCESS ||
            rj_waitcheck_analyze(image.data(), image.size(), nullptr, &result) !=
                ROCJITSU_STATUS_SUCCESS ||
            result.passed != 0 || result.diagnostics_observed != 1 ||
            result.target != ROCJITSU_WAITCHECK_TARGET_GFX1200) {
          ++failures;
        }
      }
    });
  }
  for (std::thread &thread : threads)
    thread.join();

  EXPECT_EQ(failures.load(), 0u);
}

TEST(WaitcheckCApiTest, RejectsMalformedBufferAndUnknownKernelOffset) {
  const uint8_t malformed[] = {0x7f, 'E', 'L', 'F'};
  CallbackState state;
  rj_waitcheck_options_t options = callback_options(state);
  rj_waitcheck_result_t result = initialized_result();

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
  rj_waitcheck_result_t result = initialized_result();
  EXPECT_EQ(rj_waitcheck_analyze(nullptr, 1, nullptr, &result), ROCJITSU_STATUS_INVALID_ARGUMENT);
  const uint8_t byte = 0;
  EXPECT_EQ(rj_waitcheck_analyze(&byte, 0, nullptr, &result), ROCJITSU_STATUS_INVALID_ARGUMENT);
  EXPECT_EQ(rj_waitcheck_analyze(&byte, 1, nullptr, nullptr), ROCJITSU_STATUS_INVALID_ARGUMENT);
}

} // namespace
