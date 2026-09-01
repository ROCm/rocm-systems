// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_program_analysis_target_ops_internal.h
/// @brief Target-owned program-analysis facet registrations.

#pragma once

#include "rocjitsu/code/patch/consan/consan_program_analysis_target_ops.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"

namespace rocjitsu {

namespace consan_program_analysis_target_detail {

[[nodiscard]] inline ConSanWaitInstructionEncoding classify_target_wait_instruction(
    uint32_t word, rj_code_arch_t arch, std::optional<uint32_t> extra_release,
    bool store_waits_are_release_boundaries, bool bounded_release_counter_form) {
  namespace ib = instrumentation;
  ConSanWaitInstructionEncoding result{.bounded_release_counter_form =
                                           bounded_release_counter_form};
  const auto matches = [&](const std::optional<uint32_t> &encoding) {
    return encoding && word == *encoding;
  };
  const bool load = matches(ib::build_s_wait_global_load0(arch));
  const bool store = matches(ib::build_s_wait_global_store0(arch));
  const bool flat_load = matches(ib::build_s_wait_flat_load0(arch));
  const bool flat_store = matches(ib::build_s_wait_flat_store0(arch));
  const bool lds = matches(ib::build_s_wait_lds0(arch));
  const bool release =
      matches(extra_release) || (store_waits_are_release_boundaries && (store || flat_store));
  result.drains_load = load || flat_load;
  result.drains_store = store || flat_store || release;
  result.drains_lds = flat_load || flat_store || lds;
  result.release_boundary = release;
  return result;
}

} // namespace consan_program_analysis_target_detail

extern const ConSanProgramAnalysisTargetOperations kConSanGfx9CdnaProgramAnalysisOperations;
extern const ConSanProgramAnalysisTargetOperations kConSanGfx1100ProgramAnalysisOperations;
extern const ConSanProgramAnalysisTargetOperations kConSanGfx1201ProgramAnalysisOperations;
extern const ConSanProgramAnalysisTargetOperations kConSanGfx1250ProgramAnalysisOperations;

} // namespace rocjitsu
