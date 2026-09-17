// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_window_emission.h
/// @brief ConSan-mode causal-window validation emission.

#pragma once

#include "rocjitsu/code/patch/consan/consan_internal.h"
#include "rocjitsu/code/patch/consan/consan_report_emission.h"
#include "rocjitsu/code/patch/instruction_sequence.h"

namespace rocjitsu::consan::detail {

/// Complete identity and register contract for validating a selected ConSan
/// causal window. Every failed field comparison narrows EXEC and branches to
/// mismatch_label when no lanes remain.
struct CausalWindowValidationRequest {
  uint64_t generation = 0;
  detail::ReportDispatchIdSource dispatch_id;
  const WorkgroupSources &workgroup_sources;
  uint16_t address_vgpr = 0;
  uint16_t value_vgpr = 0;
  uint16_t expected_vgpr = 0;
  uint16_t epoch_vgpr = 0;
  uint16_t first_entry_vgpr = 0;
  uint16_t temporary_exec_sgpr = 0;
  InstructionSequence::Label mismatch_label = 0;
  rj_code_arch_t arch{};
};

[[nodiscard]] bool append_causal_window_validation(std::vector<uint32_t> &words,
                                                   InstructionSequence &sequence,
                                                   const CausalWindowValidationRequest &request);

} // namespace rocjitsu::consan::detail
