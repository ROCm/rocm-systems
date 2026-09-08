// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_final_validation_internal.h"

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/consan/consan_instruction_semantics.h"
#include "rocjitsu/code/patch/consan/consan_moi_internal.h"
#include "rocjitsu/code/patch/consan/modes/inline_shadow/consan_moi_inline_register_layout.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"
#include "rocjitsu/isa/instruction.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>

namespace rocjitsu::consan_validation_detail {
namespace {

#include "consan_moi_inline_shadow_final_validation.inc"

} // namespace

void validate_inline_shadow_final_semantics(const FinalValidationEnvironment &environment,
                                            const ConSanFinalValidationInput &result,
                                            uint64_t expected_moi_report_dispatch_id,
                                            std::vector<std::string> &errors) {
  validate_inline_exact_shadow_semantics(environment, result, expected_moi_report_dispatch_id,
                                         errors);
  validate_inline_release_transaction_semantics(environment, result, errors);
}

} // namespace rocjitsu::consan_validation_detail
