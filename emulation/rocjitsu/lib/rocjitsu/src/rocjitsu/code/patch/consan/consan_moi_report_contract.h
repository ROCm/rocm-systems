// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_report_contract.h
/// @brief Shared MOI evidence, report ABI, and host-analysis contract.

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/consan/consan_moi_abi.h"
#include "rocjitsu/code/rj_code.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <string_view>
#include <variant>
#include <vector>

// These fragments declare the report-side contract in dependency order:
// shared ABI and mechanisms first, then exact-subset and mode-owned models.
// None exposes lowerer state, patch proof, or resource placement.
#include "rocjitsu/code/patch/consan/consan_moi_core_types.h.inc"

#include "rocjitsu/code/patch/consan/consan_moi_record_replay_types.h.inc"

#include "rocjitsu/code/patch/consan/consan_moi_inline_model.h.inc"

#include "rocjitsu/code/patch/consan/consan_moi_report_layout.h.inc"

#include "rocjitsu/code/patch/consan/consan_moi_engine_results.h.inc"

#include "rocjitsu/code/patch/consan/consan_moi_report_helpers.h.inc"

#include "rocjitsu/code/patch/consan/consan_moi_record_replay_report_contract.h.inc"

#include "rocjitsu/code/patch/consan/consan_moi_sampled_report_contract.h.inc"

#include "rocjitsu/code/patch/consan/consan_moi_shadow_common.h.inc"

#include "rocjitsu/code/patch/consan/consan_moi_exact_shadow_model.h.inc"

#include "rocjitsu/code/patch/consan/consan_moi_inline_exact_model.h.inc"

#include "rocjitsu/code/patch/consan/consan_moi_inline_shadow_report_contract.h.inc"

#include "rocjitsu/code/patch/consan/consan_moi_record_replay_model.h.inc"

#include "rocjitsu/code/patch/consan/consan_moi_sampled_model.h.inc"

} // namespace rocjitsu
