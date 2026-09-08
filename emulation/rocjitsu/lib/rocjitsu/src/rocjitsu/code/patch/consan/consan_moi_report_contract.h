// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_report_contract.h
/// @brief Explicit aggregation facade for all MOI report modes.

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_report_common_contract.h"
#include "rocjitsu/code/patch/consan/modes/inline_shadow/consan_moi_inline_shadow_report.h"
#include "rocjitsu/code/patch/consan/modes/record_replay/consan_moi_record_replay_report.h"
#include "rocjitsu/code/patch/consan/modes/sampled/consan_moi_sampled_report.h"

namespace rocjitsu {

// Only cross-mode sizing and the closed evidence-requirement variant belong
// here. Mode-local readers, emitters, and tests include their mode header.
#include "rocjitsu/code/patch/consan/consan_moi_report_layout.h.inc"

} // namespace rocjitsu
