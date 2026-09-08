// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_record_replay_report.h
/// @brief Record/Replay report geometry and host-model contract.

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_report_common_contract.h"

namespace rocjitsu {

#include "rocjitsu/code/patch/consan/modes/record_replay/consan_moi_record_replay_types.h.inc"

#include "rocjitsu/code/patch/consan/modes/record_replay/consan_moi_record_replay_report_layout.h.inc"

#include "rocjitsu/code/patch/consan/modes/record_replay/consan_moi_record_replay_results.h.inc"

#include "rocjitsu/code/patch/consan/modes/record_replay/consan_moi_record_replay_sparse_shadow.h.inc"

#include "rocjitsu/code/patch/consan/modes/record_replay/consan_moi_record_replay_report_contract.h.inc"

#include "rocjitsu/code/patch/consan/modes/record_replay/consan_moi_record_replay_model.h.inc"

} // namespace rocjitsu
