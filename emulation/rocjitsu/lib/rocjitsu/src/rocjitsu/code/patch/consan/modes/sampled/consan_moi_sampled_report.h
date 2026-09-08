// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_sampled_report.h
/// @brief Sampled report geometry and host-model contract.

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_report_common_contract.h"

namespace rocjitsu {

#include "rocjitsu/code/patch/consan/modes/sampled/consan_moi_sampled_results.h.inc"

#include "rocjitsu/code/patch/consan/modes/sampled/consan_moi_sampled_report_contract.h.inc"

#include "rocjitsu/code/patch/consan/modes/sampled/consan_moi_sampled_model.h.inc"

} // namespace rocjitsu
