// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi_inline_shadow_report.h
/// @brief InlineShadow report geometry and host-model contract.

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_report_common_contract.h"

namespace rocjitsu {

#include "rocjitsu/code/patch/consan/modes/inline_shadow/consan_moi_inline_model.h.inc"

#include "rocjitsu/code/patch/consan/modes/inline_shadow/consan_moi_inline_exact_model.h.inc"

#include "rocjitsu/code/patch/consan/modes/inline_shadow/consan_moi_inline_shadow_report_contract.h.inc"

} // namespace rocjitsu
