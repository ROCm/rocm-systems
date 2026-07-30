// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_moi.h
/// @brief MOI instrumentation mode entry points for ConSan DBI patching.

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
#include <span>
#include <string_view>
#include <vector>

#include "rocjitsu/code/patch/consan/consan_moi_core_types.h.inc"

#include "rocjitsu/code/patch/consan/consan_moi_record_replay_types.h.inc"

#include "rocjitsu/code/patch/consan/consan_moi_inline_model.h.inc"

#include "rocjitsu/code/patch/consan/consan_moi_report_layout.h.inc"

#include "rocjitsu/code/patch/consan/consan_moi_engine_results.h.inc"

#include "rocjitsu/code/patch/consan/consan_moi_report_helpers.h.inc"

#include "rocjitsu/code/patch/consan/consan_moi_shadow_models.h.inc"

} // namespace rocjitsu
