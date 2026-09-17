// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_report.h
/// @brief ConSan report ABI, evidence sizing, and host-model contracts.

#pragma once

#include "rocjitsu/code/patch/consan/consan.h"
#include "rocjitsu/code/patch/consan/consan_abi.h"
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

namespace rocjitsu::consan {

#include "rocjitsu/code/patch/consan/consan_core_types.h.inc"

#include "rocjitsu/code/patch/consan/consan_report_helpers.h.inc"

#include "rocjitsu/code/patch/consan/consan_results.h.inc"

#include "rocjitsu/code/patch/consan/consan_report_contract.h.inc"

#include "rocjitsu/code/patch/consan/consan_model.h.inc"

#include "rocjitsu/code/patch/consan/consan_report_layout.h.inc"

} // namespace rocjitsu::consan
