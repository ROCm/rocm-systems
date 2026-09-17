// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_report_common_contract.h
/// @brief Mode-independent ConSan report ABI and host-model primitives.

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

#include "rocjitsu/code/patch/consan/consan_report_common_layout.h.inc"

#include "rocjitsu/code/patch/consan/consan_report_helpers.h.inc"

#include "rocjitsu/code/patch/consan/consan_shadow_common.h.inc"

} // namespace rocjitsu::consan
