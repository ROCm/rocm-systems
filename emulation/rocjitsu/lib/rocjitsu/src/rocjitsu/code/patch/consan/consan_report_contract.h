// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_report_contract.h
/// @brief Explicit aggregation facade for all ConSan report modes.

#pragma once

#include "rocjitsu/code/patch/consan/consan_report.h"
#include "rocjitsu/code/patch/consan/consan_report_common_contract.h"

namespace rocjitsu::consan {

// Only cross-mode sizing and the closed evidence-requirement variant belong
// here. Mode-local readers, emitters, and tests include their mode header.
#include "rocjitsu/code/patch/consan/consan_report_layout.h.inc"

} // namespace rocjitsu::consan
