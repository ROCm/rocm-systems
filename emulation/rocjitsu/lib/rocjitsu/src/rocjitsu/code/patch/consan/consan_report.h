// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_report.h
/// @brief ConSan report geometry and host-model contract.

#pragma once

#include "rocjitsu/code/patch/consan/consan_report_common_contract.h"

namespace rocjitsu::consan {

#include "rocjitsu/code/patch/consan/consan_results.h.inc"

#include "rocjitsu/code/patch/consan/consan_report_contract.h.inc"

#include "rocjitsu/code/patch/consan/consan_model.h.inc"

} // namespace rocjitsu::consan
