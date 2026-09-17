// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_access_classifier.h
/// @brief Exact target classification for decoded ConSan accesses.

#pragma once

#include "rocjitsu/code/patch/consan/consan_access_shape.h"
#include "rocjitsu/code/rj_code.h"

namespace rocjitsu::consan {

struct ProgramSite;

/// Normalize one completed inventory access and classify the exact target
/// operations that can consume it. This is the sole access-form admission
/// authority; semantic policy and emitters consume its typed result.
[[nodiscard]] AccessLoweringClassification classify_access_lowering(const ProgramSite &access,
                                                                    rj_code_arch_t arch);

} // namespace rocjitsu::consan
