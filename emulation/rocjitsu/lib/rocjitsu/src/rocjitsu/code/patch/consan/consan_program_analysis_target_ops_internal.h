// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file consan_program_analysis_target_ops_internal.h
/// @brief Target-owned program-analysis facet registrations.

#pragma once

#include "rocjitsu/code/patch/consan/consan_program_analysis_target_ops.h"

namespace rocjitsu {

extern const ConSanProgramAnalysisTargetOperations kConSanGfx9CdnaProgramAnalysisOperations;
extern const ConSanProgramAnalysisTargetOperations kConSanGfx1100ProgramAnalysisOperations;
extern const ConSanProgramAnalysisTargetOperations kConSanGfx1201ProgramAnalysisOperations;
extern const ConSanProgramAnalysisTargetOperations kConSanGfx1250ProgramAnalysisOperations;

} // namespace rocjitsu
