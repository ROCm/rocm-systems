// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/defines.h"
#include "core/timemory.hpp"

#include <cstdint>

namespace rocprofsys
{

/// Pass A (Option B): emit span begin/end samples with explicit ids + parent ids when
/// ROCPROFSYS_PROFILE is enabled. Mirrors nesting semantics of timemory instrumentation.
void ROCPROFSYS_HIDDEN_API
wall_clock_span_push_region(tim::hash_value_t name_hash, std::string_view label,
                            std::string_view category);

void ROCPROFSYS_HIDDEN_API
wall_clock_span_pop_region(tim::hash_value_t name_hash);

/// Emit span-end samples for any regions still open on **this** thread (e.g. at process
/// shutdown). Call before trace-cache buffer shutdown so Pass B can pair begins/ends.
void ROCPROFSYS_HIDDEN_API
wall_clock_span_flush_open_regions();

}  // namespace rocprofsys
