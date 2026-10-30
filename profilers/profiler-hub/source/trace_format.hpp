// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "profiler-hub/c/profiler_hub_types.h"

#include <string>

namespace profiler_hub
{
ph_trace_format_t
detect_trace_format(const std::string& path);
}  // namespace profiler_hub
