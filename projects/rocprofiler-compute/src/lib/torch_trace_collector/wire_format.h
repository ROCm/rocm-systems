// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "stack_entry.h"

#include <string>
#include <vector>

namespace torch_trace_collector::detail
{

// Percent-encoding of the two characters that would otherwise collide with the
// marker-path grammar. The Python side mirrors this table: encode_marker_name
// in utils/inject_roctx/core.py and decode_marker_name in
// utils/utils_analysis.py.
inline constexpr const char* kEncodedPercent = "%25";
inline constexpr const char* kEncodedSlash   = "%2F";

// Renders the stack as "marker1/.../markerN:context1/.../contextN". Marker names
// are percent-encoded so an embedded '/' is not read as the frame separator.
std::string build_marker_string(const std::vector<StackEntry>& stack);

}  // namespace torch_trace_collector::detail
