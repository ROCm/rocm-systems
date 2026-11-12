// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include <string>
#include <string_view>

namespace rocprofiler_compute_tool
{

/// Drops a trailing ".kd" and demangles.
std::string format_kernel_name(const char* mangled_name);

/// Extracts the bare kernel name from a demangled one:
/// 'Foo<int, float>::foo(a[], int (int))' -> 'foo'.
std::string truncate_name(std::string_view name);

}  // namespace rocprofiler_compute_tool
