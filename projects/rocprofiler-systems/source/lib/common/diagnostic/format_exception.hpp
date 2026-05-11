// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/diagnostic/stacktrace.hpp"

#include <exception>
#include <string>

namespace rocprofsys
{
inline namespace common
{
namespace diagnostic
{
struct exception_format_options
{
    stacktrace::format_options trace_options{};
    /// When true, capture a fresh trace at format-time. Phase 1 always
    /// uses this (no exception type embeds a trace yet).
    bool capture_trace_at_call_site = true;
};

/// Render an exception as `error: <what>` followed by a stacktrace,
/// per the format spec.
std::string
format_exception(const std::exception& e, exception_format_options opt = {});

/// Print to stderr with auto color detection.
void
print_exception(const std::exception& e);
}  // namespace diagnostic
}  // namespace common
}  // namespace rocprofsys
