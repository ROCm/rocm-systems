// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/diagnostic/stacktrace.hpp"

#include <exception>
#include <optional>
#include <string>

namespace rocprofsys
{
inline namespace common
{
namespace diagnostic
{
struct exception_format_options
{
    /// Stacktrace formatting options. `trace_options.with_color` is taken as
    /// the explicit override when set; if left at the default and `with_color`
    /// below is `nullopt`, `format_exception` resolves the effective color
    /// from the environment (`NO_COLOR` flips the default to off).
    stacktrace::format_options trace_options{};

    /// Optional explicit color override. When set, takes precedence over
    /// `NO_COLOR` and any other defaulting. When unset, `format_exception`
    /// uses `true` unless `NO_COLOR` is present in the environment.
    std::optional<bool> with_color;

    /// When true, capture a fresh trace at format-time. Phase 1 always
    /// uses this (no exception type embeds a trace yet).
    bool capture_trace_at_call_site = true;
};

/// Render an exception (`what()` + stacktrace) wrapped in a side-bar block
/// labeled `error`, per the format spec.
std::string
format_exception(const std::exception& e, exception_format_options opt = {});

/// Print to stderr. Color defaults to on; `NO_COLOR` env disables.
void
print_exception(const std::exception& e);
}  // namespace diagnostic
}  // namespace common
}  // namespace rocprofsys
