// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

namespace rocprofsys
{
inline namespace common
{
namespace diagnostic
{
namespace color
{
constexpr const char* reset      = "\033[0m";
constexpr const char* bold       = "\033[1m";
constexpr const char* dim        = "\033[2m";
constexpr const char* red        = "\033[31m";
constexpr const char* green      = "\033[32m";
constexpr const char* yellow     = "\033[33m";
constexpr const char* blue       = "\033[34m";
constexpr const char* magenta    = "\033[35m";
constexpr const char* cyan       = "\033[36m";
constexpr const char* gray       = "\033[90m";
constexpr const char* bright_red = "\033[91m";

// Composite styles used by the format spec.
constexpr const char* error_kw    = "\033[1;91m";
constexpr const char* fn_name     = "\033[36m";
constexpr const char* file_path   = "\033[90m";
constexpr const char* line_num    = "\033[33m";
constexpr const char* tag         = "\033[90m";
constexpr const char* keyword_dim = "\033[2m";
constexpr const char* frame_idx   = "\033[90m";
constexpr const char* trailer     = "\033[2m";
constexpr const char* box_border  = "\033[90m";

/// True iff `fd` refers to a TTY.
bool
stderr_is_tty() noexcept;
bool
stdout_is_tty() noexcept;

/// True iff the `NO_COLOR` environment variable is set (any value).
/// Public entry points use this to flip their default `with_color` to false.
bool
no_color_env() noexcept;
}  // namespace color
}  // namespace diagnostic
}  // namespace common
}  // namespace rocprofsys
