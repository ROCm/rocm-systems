// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <chrono>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys::output
{

// Maximum GPU-id tokens rendered before the list is truncated.
inline constexpr std::size_t MAX_RENDERED_GPU_IDS = 16;

// Display column count assuming every UTF-8 code point is one column
// wide (true for ASCII plus the box-drawing and bullet glyphs the
// summary uses). Counts UTF-8 lead bytes, ignoring continuation bytes.
[[nodiscard]] std::size_t
display_width(std::string_view text);

// Concatenates `count` copies of `glyph`. Used to draw box rules.
[[nodiscard]] std::string
repeat_glyph(std::string_view glyph, std::size_t count);

// Condenses a full command line to a short program label: the basename
// of the first whitespace-delimited token, terminal control characters
// stripped. Empty input returns empty.
[[nodiscard]] std::string
summarize_command(std::string_view command);

// Renders a sort-unique GPU-id list as `:N`, compressing contiguous
// runs of length >= 3 into `:min-max`. Empty input returns empty.
// Truncates rendered output to MAX_RENDERED_GPU_IDS tokens.
[[nodiscard]] std::string
format_gpu_ids(const std::vector<int>& gpu_ids);

// Two-decimal seconds; "?" for non-positive durations.
[[nodiscard]] std::string
format_duration(std::chrono::nanoseconds dur);

// Drops C0 control bytes, DEL, and CSI escape sequences (ESC [ ... letter)
// so peer-controlled strings cannot inject terminal effects when rendered
// to stdout.
[[nodiscard]] std::string
strip_terminal_control_chars(std::string_view s);

}  // namespace rocprofsys::output
