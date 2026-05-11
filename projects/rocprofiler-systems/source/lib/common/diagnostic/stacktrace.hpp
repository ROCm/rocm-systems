// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace rocprofsys
{
inline namespace common
{
namespace diagnostic
{
/// One resolved entry in a stack trace.
///
/// Fields may be empty when symbol resolution fails. `address` is always
/// populated (it's the raw value captured by libunwind / `_Unwind_Backtrace`).
struct stacktrace_frame
{
    std::uintptr_t address = 0;
    std::string    module;    ///< basename of the .so / executable
    std::string    function;  ///< demangled name (mangled if demangle fails)
    std::string    file;      ///< source path (relative if under PROJECT_SOURCE_DIR)
    std::uint32_t  line    = 0;
    std::uint32_t  column  = 0;  ///< 0 = not available (libdw rarely provides it)
    std::uint32_t  offset  = 0;  ///< PC offset from start of resolved symbol
    bool           inlined = false;
};

enum class trace_color_mode
{
    off,
    on,
    auto_detect
};

struct trace_format_options
{
    trace_color_mode color               = trace_color_mode::auto_detect;
    bool             with_module         = true;
    bool             with_offset         = false;
    bool             with_file_line      = true;
    bool             with_inlined        = true;
    bool             with_source_excerpt = false;
    // Empty here means "use default_skip_filters() at to_string time".
    // Set to an explicit list to override.
    std::vector<std::string> skip_substrings;
    std::size_t              max_function_width = 100;
    std::size_t              max_frames_shown   = 32;
};

/// Captured stack with lazy symbolization.
///
/// `capture` records raw return-addresses inline. `frames()` triggers the
/// (potentially expensive) libdw + demangle pass on first call and caches.
class stacktrace
{
public:
    using color_mode     = trace_color_mode;
    using format_options = trace_format_options;

    /// Capture the current call stack.
    ///
    /// @param skip_frames How many of the top frames to drop. The frame for
    ///        `capture()` itself is always dropped, so a caller passing 0
    ///        gets its own frame at the top.
    /// @param max_frames Hard cap on collected frames.
    static stacktrace capture(int skip_frames = 0, int max_frames = 64) noexcept;

    bool empty() const noexcept;

    /// Raw addresses, in capture order (top-of-stack first).
    std::pair<const std::uintptr_t*, std::size_t> raw() const noexcept;

    /// Resolved frames. Symbolizes lazily on first call.
    const std::vector<stacktrace_frame>& frames() const;

    /// Render the trace as a string per the format spec.
    std::string to_string(format_options opt = {}) const;

    /// Substring filters that drop common noise frames by default.
    static const std::vector<std::string>& default_skip_filters();

private:
    std::vector<std::uintptr_t>                          m_raw;
    mutable std::optional<std::vector<stacktrace_frame>> m_frames;
};
}  // namespace diagnostic
}  // namespace common
}  // namespace rocprofsys
