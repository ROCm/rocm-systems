// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Demo program for the rocprofsys diagnostic format.
//
// Two demonstration paths:
//
//  PART A - "throw-site capture": captures the trace at the deep call site
//           BEFORE throwing. This is the model the future custom
//           rocprofsys::exception will use - the trace travels with the
//           exception object. Phase 1 simulates it by capturing into a
//           local stacktrace and rendering directly.
//
//  PART B - "catch-site capture": format_exception() captures at the catch
//           site. Stack already unwound; we only see the catcher's frame.
//           Useful for legacy std::exception that doesn't carry a trace.
//
// Both paths exercise the format-options permutations.

#include "common/diagnostic/color.hpp"
#include "common/diagnostic/format_exception.hpp"
#include "common/diagnostic/stacktrace.hpp"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

using rocprofsys::common::diagnostic::exception_format_options;
using rocprofsys::common::diagnostic::format_exception;
using rocprofsys::common::diagnostic::stacktrace;

namespace
{
[[gnu::noinline]] stacktrace
parse_metric_value(const std::string& s)
{
    asm volatile("");
    if(s.empty() || s[0] == 'b')
    {
        // Capture HERE - simulating the future custom exception ctor.
        return stacktrace::capture();
    }
    return stacktrace::capture();
}

[[gnu::noinline]] stacktrace
collector_setup(const std::string& s)
{
    asm volatile("");
    return parse_metric_value(s);
}

[[gnu::noinline]] stacktrace
sampler_setup(const std::string& s)
{
    asm volatile("");
    return collector_setup(s);
}

[[gnu::noinline]] stacktrace
library_initialize(const std::string& s)
{
    asm volatile("");
    return sampler_setup(s);
}

[[gnu::noinline]] void
parse_throw(const std::string& s)
{
    asm volatile("");
    if(s.empty() || s[0] == 'b')
    {
        throw std::runtime_error{ "Failed to parse PMC metric '" + s +
                                  "': value out of range" };
    }
}

[[gnu::noinline]] void
collector_throw(const std::string& s)
{
    asm volatile("");
    parse_throw(s);
}

[[gnu::noinline]] void
sampler_throw(const std::string& s)
{
    asm volatile("");
    collector_throw(s);
}

[[gnu::noinline]] void
library_throw(const std::string& s)
{
    asm volatile("");
    sampler_throw(s);
}

void
banner(const char* title)
{
    std::printf("\n==================================================\n");
    std::printf("%s\n", title);
    std::printf("==================================================\n");
}

void
render_trace(const stacktrace& trace, stacktrace::format_options opt,
             const std::string& msg, const char* title)
{
    banner(title);
    bool color_on = opt.color == stacktrace::color_mode::on ||
                    (opt.color == stacktrace::color_mode::auto_detect &&
                     rocprofsys::common::diagnostic::color::color_supported_for(1));
    if(color_on)
    {
        std::printf("\033[1;91merror:\033[0m %s\n\n", msg.c_str());
    }
    else
    {
        std::printf("error: %s\n\n", msg.c_str());
    }
    auto s = trace.to_string(opt);
    std::fputs(s.c_str(), stdout);
}

void
render_caught(const std::exception& e, exception_format_options opt, const char* title)
{
    banner(title);
    auto s = format_exception(e, opt);
    std::fputs(s.c_str(), stdout);
}
}  // namespace

int
main()
{
    // ------------------------------------------------------------------
    // PART A - throw-site capture (the good case).
    // ------------------------------------------------------------------
    auto              t   = library_initialize("bad_metric");
    const std::string msg = "Failed to parse PMC metric 'bad_metric': "
                            "value out of range";

    {
        stacktrace::format_options opt;
        render_trace(t, opt, msg, "[A1] throw-site capture - default (auto color)");
    }
    {
        stacktrace::format_options opt;
        opt.color = stacktrace::color_mode::off;
        render_trace(t, opt, msg, "[A2] throw-site capture - color off");
    }
    {
        stacktrace::format_options opt;
        opt.color       = stacktrace::color_mode::on;
        opt.with_offset = true;
        opt.with_module = true;
        render_trace(t, opt, msg,
                     "[A3] throw-site capture - color on + offsets + modules");
    }
    {
        stacktrace::format_options opt;
        opt.color               = stacktrace::color_mode::on;
        opt.with_source_excerpt = true;
        render_trace(t, opt, msg, "[A4] throw-site capture - color on + source excerpt");
    }
    {
        stacktrace::format_options opt;
        opt.color            = stacktrace::color_mode::on;
        opt.max_frames_shown = 2;
        render_trace(t, opt, msg, "[A5] throw-site capture - max_frames_shown=2");
    }

    // ------------------------------------------------------------------
    // PART B - catch-site capture via format_exception.
    // Phase 1's standard path for legacy std::exception that doesn't
    // carry an embedded trace. After throw, the unwind has discarded
    // the frames between throw and catch; we only see the catch site.
    // ------------------------------------------------------------------
    try
    {
        library_throw("bad_metric");
    } catch(const std::exception& e)
    {
        exception_format_options opt;
        opt.trace_options.color = stacktrace::color_mode::on;
        render_caught(e, opt, "[B1] catch-site capture (format_exception) - color on");
    }

    return 0;
}
