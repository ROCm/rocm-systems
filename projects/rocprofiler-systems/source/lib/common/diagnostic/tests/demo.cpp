// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Demo program for the rocprofsys diagnostic format.
//
// Captures one trace from a deep call chain and renders it in three modes
// to exercise the gdb-style compact format:
//
//   1. Default        - color decided by auto_detect (TTY + NO_COLOR rules)
//   2. Color OFF      - explicit color_mode::off
//   3. Color FORCED   - explicit color_mode::on (bypasses TTY check)

#include "common/diagnostic/format_exception.hpp"
#include "common/diagnostic/stacktrace.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>

using rocprofsys::common::diagnostic::stacktrace;

namespace
{
[[gnu::noinline]] stacktrace
parse_metric_value(const std::string& s)
{
    asm volatile("");
    if(s.empty() || s[0] == 'b')
    {
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

void
section(const stacktrace& trace, stacktrace::format_options opt, const char* title,
        const std::string& msg)
{
    std::printf("=== %s ===\n", title);

    const bool color_on = opt.color == stacktrace::color_mode::on;
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
    std::fputc('\n', stdout);
}
}  // namespace

int
main()
{
    auto              trace = library_initialize("bad_metric");
    const std::string msg   = "Failed to parse PMC metric 'bad_metric': "
                              "value out of range";

    {
        stacktrace::format_options opt;
        opt.color = stacktrace::color_mode::auto_detect;
        section(trace, opt, "gdb-style (default)", msg);
    }

    {
        stacktrace::format_options opt;
        opt.color = stacktrace::color_mode::off;
        section(trace, opt, "gdb-style (NO color)", msg);
    }

    {
        stacktrace::format_options opt;
        opt.color = stacktrace::color_mode::on;
        section(trace, opt, "gdb-style (FORCED color)", msg);
    }

    return 0;
}
