// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Demo program for the rocprofsys diagnostic format.
//
// Throws an exception from a deep call chain and renders the side-bar
// wrapped gdb-style output in the two supported modes:
//
//   1. Default (with color, UTF-8 side-bar)
//   2. No color (ASCII side-bar)

#include "common/diagnostic/format_exception.hpp"
#include "common/diagnostic/stacktrace.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>

using rocprofsys::common::diagnostic::exception_format_options;
using rocprofsys::common::diagnostic::format_exception;

namespace
{
[[gnu::noinline]] void
parse_metric_value(const std::string& s)
{
    asm volatile("");
    throw std::runtime_error{ "Failed to parse PMC metric '" + s +
                              "': value out of range" };
}

[[gnu::noinline]] void
collector_setup(const std::string& s)
{
    asm volatile("");
    parse_metric_value(s);
}

[[gnu::noinline]] void
sampler_setup(const std::string& s)
{
    asm volatile("");
    collector_setup(s);
}

[[gnu::noinline]] void
library_initialize(const std::string& s)
{
    asm volatile("");
    sampler_setup(s);
}

void
section(const std::exception& e, bool color_on, const char* title)
{
    std::printf("=== %s ===\n", title);
    exception_format_options opt;
    opt.with_color = color_on;
    auto s         = format_exception(e, opt);
    std::fputs(s.c_str(), stdout);
    std::fputc('\n', stdout);
}
}  // namespace

int
main()
{
    try
    {
        library_initialize("bad_metric");
    } catch(const std::exception& e)
    {
        section(e, /*color_on=*/true, "side-bar wrap (default - with color)");
        section(e, /*color_on=*/false, "side-bar wrap (no color)");
    }
    return 0;
}
