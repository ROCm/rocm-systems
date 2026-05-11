// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/diagnostic/format_exception.hpp"
#include "common/diagnostic/color.hpp"

#include <cstdio>
#include <sstream>
#include <string>
#include <string_view>

namespace rocprofsys
{
inline namespace common
{
namespace diagnostic
{
namespace
{
// timemory's exception<T>::what() embeds the backtrace as an extra line.
// Strip everything from the first newline onward so the header stays
// single-line; the trace we render below replaces the embedded one.
std::string_view
trim_what(const char* w)
{
    if(w == nullptr)
    {
        return {};
    }
    std::string_view s{ w };
    if(auto pos = s.find('\n'); pos != std::string_view::npos)
    {
        return s.substr(0, pos);
    }
    return s;
}
}  // namespace

std::string
format_exception(const std::exception& e, exception_format_options opt)
{
    bool color_on = false;
    switch(opt.trace_options.color)
    {
        case stacktrace::color_mode::off: color_on = false; break;
        case stacktrace::color_mode::on: color_on = true; break;
        case stacktrace::color_mode::auto_detect:
            color_on = color::color_supported_for(2);
            break;
    }

    auto col = [color_on](const char* code) -> const char* {
        return color_on ? code : "";
    };

    std::ostringstream out;
    out << col(color::error_kw) << "error:" << col(color::reset) << ' '
        << trim_what(e.what()) << '\n'
        << '\n';

    if(opt.capture_trace_at_call_site)
    {
        // skip 1 to drop format_exception's own frame.
        auto trace = stacktrace::capture(/*skip_frames=*/1);
        out << trace.to_string(opt.trace_options);
    }

    return out.str();
}

void
print_exception(const std::exception& e)
{
    exception_format_options opt;
    opt.trace_options.color = stacktrace::color_mode::auto_detect;
    auto s                  = format_exception(e, opt);
    std::fputs(s.c_str(), stderr);
    std::fputc('\n', stderr);
    std::fflush(stderr);
}
}  // namespace diagnostic
}  // namespace common
}  // namespace rocprofsys
