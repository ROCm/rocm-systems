// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/diagnostic/format_exception.hpp"
#include "common/diagnostic/color.hpp"
#include "common/diagnostic/exception.hpp"

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

// Side-bar glyphs. UTF-8 box-drawing when colored, ASCII when not.
struct bar_chars
{
    const char* tl;  // header lead-in (┌─ / +-)
    const char* bl;  // closer        (└─ / +-)
    const char* v;   // left margin   (│  / |)
};

constexpr bar_chars
utf8_chars()
{
    return { "\xe2\x94\x8c\xe2\x94\x80",  // U+250C U+2500
             "\xe2\x94\x94\xe2\x94\x80",  // U+2514 U+2500
             "\xe2\x94\x82" };            // U+2502
}

constexpr bar_chars
ascii_chars()
{
    return { "+-", "+-", "|" };
}

// Wrap `body` (already-rendered, may be multi-line) in a side-bar block.
// Every line gets the `<bar> ` prefix; blank lines get a bare `<bar>` (no
// trailing space) so they stay clean. The header line opens the block and
// labels it `error`; the closer line ends with the bar lead-in glyph.
std::string
wrap_in_sidebar(std::string_view body, bool color_on)
{
    const bar_chars chars  = color_on ? utf8_chars() : ascii_chars();
    const char*     border = color_on ? color::box_border : "";
    const char*     err_kw = color_on ? color::error_kw : "";
    const char*     reset  = color_on ? color::reset : "";

    std::ostringstream out;

    // Header: `┌─ error` (border + glyph dim, `error` bold bright-red).
    out << border << chars.tl << reset << ' ' << err_kw << "error" << reset << '\n';

    // Body: split on '\n' and prefix every line with the left bar.
    std::size_t i = 0;
    const auto  n = body.size();
    while(i <= n)
    {
        const std::size_t j    = body.find('\n', i);
        const std::size_t end  = (j == std::string_view::npos) ? n : j;
        const auto        line = body.substr(i, end - i);

        if(line.empty())
        {
            out << border << chars.v << reset << '\n';
        }
        else
        {
            out << border << chars.v << reset << ' ' << line << '\n';
        }

        if(j == std::string_view::npos)
        {
            break;
        }
        i = j + 1;
    }

    // Closer.
    out << border << chars.bl << reset << '\n';

    return out.str();
}
}  // namespace

std::string
format_exception(const std::exception& e, exception_format_options opt)
{
    // Resolve effective color:
    //  - explicit `opt.with_color` override always wins;
    //  - otherwise default to true, flipped to false when `NO_COLOR` is set.
    const bool color_on =
        opt.with_color.has_value() ? *opt.with_color : !color::no_color_env();
    opt.trace_options.with_color = color_on;

    // Assemble the inner block (message + blank line + frames) before
    // wrapping. This lets the side-bar prefixer treat the whole exception
    // uniformly without leaking into the trace renderer.
    std::ostringstream inner;
    inner << trim_what(e.what());

    // Prefer an embedded trace when the exception is a rocprofsys::exception:
    // it was captured at the throw site, which is what the user wants to see.
    // Otherwise capture at format-time (still useful, but points at the
    // catch-formatting site, not the throw).
    std::string rendered;
    if(const auto* re = dynamic_cast<const ::rocprofsys::exception*>(&e); re != nullptr)
    {
        rendered = re->trace().to_string(opt.trace_options);
    }
    else if(opt.capture_trace_at_call_site)
    {
        // skip 1 to drop format_exception's own frame.
        auto trace = stacktrace::capture(/*skip_frames=*/1);
        rendered   = trace.to_string(opt.trace_options);
    }

    // Drop a single trailing newline from the trace render so the side-bar
    // prefixer doesn't append a phantom bare-bar line under the last frame.
    if(!rendered.empty() && rendered.back() == '\n')
    {
        rendered.pop_back();
    }

    if(!rendered.empty())
    {
        inner << "\n\n" << rendered;
    }

    return wrap_in_sidebar(inner.str(), color_on);
}

void
print_exception(const std::exception& e)
{
    auto s = format_exception(e);
    std::fputs(s.c_str(), stderr);
    std::fputc('\n', stderr);
    std::fflush(stderr);
}
}  // namespace diagnostic
}  // namespace common
}  // namespace rocprofsys
