// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/diagnostic/format_exception.hpp"
#include "common/diagnostic/color.hpp"

#include <algorithm>
#include <cstdio>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

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

// Box geometry: minimum interior width keeps the frame from collapsing
// around very short messages.
constexpr std::size_t k_box_min_inner = 56;
constexpr std::size_t k_box_max_inner = 116;
constexpr std::size_t k_box_pad       = 1;  // single space inside each border

// Wrap a single logical message line to fit the chosen interior width.
// Greedy word-wrap; breaks mid-word only when a token exceeds inner_width.
std::vector<std::string>
wrap_line(const std::string& line, std::size_t inner_width)
{
    std::vector<std::string> out;
    if(inner_width == 0)
    {
        out.push_back(line);
        return out;
    }
    std::string cur;
    cur.reserve(inner_width);

    std::size_t i = 0;
    const auto  n = line.size();
    while(i < n)
    {
        std::size_t j = line.find(' ', i);
        if(j == std::string::npos) j = n;
        const std::string word = line.substr(i, j - i);
        i                      = (j < n) ? j + 1 : j;

        if(word.size() > inner_width)
        {
            // Token longer than the box interior: hard-break.
            if(!cur.empty())
            {
                out.push_back(cur);
                cur.clear();
            }
            std::size_t k = 0;
            while(k < word.size())
            {
                const std::size_t take = std::min(inner_width, word.size() - k);
                out.emplace_back(word.substr(k, take));
                k += take;
            }
            continue;
        }

        const std::size_t need = cur.empty() ? word.size() : cur.size() + 1 + word.size();
        if(need > inner_width)
        {
            out.push_back(cur);
            cur = word;
        }
        else
        {
            if(!cur.empty()) cur.push_back(' ');
            cur.append(word);
        }
    }
    if(!cur.empty() || out.empty())
    {
        out.push_back(cur);
    }
    return out;
}

struct box_chars
{
    const char* tl;  // top-left corner
    const char* tr;  // top-right corner
    const char* bl;  // bottom-left corner
    const char* br;  // bottom-right corner
    const char* h;   // horizontal segment
    const char* v;   // vertical segment
};

constexpr box_chars
utf8_chars()
{
    return { "\xe2\x94\x8c",    // U+250C
             "\xe2\x94\x90",    // U+2510
             "\xe2\x94\x94",    // U+2514
             "\xe2\x94\x98",    // U+2518
             "\xe2\x94\x80",    // U+2500
             "\xe2\x94\x82" };  // U+2502
}

constexpr box_chars
ascii_chars()
{
    return { "+", "+", "+", "+", "-", "|" };
}

// Repeat `s` `n` times.
std::string
repeat(const char* s, std::size_t n)
{
    std::string out;
    out.reserve(n * 4);
    for(std::size_t i = 0; i < n; ++i)
        out.append(s);
    return out;
}

// Render the framed `error: <message>` block. The error keyword inside the
// box is painted bright-red bold when colored.
std::string
render_error_box(std::string_view message, bool color_on)
{
    const box_chars chars  = color_on ? utf8_chars() : ascii_chars();
    const char*     border = color_on ? color::box_border : "";
    const char*     err_kw = color_on ? color::error_kw : "";
    const char*     reset  = color_on ? color::reset : "";

    // The first wrapped line carries the `error: ` prefix. We treat that
    // literal text as part of the visible width when sizing the box.
    const std::string error_prefix       = "error: ";
    const std::string first_logical_line = error_prefix + std::string{ message };

    // Pick interior width: longest natural line up to the cap, never below
    // the floor.
    std::size_t longest = first_logical_line.size();
    if(longest < k_box_min_inner)
    {
        longest = k_box_min_inner;
    }
    if(longest > k_box_max_inner)
    {
        longest = k_box_max_inner;
    }
    const std::size_t inner_width = longest + 2 * k_box_pad;
    const std::size_t wrap_width  = longest;

    auto wrapped = wrap_line(first_logical_line, wrap_width);

    std::ostringstream out;

    // Top border.
    out << border << chars.tl << repeat(chars.h, inner_width) << chars.tr << reset
        << '\n';

    // Body.
    for(std::size_t li = 0; li < wrapped.size(); ++li)
    {
        const std::string& body = wrapped[li];
        out << border << chars.v << reset << ' ';

        if(li == 0)
        {
            // Highlight the `error:` keyword on the first line only.
            out << err_kw << "error:" << reset
                << body.substr(error_prefix.size() - 1);  // includes leading space
        }
        else
        {
            out << body;
        }

        const std::size_t visible = body.size();
        if(visible < wrap_width)
        {
            out << std::string(wrap_width - visible, ' ');
        }
        out << ' ' << border << chars.v << reset << '\n';
    }

    // Bottom border.
    out << border << chars.bl << repeat(chars.h, inner_width) << chars.br << reset
        << '\n';

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

    std::ostringstream out;
    out << render_error_box(trim_what(e.what()), color_on) << '\n';

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
    auto s = format_exception(e);
    std::fputs(s.c_str(), stderr);
    std::fputc('\n', stderr);
    std::fflush(stderr);
}
}  // namespace diagnostic
}  // namespace common
}  // namespace rocprofsys
