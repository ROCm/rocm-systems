// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/diagnostic/format_exception.hpp"
#include "common/diagnostic/stacktrace.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

using rocprofsys::common::diagnostic::exception_format_options;
using rocprofsys::common::diagnostic::format_exception;
using rocprofsys::common::diagnostic::stacktrace;

namespace
{
[[gnu::noinline]] void
exception_test_throw_runtime()
{
    asm volatile("");
    throw std::runtime_error{ "value out of range" };
}

[[gnu::noinline]] void
exception_test_throw_range()
{
    asm volatile("");
    throw std::out_of_range{ "index 9 of 5" };
}

class env_guard
{
public:
    env_guard(const char* name, const char* value)
    : m_name{ name }
    {
        if(const char* prev = std::getenv(name); prev != nullptr)
        {
            m_had   = true;
            m_saved = prev;
        }
        if(value != nullptr)
        {
            ::setenv(name, value, 1);
        }
        else
        {
            ::unsetenv(name);
        }
    }

    ~env_guard()
    {
        if(m_had)
        {
            ::setenv(m_name, m_saved.c_str(), 1);
        }
        else
        {
            ::unsetenv(m_name);
        }
    }

    env_guard(const env_guard&)            = delete;
    env_guard& operator=(const env_guard&) = delete;
    env_guard(env_guard&&)                 = delete;
    env_guard& operator=(env_guard&&)      = delete;

private:
    const char* m_name = nullptr;
    bool        m_had  = false;
    std::string m_saved;
};

exception_format_options
plain_opts()
{
    exception_format_options o;
    o.with_color = false;
    return o;
}
}  // namespace

TEST(diagnostic_format_exception_test, FormatsRuntimeErrorWithMessage)
{
    try
    {
        exception_test_throw_runtime();
        FAIL() << "expected std::runtime_error";
    } catch(const std::exception& e)
    {
        auto s = format_exception(e, plain_opts());
        EXPECT_NE(s.find("error"), std::string::npos);
        EXPECT_NE(s.find("value out of range"), std::string::npos);
        EXPECT_NE(s.find("#0 "), std::string::npos);
    }
}

TEST(diagnostic_format_exception_test, FormatsOutOfRangeWithMessage)
{
    try
    {
        exception_test_throw_range();
        FAIL() << "expected std::out_of_range";
    } catch(const std::exception& e)
    {
        auto s = format_exception(e, plain_opts());
        EXPECT_NE(s.find("error"), std::string::npos);
        EXPECT_NE(s.find("index 9 of 5"), std::string::npos);
    }
}

TEST(diagnostic_format_exception_test, ColorOffProducesNoEscape)
{
    try
    {
        exception_test_throw_runtime();
    } catch(const std::exception& e)
    {
        auto s = format_exception(e, plain_opts());
        EXPECT_EQ(s.find("\033["), std::string::npos);
    }
}

TEST(diagnostic_format_exception_test, ColorOnProducesEscape)
{
    try
    {
        exception_test_throw_runtime();
    } catch(const std::exception& e)
    {
        exception_format_options opt;
        opt.with_color = true;
        auto s         = format_exception(e, opt);
        EXPECT_NE(s.find("\033["), std::string::npos);
    }
}

TEST(diagnostic_format_exception_test, HeaderHasSingleLineWhat)
{
    try
    {
        throw std::runtime_error{ "first line\nsecond line should be trimmed" };
    } catch(const std::exception& e)
    {
        auto s = format_exception(e, plain_opts());
        // The trimmed second-line text must not appear anywhere in the
        // formatted block since `what()` is single-line-trimmed.
        EXPECT_EQ(s.find("second line"), std::string::npos)
            << "second line not trimmed; output was:\n"
            << s;
    }
}

TEST(diagnostic_format_exception_test, NoColorEnvDisablesDefault)
{
    env_guard nc{ "NO_COLOR", "1" };
    try
    {
        exception_test_throw_runtime();
    } catch(const std::exception& e)
    {
        // No explicit `with_color` -> NO_COLOR env should disable color.
        auto s = format_exception(e);
        EXPECT_EQ(s.find("\033["), std::string::npos)
            << "NO_COLOR must suppress all ANSI escapes; output was:\n"
            << s;
    }
}

TEST(diagnostic_format_exception_test, ExplicitColorOverridesNoColorEnv)
{
    env_guard nc{ "NO_COLOR", "1" };
    try
    {
        exception_test_throw_runtime();
    } catch(const std::exception& e)
    {
        exception_format_options opt;
        opt.with_color = true;  // explicit override wins over NO_COLOR
        auto s         = format_exception(e, opt);
        EXPECT_NE(s.find("\033["), std::string::npos);
    }
}

// Collect lines from a rendered block; trailing empty line (after final '\n')
// is dropped so callers can iterate "real" lines only.
std::vector<std::string>
split_lines(const std::string& s)
{
    std::vector<std::string> out;
    std::string              cur;
    for(char c : s)
    {
        if(c == '\n')
        {
            out.push_back(std::move(cur));
            cur.clear();
        }
        else
        {
            cur.push_back(c);
        }
    }
    if(!cur.empty())
    {
        out.push_back(std::move(cur));
    }
    return out;
}

TEST(diagnostic_format_exception_test, header_line_present_default)
{
    try
    {
        exception_test_throw_runtime();
    } catch(const std::exception& e)
    {
        auto s     = format_exception(e);
        auto lines = split_lines(s);
        ASSERT_FALSE(lines.empty());
        // First line opens the side-bar: `┌─` (U+250C U+2500) + ` error`.
        const std::string utf8_lead = "\xe2\x94\x8c\xe2\x94\x80";
        EXPECT_NE(lines.front().find(utf8_lead), std::string::npos)
            << "expected UTF-8 side-bar lead-in; first line was:\n"
            << lines.front();
        EXPECT_NE(lines.front().find("error"), std::string::npos);
    }
}

TEST(diagnostic_format_exception_test, header_line_ascii_when_no_color)
{
    try
    {
        exception_test_throw_runtime();
    } catch(const std::exception& e)
    {
        auto s     = format_exception(e, plain_opts());
        auto lines = split_lines(s);
        ASSERT_FALSE(lines.empty());
        EXPECT_EQ(lines.front(), std::string{ "+- error" })
            << "ASCII header must be `+- error`; first line was:\n"
            << lines.front();
    }
}

TEST(diagnostic_format_exception_test, error_label_red_when_colored)
{
    try
    {
        exception_test_throw_runtime();
    } catch(const std::exception& e)
    {
        auto s       = format_exception(e);
        auto err_pos = s.find("error");
        ASSERT_NE(err_pos, std::string::npos);
        // Bright-red bold escape must appear immediately before the label.
        const std::string red_bold = "\033[1;91m";
        auto              red_pos  = s.rfind(red_bold, err_pos);
        ASSERT_NE(red_pos, std::string::npos)
            << "expected bright-red-bold prefix; output was:\n"
            << s;
        // No content between the escape and the label other than the reset
        // following the bar glyph - the `error` token sits directly after the
        // composite escape.
        EXPECT_EQ(s.compare(red_pos + red_bold.size(), 5, "error"), 0)
            << "expected `error` immediately after red-bold escape";
    }
}

TEST(diagnostic_format_exception_test, closing_line_present)
{
    try
    {
        exception_test_throw_runtime();
    } catch(const std::exception& e)
    {
        // Colored output: last line (after splitting on '\n') must contain
        // the UTF-8 bottom-left + horizontal lead-in. The line itself is
        // wrapped in border-color + reset escapes when colored.
        auto s_color = format_exception(e);
        auto lines_c = split_lines(s_color);
        ASSERT_FALSE(lines_c.empty());
        const std::string utf8_close = "\xe2\x94\x94\xe2\x94\x80";
        EXPECT_NE(lines_c.back().find(utf8_close), std::string::npos)
            << "expected UTF-8 closer on last line; was:\n"
            << lines_c.back();

        // Plain output: last line is exactly `+-`.
        auto s_plain = format_exception(e, plain_opts());
        auto lines_p = split_lines(s_plain);
        ASSERT_FALSE(lines_p.empty());
        EXPECT_EQ(lines_p.back(), std::string{ "+-" })
            << "expected `+-` on last line; was:\n"
            << lines_p.back();
    }
}

TEST(diagnostic_format_exception_test, every_content_line_has_left_bar)
{
    try
    {
        exception_test_throw_runtime();
    } catch(const std::exception& e)
    {
        auto s     = format_exception(e, plain_opts());
        auto lines = split_lines(s);
        ASSERT_GE(lines.size(), 3u);  // header + at least one body + closer
        // Lines between the first (header) and last (closer) must start with
        // `|` (a `| ` for content lines, bare `|` for the blank separator).
        for(std::size_t i = 1; i + 1 < lines.size(); ++i)
        {
            const auto& line = lines[i];
            ASSERT_FALSE(line.empty()) << "line " << i << " was empty";
            EXPECT_EQ(line[0], '|') << "line " << i << " missing left-bar; was:\n"
                                    << line;
        }
    }
}

TEST(diagnostic_format_exception_test, frames_inside_bar)
{
    try
    {
        exception_test_throw_runtime();
    } catch(const std::exception& e)
    {
        auto s           = format_exception(e, plain_opts());
        auto lines       = split_lines(s);
        bool found_frame = false;
        for(std::size_t i = 1; i + 1 < lines.size(); ++i)
        {
            const auto& line = lines[i];
            if(line.find("#0") != std::string::npos)
            {
                ASSERT_FALSE(line.empty());
                EXPECT_EQ(line[0], '|') << "frame `#0` line missing left-bar";
                found_frame = true;
                break;
            }
        }
        EXPECT_TRUE(found_frame) << "expected a `#0` frame line in:\n" << s;
    }
}
