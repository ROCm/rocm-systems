// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/diagnostic/format_exception.hpp"
#include "common/diagnostic/stacktrace.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>

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
        EXPECT_NE(s.find("error:"), std::string::npos);
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
        EXPECT_NE(s.find("error:"), std::string::npos);
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
        // Header (the box) must end before the first blank line.
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

TEST(diagnostic_format_exception_test, BoxPresentDefault)
{
    try
    {
        exception_test_throw_runtime();
    } catch(const std::exception& e)
    {
        // Default (with color) -> UTF-8 box-drawing top-left corner.
        auto s = format_exception(e);
        // U+250C top-left corner.
        EXPECT_NE(s.find("\xe2\x94\x8c"), std::string::npos)
            << "expected UTF-8 box-drawing corner; output was:\n"
            << s;
    }
}

TEST(diagnostic_format_exception_test, BoxPresentAsciiWhenNoColor)
{
    try
    {
        exception_test_throw_runtime();
    } catch(const std::exception& e)
    {
        auto s = format_exception(e, plain_opts());
        // First character of the framed block is `+`.
        ASSERT_FALSE(s.empty());
        EXPECT_EQ(s[0], '+') << "expected ASCII box corner; output was:\n" << s;
        EXPECT_NE(s.find("|"), std::string::npos);
        EXPECT_NE(s.find("+--"), std::string::npos);
    }
}

TEST(diagnostic_format_exception_test, ErrorInsideBoxWithRedWhenColored)
{
    try
    {
        exception_test_throw_runtime();
    } catch(const std::exception& e)
    {
        auto s = format_exception(e);
        // The `error:` keyword must be styled bright-red bold.
        // Composite escape used by color::error_kw is `\033[1;91m`.
        auto err_pos = s.find("error:");
        ASSERT_NE(err_pos, std::string::npos);
        // The bright-red bold escape must precede the keyword.
        const std::string red_bold = "\033[1;91m";
        auto              red_pos  = s.rfind(red_bold, err_pos);
        EXPECT_NE(red_pos, std::string::npos)
            << "error: keyword must be bright-red bold; output was:\n"
            << s;
        // The line carrying `error:` must be wrapped in vertical box borders
        // (UTF-8 U+2502).
        const std::string line_start = "\xe2\x94\x82";  // |
        auto              before     = s.rfind(line_start, err_pos);
        EXPECT_NE(before, std::string::npos)
            << "error: keyword must sit inside a vertical box border";
    }
}
