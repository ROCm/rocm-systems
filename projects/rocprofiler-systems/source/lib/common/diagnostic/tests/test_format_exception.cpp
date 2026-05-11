// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/diagnostic/format_exception.hpp"
#include "common/diagnostic/stacktrace.hpp"

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
}  // namespace

TEST(diagnostic_format_exception_test, FormatsRuntimeErrorWithMessage)
{
    try
    {
        exception_test_throw_runtime();
        FAIL() << "expected std::runtime_error";
    } catch(const std::exception& e)
    {
        exception_format_options opt;
        opt.trace_options.color = stacktrace::color_mode::off;
        auto s                  = format_exception(e, opt);
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
        exception_format_options opt;
        opt.trace_options.color = stacktrace::color_mode::off;
        auto s                  = format_exception(e, opt);
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
        exception_format_options opt;
        opt.trace_options.color = stacktrace::color_mode::off;
        auto s                  = format_exception(e, opt);
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
        opt.trace_options.color = stacktrace::color_mode::on;
        auto s                  = format_exception(e, opt);
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
        exception_format_options opt;
        opt.trace_options.color = stacktrace::color_mode::off;
        auto s                  = format_exception(e, opt);
        // Header must end at first newline; second line text must not be
        // present in the header itself. We check by ensuring the literal
        // "second line" text does not appear before the first blank line.
        auto blank = s.find("\n\n");
        ASSERT_NE(blank, std::string::npos);
        auto hdr = s.substr(0, blank);
        EXPECT_EQ(hdr.find("second line"), std::string::npos) << "header was:\n" << hdr;
    }
}
