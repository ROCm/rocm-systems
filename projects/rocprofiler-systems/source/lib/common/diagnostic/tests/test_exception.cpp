// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/diagnostic/exception.hpp"
#include "common/diagnostic/format_exception.hpp"

#include <exception>
#include <stdexcept>
#include <string>

#include <gtest/gtest.h>

namespace
{
[[gnu::noinline]] void
throw_config_error()
{
    asm volatile("");
    throw rocprofsys::config_error{ "bad config" };
}

[[gnu::noinline]] void
throw_storage_error()
{
    asm volatile("");
    throw rocprofsys::storage_error{ "rocpd write failed" };
}

[[gnu::noinline]] void
throw_gpu_error()
{
    asm volatile("");
    throw rocprofsys::gpu_error{ "kernel dispatch failed" };
}
}  // namespace

TEST(exception, what_is_message_only)
{
    rocprofsys::config_error e{ "hello" };
    EXPECT_STREQ(e.what(), "hello");
    EXPECT_EQ(e.message(), "hello");
}

TEST(exception, trace_captured_at_construction)
{
    rocprofsys::runtime_error e{ "x" };
    EXPECT_FALSE(e.trace().empty());
}

TEST(exception, derived_caught_as_base)
{
    try
    {
        throw_config_error();
    } catch(const rocprofsys::exception& e)
    {
        EXPECT_STREQ(e.what(), "bad config");
        return;
    }
    FAIL() << "expected rocprofsys::exception";
}

TEST(exception, derived_caught_as_std_runtime_error)
{
    try
    {
        throw_storage_error();
    } catch(const std::runtime_error& e)
    {
        EXPECT_STREQ(e.what(), "rocpd write failed");
        return;
    }
    FAIL() << "expected std::runtime_error";
}

TEST(exception, derived_caught_as_std_exception)
{
    try
    {
        throw_gpu_error();
    } catch(const std::exception& e)
    {
        EXPECT_STREQ(e.what(), "kernel dispatch failed");
        return;
    }
    FAIL() << "expected std::exception";
}

TEST(exception, hierarchy_dynamic_cast_chain)
{
    try
    {
        throw_storage_error();
    } catch(const std::exception& e)
    {
        EXPECT_NE(dynamic_cast<const rocprofsys::exception*>(&e), nullptr);
        EXPECT_NE(dynamic_cast<const rocprofsys::serialization_error*>(&e), nullptr);
        EXPECT_NE(dynamic_cast<const rocprofsys::storage_error*>(&e), nullptr);
        EXPECT_EQ(dynamic_cast<const rocprofsys::schema_error*>(&e), nullptr);
        EXPECT_EQ(dynamic_cast<const rocprofsys::gpu_error*>(&e), nullptr);
        return;
    }
    FAIL();
}

TEST(exception, gpu_error_is_device_error)
{
    try
    {
        throw_gpu_error();
    } catch(const rocprofsys::device_error& e)
    {
        EXPECT_STREQ(e.what(), "kernel dispatch failed");
        return;
    }
    FAIL() << "expected device_error";
}

TEST(exception, format_exception_uses_embedded_trace)
{
    // The embedded trace was captured inside throw_config_error; the format
    // call site is here in the test. If the formatter uses the embedded
    // trace, the rendered output must mention the throw helper, not the
    // test body / format_exception itself.
    try
    {
        throw_config_error();
    } catch(const std::exception& e)
    {
        rocprofsys::common::diagnostic::exception_format_options opt;
        opt.with_color                = false;
        opt.trace_options.with_color  = false;
        opt.trace_options.with_module = false;
        const auto rendered = rocprofsys::common::diagnostic::format_exception(e, opt);
        EXPECT_NE(rendered.find("throw_config_error"), std::string::npos) << "rendered:\n"
                                                                          << rendered;
        return;
    }
    FAIL();
}

TEST(exception, format_exception_falls_back_for_std_runtime_error)
{
    // Non-rocprofsys exception still gets a trace, captured at format time.
    try
    {
        throw std::runtime_error{ "plain" };
    } catch(const std::exception& e)
    {
        rocprofsys::common::diagnostic::exception_format_options opt;
        opt.with_color      = false;
        const auto rendered = rocprofsys::common::diagnostic::format_exception(e, opt);
        EXPECT_NE(rendered.find("plain"), std::string::npos);
        return;
    }
    FAIL();
}
