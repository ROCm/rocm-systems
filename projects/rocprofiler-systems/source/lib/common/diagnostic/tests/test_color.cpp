// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/diagnostic/color.hpp"

#include <cstdlib>
#include <string>

#include <gtest/gtest.h>

namespace
{
class color_env_guard
{
public:
    explicit color_env_guard(const char* name)
    : m_name{ name }
    {
        if(const char* prev = std::getenv(name); prev != nullptr)
        {
            m_had   = true;
            m_saved = prev;
        }
    }

    ~color_env_guard()
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

    color_env_guard(const color_env_guard&)            = delete;
    color_env_guard& operator=(const color_env_guard&) = delete;
    color_env_guard(color_env_guard&&)                 = delete;
    color_env_guard& operator=(color_env_guard&&)      = delete;

private:
    const char* m_name = nullptr;
    bool        m_had  = false;
    std::string m_saved;
};
}  // namespace

namespace color = rocprofsys::common::diagnostic::color;

TEST(diagnostic_color_test, NoColorEnvDetected)
{
    color_env_guard nc{ "NO_COLOR" };
    ::setenv("NO_COLOR", "1", 1);
    EXPECT_TRUE(color::no_color_env());
}

TEST(diagnostic_color_test, NoColorEnvUnsetReportsFalse)
{
    color_env_guard nc{ "NO_COLOR" };
    ::unsetenv("NO_COLOR");
    EXPECT_FALSE(color::no_color_env());
}

TEST(diagnostic_color_test, NoColorEnvAnyValueDetected)
{
    color_env_guard nc{ "NO_COLOR" };
    ::setenv("NO_COLOR", "", 1);  // even empty string counts per no-color.org
    EXPECT_TRUE(color::no_color_env());
}
