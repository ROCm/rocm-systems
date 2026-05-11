// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/diagnostic/color.hpp"

#include <cstdlib>

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

TEST(diagnostic_color_test, NoColorEnvDisablesEvenForcedTty)
{
    color_env_guard nc{ "NO_COLOR" };
    color_env_guard cf{ "CLICOLOR_FORCE" };
    ::setenv("NO_COLOR", "1", 1);
    ::setenv("CLICOLOR_FORCE", "1", 1);
    // Use fd 1; whether it's a TTY in CI doesn't matter - NO_COLOR wins.
    EXPECT_FALSE(color::color_supported_for(1));
}

TEST(diagnostic_color_test, ClicolorForceEnablesOffTty)
{
    color_env_guard nc{ "NO_COLOR" };
    color_env_guard cf{ "CLICOLOR_FORCE" };
    ::unsetenv("NO_COLOR");
    ::setenv("CLICOLOR_FORCE", "1", 1);
    // /dev/null fd if needed - here we use the closed fd 99 which is not
    // a tty; CLICOLOR_FORCE should still enable.
    EXPECT_TRUE(color::color_supported_for(99));
}

TEST(diagnostic_color_test, NoTtyNoForceDisables)
{
    color_env_guard nc{ "NO_COLOR" };
    color_env_guard cf{ "CLICOLOR_FORCE" };
    ::unsetenv("NO_COLOR");
    ::unsetenv("CLICOLOR_FORCE");
    EXPECT_FALSE(color::color_supported_for(99));
}
