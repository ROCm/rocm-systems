// Copyright (c) 2024 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "details/environment.hpp"

#include <gtest/gtest.h>

#if defined(_WIN32)
#    include "details/platform/windows/encoding.hpp"  // also pulls in rocprofiler_register_windows.h
#endif

#include <cstdlib>
#include <string>

namespace
{
constexpr const char* kEnvName = "ROCPROFILER_REGISTER_TEST_ENV_VAR";

void
unset_env(const char* name)
{
#if defined(_WIN32)
    auto wide_name = rocprofiler_register::platform::encoding::utf8_to_wide(name);
    ::SetEnvironmentVariableW(wide_name.c_str(), nullptr);
#else
    ::unsetenv(name);
#endif
}
}  // namespace

TEST(environment, read_unset_returns_nullopt)
{
    using namespace rocprofiler_register::common;
    unset_env(kEnvName);

    auto value = read_env_string(kEnvName);
    EXPECT_FALSE(value.has_value());
}

TEST(environment, set_then_get_round_trips)
{
    using namespace rocprofiler_register::common;
    unset_env(kEnvName);

    auto rc = write_env_string(kEnvName, "alpha", /*overwrite=*/true);
    EXPECT_EQ(rc, 0);

    auto value = read_env_string(kEnvName);
    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(*value, std::string{ "alpha" });

    unset_env(kEnvName);
}

TEST(environment, overwrite_false_preserves_existing)
{
    using namespace rocprofiler_register::common;
    unset_env(kEnvName);
    write_env_string(kEnvName, "first", /*overwrite=*/true);

    auto rc = write_env_string(kEnvName, "second", /*overwrite=*/false);
    EXPECT_EQ(rc, 0);

    auto value = read_env_string(kEnvName);
    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(*value, std::string{ "first" });

    unset_env(kEnvName);
}

TEST(environment, overwrite_true_replaces_existing)
{
    using namespace rocprofiler_register::common;
    unset_env(kEnvName);
    write_env_string(kEnvName, "first", /*overwrite=*/true);
    write_env_string(kEnvName, "second", /*overwrite=*/true);

    auto value = read_env_string(kEnvName);
    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(*value, std::string{ "second" });

    unset_env(kEnvName);
}

TEST(environment, path_list_separator_is_platform_correct)
{
    using namespace rocprofiler_register::common;
#if defined(_WIN32)
    EXPECT_EQ(path_list_separator(), ';');
#else
    EXPECT_EQ(path_list_separator(), ':');
#endif
}

TEST(environment, get_env_returns_default_when_unset)
{
    using namespace rocprofiler_register::common;
    unset_env(kEnvName);

    auto value = get_env(kEnvName, std::string{ "fallback" });
    EXPECT_EQ(value, std::string{ "fallback" });
}

TEST(environment, get_env_returns_value_when_set)
{
    using namespace rocprofiler_register::common;
    unset_env(kEnvName);
    write_env_string(kEnvName, "from-env", /*overwrite=*/true);

    auto value = get_env(kEnvName, std::string{ "fallback" });
    EXPECT_EQ(value, std::string{ "from-env" });

    unset_env(kEnvName);
}

#if defined(_WIN32)
// WINDOWS-DIVERGENCE: this test verifies that read_env_string sees values set
// via Win32 SetEnvironmentVariableW even if the CRT's getenv() snapshot is
// stale. Linux does not have this CRT/kernel split.
TEST(environment, win32_direct_set_visible_to_read_env_string)
{
    using namespace rocprofiler_register::common;
    unset_env(kEnvName);

    auto wide_name  = std::wstring(L"ROCPROFILER_REGISTER_TEST_ENV_VAR");
    auto wide_value = std::wstring(L"win32-value");
    ASSERT_NE(::SetEnvironmentVariableW(wide_name.c_str(), wide_value.c_str()), 0);

    auto value = read_env_string(kEnvName);
    ASSERT_TRUE(value.has_value());
    EXPECT_EQ(*value, std::string{ "win32-value" });

    ::SetEnvironmentVariableW(wide_name.c_str(), nullptr);
    unset_env(kEnvName);
}
#endif
