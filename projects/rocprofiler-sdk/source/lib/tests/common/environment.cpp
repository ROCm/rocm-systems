// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "lib/common/environment.hpp"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <string>

namespace
{
bool _common_environment_test_init_logging = (rocprofiler::common::init_logging("TEST"), true);
}

TEST(common, environment)
{
    using rocprofiler::common::env_config;
    using rocprofiler::common::get_env;

    enum TestBareEnum : unsigned short  // NOLINT(performance-enum-size)
    {
        BZero = 0,
        BOne  = 1,
    };

    enum class TestClassEnum : unsigned short  // NOLINT(performance-enum-size)
    {
        CZero = 0,
        COne  = 1,
    };

    //
    //  int testing section
    //
    EXPECT_EQ(get_env("ROCPROFILER_ENV_TEST_INT", 0), 0);

    setenv("ROCPROFILER_ENV_TEST_INT", "1", 1);
    EXPECT_EQ(get_env("ROCPROFILER_ENV_TEST_INT", 0), 1);

    env_config{"ROCPROFILER_ENV_TEST_INT", "2"}();
    EXPECT_EQ(get_env("ROCPROFILER_ENV_TEST_INT", 0), 1);

    env_config{"ROCPROFILER_ENV_TEST_INT", "2", 1}();
    EXPECT_EQ(get_env("ROCPROFILER_ENV_TEST_INT", 0), 2);

    //
    //  enum testing section
    //
    EXPECT_EQ(get_env("ROCPROFILER_ENV_TEST_BARE_ENUM", BZero), BZero);

    env_config{"ROCPROFILER_ENV_TEST_BARE_ENUM", "1", 1}();
    EXPECT_EQ(get_env("ROCPROFILER_ENV_TEST_BARE_ENUM", BZero), BOne);

    EXPECT_EQ(get_env("ROCPROFILER_ENV_TEST_CLASS_ENUM", TestClassEnum::CZero),
              TestClassEnum::CZero);

    env_config{"ROCPROFILER_ENV_TEST_CLASS_ENUM", "1", 1}();
    EXPECT_EQ(get_env("ROCPROFILER_ENV_TEST_CLASS_ENUM", TestClassEnum::CZero),
              TestClassEnum::COne);

    //
    //  string testing section
    //
    EXPECT_EQ(get_env("ROCPROFILER_ENV_TEST_STR", "nostr"), std::string_view{"nostr"});

    env_config{"ROCPROFILER_ENV_TEST_STR", "hasstr", 0}();
    EXPECT_EQ(get_env("ROCPROFILER_ENV_TEST_STR", "nostr"), std::string_view{"hasstr"});

    //
    //  bool testing section
    //
    EXPECT_FALSE(get_env("ROCPROFILER_ENV_TEST_BOOL", false));

    env_config{"ROCPROFILER_ENV_TEST_BOOL", "YES", 1}();
    EXPECT_TRUE(get_env("ROCPROFILER_ENV_TEST_BOOL", false));

    env_config{"ROCPROFILER_ENV_TEST_BOOL", "yes", 1}();
    EXPECT_TRUE(get_env("ROCPROFILER_ENV_TEST_BOOL", false));

    env_config{"ROCPROFILER_ENV_TEST_BOOL", "y", 1}();
    EXPECT_TRUE(get_env("ROCPROFILER_ENV_TEST_BOOL", false));

    env_config{"ROCPROFILER_ENV_TEST_BOOL", "true", 1}();
    EXPECT_TRUE(get_env("ROCPROFILER_ENV_TEST_BOOL", false));

    env_config{"ROCPROFILER_ENV_TEST_BOOL", "on", 1}();
    EXPECT_TRUE(get_env("ROCPROFILER_ENV_TEST_BOOL", false));

    env_config{"ROCPROFILER_ENV_TEST_BOOL", "no", 1}();
    EXPECT_FALSE(get_env("ROCPROFILER_ENV_TEST_BOOL", true));

    env_config{"ROCPROFILER_ENV_TEST_BOOL", "false", 1}();
    EXPECT_FALSE(get_env("ROCPROFILER_ENV_TEST_BOOL", true));

    env_config{"ROCPROFILER_ENV_TEST_BOOL", "0", 1}();
    EXPECT_FALSE(get_env("ROCPROFILER_ENV_TEST_BOOL", true));

    for(auto n : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10})
    {
        env_config{"ROCPROFILER_ENV_TEST_BOOL", std::to_string(n), 1}();
        EXPECT_TRUE(get_env("ROCPROFILER_ENV_TEST_BOOL", false));
    }
}

TEST(common, environment_push_pop)
{
    using rocprofiler::common::env_config;
    using rocprofiler::common::env_store;
    using rocprofiler::common::get_env;

    namespace common = ::rocprofiler::common;

    common::set_env("ROCPROFILER_ENV_TEST_A", "0", 1);
    common::set_env("ROCPROFILER_ENV_TEST_B", "2", 1);

    auto _store = env_store{{env_config{"ROCPROFILER_ENV_TEST_A", "1", 1},
                             env_config{"ROCPROFILER_ENV_TEST_B", "2", 1},
                             env_config{"ROCPROFILER_ENV_TEST_C", "3", 0}}};

    for(size_t i = 0; i < 5; ++i)
    {
        EXPECT_EQ(get_env("ROCPROFILER_ENV_TEST_A", 3), 0) << fmt::format("iteration={}", i);
        EXPECT_EQ(get_env("ROCPROFILER_ENV_TEST_B", 0), 2) << fmt::format("iteration={}", i);
        EXPECT_EQ(get_env("ROCPROFILER_ENV_TEST_C", 1), 1) << fmt::format("iteration={}", i);

        EXPECT_FALSE(_store.is_pushed()) << fmt::format("iteration={}", i);
        EXPECT_TRUE(_store.push()) << fmt::format("iteration={}", i);
        EXPECT_FALSE(_store.push()) << fmt::format("iteration={}", i);

        EXPECT_EQ(get_env("ROCPROFILER_ENV_TEST_A", 3), 1) << fmt::format("iteration={}", i);
        EXPECT_EQ(get_env("ROCPROFILER_ENV_TEST_B", 0), 2) << fmt::format("iteration={}", i);
        EXPECT_EQ(get_env("ROCPROFILER_ENV_TEST_C", 1), 3) << fmt::format("iteration={}", i);

        EXPECT_TRUE(_store.is_pushed()) << fmt::format("iteration={}", i);
        EXPECT_TRUE(_store.pop()) << fmt::format("iteration={}", i);
        EXPECT_FALSE(_store.pop()) << fmt::format("iteration={}", i);

        EXPECT_EQ(get_env("ROCPROFILER_ENV_TEST_A", 3), 0) << fmt::format("iteration={}", i);
        EXPECT_EQ(get_env("ROCPROFILER_ENV_TEST_B", 0), 2) << fmt::format("iteration={}", i);
        EXPECT_EQ(get_env("ROCPROFILER_ENV_TEST_C", 0), 0) << fmt::format("iteration={}", i);
    }
}

// Unit test verifying that get_env() correctly reads environment variables.
//
// PURPOSE:
// Validates that get_env() returns correct values for environment variables,
// regardless of std::getenv() behavior or target application.
//
// CONTEXT:
// Implementation reads from POSIX environ array directly (not std::getenv()) to avoid
// dependency on application-specific getenv() implementations. Applications like bash
// export custom getenv() that can fail when called before initialization.
//
// VALIDATES:
// 1. get_env() returns correct values for variables that exist in environ
// 2. type overloads (string, int, bool) return correct values
TEST(common, environment_reads_directly_from_environ)
{
    using rocprofiler::common::get_env;

    const char* test_var = "ROCPROFILER_BASH_REGRESSION_TEST";
    const char* test_val = "regression_test_value_12345";

    setenv(test_var, test_val, 1);

    // Verify the variable exists in the actual environ array
    bool        found_in_environ = false;
    std::string environ_value;
    size_t      test_var_len = std::strlen(test_var);

    for(char** env = environ; *env; ++env)
    {
        if(std::strncmp(*env, test_var, test_var_len) == 0 && (*env)[test_var_len] == '=')
        {
            found_in_environ = true;
            environ_value    = *env + test_var_len + 1;
            break;
        }
    }

    // The variable MUST exist in environ (this is the ground truth)
    EXPECT_TRUE(found_in_environ)
        << "Ground truth: Variable missing from environ after setenv() - setenv() failed?";
    EXPECT_EQ(environ_value, test_val)
        << "Ground truth: Variable in environ has wrong value - environ corrupted?";

    // CRITICAL: get_env() must return the value we just verified exists in environ
    // Failure could be: not reading environ, OR implementation bug, OR name parsing broken
    EXPECT_EQ(get_env(test_var, "NOTFOUND"), test_val)
        << "get_env() returned default 'NOTFOUND' instead of '" << test_val
        << "' (which exists in environ)";

    // Verify type overloads work (tests both environ access AND type conversion)
    setenv("ROCPROFILER_BASH_REGRESSION_INT", "42", 1);
    EXPECT_EQ(get_env("ROCPROFILER_BASH_REGRESSION_INT", 0), 42)
        << "get_env<int>() returned default 0 instead of 42 for value '42'";

    setenv("ROCPROFILER_BASH_REGRESSION_BOOL", "true", 1);
    EXPECT_TRUE(get_env("ROCPROFILER_BASH_REGRESSION_BOOL", false))
        << "get_env<bool>() returned false instead of true for value 'true'";

    // Clean up
    unsetenv(test_var);
    unsetenv("ROCPROFILER_BASH_REGRESSION_INT");
    unsetenv("ROCPROFILER_BASH_REGRESSION_BOOL");
}

// Unit test verifying get_env_optional() correctly distinguishes between
// "not set" and "set to empty string".
// This API was added in commit e36a8d128b to provide this distinction.
TEST(common, environment_optional_distinguishes_unset_from_empty)
{
    using rocprofiler::common::get_env_optional;

    const char* test_var = "ROCPROFILER_OPTIONAL_TEST";

    // Test 1: Variable not set - should return std::nullopt
    unsetenv(test_var);  // Ensure it's not set
    auto result_unset = get_env_optional(test_var);
    EXPECT_FALSE(result_unset.has_value()) << "Unset variable should return std::nullopt";

    // Test 2: Variable with VALID NAME but EMPTY VALUE
    // Scenario: export ROCPROFILER_OPTIONAL_TEST=""
    // This should return std::optional("") because the variable EXISTS
    setenv(test_var, "", 1);
    auto result_empty = get_env_optional(test_var);
    EXPECT_TRUE(result_empty.has_value())
        << "Variable exists (even with empty value) should return optional with value";
    EXPECT_TRUE(result_empty->empty()) << "Value should be empty string";
    EXPECT_EQ(*result_empty, "") << "Dereferenced value should be empty string";

    // Test 3: Variable set to non-empty value
    setenv(test_var, "test_value", 1);
    auto result_value = get_env_optional(test_var);
    EXPECT_TRUE(result_value.has_value())
        << "Variable with value should return optional with value";
    EXPECT_FALSE(result_value->empty()) << "Value should not be empty";
    EXPECT_EQ(*result_value, "test_value") << "Value should match what was set";

    // Test 4: EMPTY VARIABLE NAME (invalid query)
    // Scenario: get_env_optional("") - asking for value of a variable with NO NAME
    // Should return std::nullopt because you can't query a variable with no name
    auto result_empty_name = get_env_optional("");
    EXPECT_FALSE(result_empty_name.has_value())
        << "Empty variable name (invalid query) should return std::nullopt";

    // Test 5: Verify it reads from environ directly (like get_env does)
    // This ensures get_env_optional also bypasses custom getenv() implementations
    const char* test_var2 = "ROCPROFILER_OPTIONAL_ENVIRON_TEST";
    setenv(test_var2, "environ_value", 1);

    // Verify it exists in environ array
    bool   found_in_environ = false;
    size_t test_var_len     = std::strlen(test_var2);
    for(char** env = environ; *env; ++env)
    {
        if(std::strncmp(*env, test_var2, test_var_len) == 0 && (*env)[test_var_len] == '=')
        {
            found_in_environ = true;
            break;
        }
    }
    EXPECT_TRUE(found_in_environ) << "Variable should exist in environ";

    // get_env_optional should read it directly from environ
    auto environ_result = get_env_optional(test_var2);
    EXPECT_TRUE(environ_result.has_value());
    EXPECT_EQ(*environ_result, "environ_value")
        << "get_env_optional() should read from environ directly";

    // Clean up
    unsetenv(test_var);
    unsetenv(test_var2);
}
