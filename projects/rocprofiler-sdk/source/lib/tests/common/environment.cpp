// MIT License
//
// Copyright (c) 2023 ROCm Developer Tools
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

#include <gtest/gtest.h>

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
