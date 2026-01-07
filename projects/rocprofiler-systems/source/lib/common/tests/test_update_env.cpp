// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "common/environment.hpp"

#include <cstring>
#include <gtest/gtest.h>
#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

using namespace rocprofsys::common;

static std::string
find_env_var(const std::vector<char*>& env, std::string_view var_name)
{
    std::string prefix = std::string(var_name) + "=";
    for(auto* itr : env)
    {
        if(!itr) continue;
        if(std::string_view{ itr }.find(prefix) == 0)
        {
            return std::string{ itr };
        }
    }
    return "";
}

class UpdateEnvTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_env_vars.clear();
        m_updated_envs.clear();
        m_original_envs.clear();
    }

    void TearDown() override
    {
        for(auto* ptr : m_env_vars)
        {
            if(ptr) free(ptr);
        }
    }

    std::vector<char*>                   m_env_vars;
    std::unordered_set<std::string_view> m_updated_envs;
    std::unordered_set<std::string>      m_original_envs;
};

TEST_F(UpdateEnvTest, ReplaceMode_NewVariable)
{
    update_env(m_env_vars, "TEST_VAR", "test_value", update_mode::REPLACE, ":",
               m_updated_envs, m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_STREQ(m_env_vars[0], "TEST_VAR=test_value");
    EXPECT_EQ(m_updated_envs.count("TEST_VAR"), 1);
}

TEST_F(UpdateEnvTest, ReplaceMode_ExistingVariable)
{
    m_env_vars.push_back(strdup("TEST_VAR=old_value"));
    m_original_envs.insert("TEST_VAR=old_value");

    update_env(m_env_vars, "TEST_VAR", "new_value", update_mode::REPLACE, ":",
               m_updated_envs, m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_STREQ(m_env_vars[0], "TEST_VAR=new_value");
    EXPECT_EQ(m_updated_envs.count("TEST_VAR"), 1);
}

TEST_F(UpdateEnvTest, AppendMode_NewVariable)
{
    update_env(m_env_vars, "PATH", "/new/path", update_mode::APPEND, ":", m_updated_envs,
               m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_STREQ(m_env_vars[0], "PATH=/new/path");
}

TEST_F(UpdateEnvTest, AppendMode_ExistingVariable)
{
    m_env_vars.push_back(strdup("PATH=/old/path"));
    m_original_envs.insert("PATH=/old/path");

    update_env(m_env_vars, "PATH", "/new/path", update_mode::APPEND, ":", m_updated_envs,
               m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_STREQ(m_env_vars[0], "PATH=/old/path:/new/path");
    EXPECT_EQ(m_updated_envs.count("PATH"), 1);
}

TEST_F(UpdateEnvTest, PrependMode_ExistingVariable)
{
    m_env_vars.push_back(strdup("LD_LIBRARY_PATH=/old/lib"));
    m_original_envs.insert("LD_LIBRARY_PATH=/old/lib");

    update_env(m_env_vars, "LD_LIBRARY_PATH", "/new/lib", update_mode::PREPEND, ":",
               m_updated_envs, m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_STREQ(m_env_vars[0], "LD_LIBRARY_PATH=/new/lib:/old/lib");
}

TEST_F(UpdateEnvTest, WeakMode_OriginalValue)
{
    m_env_vars.push_back(strdup("WEAK_VAR=original"));
    m_original_envs.insert("WEAK_VAR=original");

    update_env(m_env_vars, "WEAK_VAR", "new_value", update_mode::WEAK, ":",
               m_updated_envs, m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_STREQ(m_env_vars[0], "WEAK_VAR=new_value");
}

TEST_F(UpdateEnvTest, WeakMode_ModifiedValue)
{
    m_env_vars.push_back(strdup("WEAK_VAR=original"));
    m_original_envs.insert("WEAK_VAR=original");

    free(m_env_vars[0]);
    m_env_vars[0] = strdup("WEAK_VAR=modified");

    update_env(m_env_vars, "WEAK_VAR", "new_value", update_mode::WEAK, ":",
               m_updated_envs, m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_STREQ(m_env_vars[0], "WEAK_VAR=modified");
}

TEST_F(UpdateEnvTest, BooleanValue_True)
{
    update_env(m_env_vars, "BOOL_VAR", true, update_mode::REPLACE, ":", m_updated_envs,
               m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_STREQ(m_env_vars[0], "BOOL_VAR=true");
}

TEST_F(UpdateEnvTest, BooleanValue_False)
{
    update_env(m_env_vars, "BOOL_VAR", false, update_mode::REPLACE, ":", m_updated_envs,
               m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_STREQ(m_env_vars[0], "BOOL_VAR=false");
}

TEST_F(UpdateEnvTest, NumericValue)
{
    update_env(m_env_vars, "NUM_VAR", 42, update_mode::REPLACE, ":", m_updated_envs,
               m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_STREQ(m_env_vars[0], "NUM_VAR=42");
}

TEST_F(UpdateEnvTest, AppendMode_AvoidsDuplicates)
{
    m_env_vars.push_back(strdup("PATH=/existing/path"));
    m_original_envs.insert("PATH=/existing/path");

    update_env(m_env_vars, "PATH", "/existing/path", update_mode::APPEND, ":",
               m_updated_envs, m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_STREQ(m_env_vars[0], "PATH=/existing/path");
}

TEST_F(UpdateEnvTest, CustomDelimiter)
{
    m_env_vars.push_back(strdup("VAR=a"));
    m_original_envs.insert("VAR=a");

    update_env(m_env_vars, "VAR", "b", update_mode::APPEND, ",", m_updated_envs,
               m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_STREQ(m_env_vars[0], "VAR=a,b");
}

TEST_F(UpdateEnvTest, RealWorld_LD_LIBRARY_PATH_Append)
{
    m_env_vars.push_back(strdup("LD_LIBRARY_PATH=/usr/lib:/usr/local/lib"));
    m_original_envs.insert("LD_LIBRARY_PATH=/usr/lib:/usr/local/lib");

    update_env(m_env_vars, "LD_LIBRARY_PATH", "/opt/rocm/lib", update_mode::APPEND, ":",
               m_updated_envs, m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_STREQ(m_env_vars[0], "LD_LIBRARY_PATH=/usr/lib:/usr/local/lib:/opt/rocm/lib");
}

TEST_F(UpdateEnvTest, RealWorld_LD_PRELOAD_Prepend)
{
    m_env_vars.push_back(strdup("LD_PRELOAD=/lib/existing.so"));
    m_original_envs.insert("LD_PRELOAD=/lib/existing.so");

    update_env(m_env_vars, "LD_PRELOAD", "/opt/rocm/librocprof-sys-dl.so",
               update_mode::PREPEND, ":", m_updated_envs, m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_STREQ(m_env_vars[0],
                 "LD_PRELOAD=/opt/rocm/librocprof-sys-dl.so:/lib/existing.so");
}

TEST_F(UpdateEnvTest, RealWorld_ROCPROFSYS_Environment_Variables)
{
    update_env(m_env_vars, "ROCPROFSYS_TRACE", true, update_mode::REPLACE, ":",
               m_updated_envs, m_original_envs);
    update_env(m_env_vars, "ROCPROFSYS_PROFILE", false, update_mode::REPLACE, ":",
               m_updated_envs, m_original_envs);
    update_env(m_env_vars, "ROCPROFSYS_USE_SAMPLING", true, update_mode::REPLACE, ":",
               m_updated_envs, m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 3);
    EXPECT_STREQ(find_env_var(m_env_vars, "ROCPROFSYS_TRACE").c_str(),
                 "ROCPROFSYS_TRACE=true");
    EXPECT_STREQ(find_env_var(m_env_vars, "ROCPROFSYS_PROFILE").c_str(),
                 "ROCPROFSYS_PROFILE=false");
    EXPECT_STREQ(find_env_var(m_env_vars, "ROCPROFSYS_USE_SAMPLING").c_str(),
                 "ROCPROFSYS_USE_SAMPLING=true");
}

TEST_F(UpdateEnvTest, RealWorld_Timing_DoubleValues)
{
    update_env(m_env_vars, "ROCPROFSYS_TRACE_DELAY", 1.5, update_mode::REPLACE, ":",
               m_updated_envs, m_original_envs);
    update_env(m_env_vars, "ROCPROFSYS_SAMPLING_FREQ", 100.0, update_mode::REPLACE, ":",
               m_updated_envs, m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 2);
    std::string delay_var = find_env_var(m_env_vars, "ROCPROFSYS_TRACE_DELAY");
    std::string freq_var  = find_env_var(m_env_vars, "ROCPROFSYS_SAMPLING_FREQ");

    EXPECT_TRUE(delay_var.find("ROCPROFSYS_TRACE_DELAY=") == 0);
    EXPECT_TRUE(freq_var.find("ROCPROFSYS_SAMPLING_FREQ=") == 0);
}

TEST_F(UpdateEnvTest, StringTypes_StdString)
{
    std::string value = "test_string_value";
    update_env(m_env_vars, "STRING_VAR", value, update_mode::REPLACE, ":", m_updated_envs,
               m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_STREQ(m_env_vars[0], "STRING_VAR=test_string_value");
}

TEST_F(UpdateEnvTest, StringTypes_ConstCharPtr)
{
    const char* value = "const_char_value";
    update_env(m_env_vars, "CHAR_VAR", value, update_mode::REPLACE, ":", m_updated_envs,
               m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_STREQ(m_env_vars[0], "CHAR_VAR=const_char_value");
}

TEST_F(UpdateEnvTest, EmptyStringValue)
{
    update_env(m_env_vars, "EMPTY_VAR", "", update_mode::REPLACE, ":", m_updated_envs,
               m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_STREQ(m_env_vars[0], "EMPTY_VAR=");
}

TEST_F(UpdateEnvTest, MultipleVariables_DifferentNames)
{
    update_env(m_env_vars, "VAR1", "value1", update_mode::REPLACE, ":", m_updated_envs,
               m_original_envs);
    update_env(m_env_vars, "VAR2", "value2", update_mode::REPLACE, ":", m_updated_envs,
               m_original_envs);
    update_env(m_env_vars, "VAR3", "value3", update_mode::REPLACE, ":", m_updated_envs,
               m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 3);
    EXPECT_EQ(m_updated_envs.size(), 3);
    EXPECT_STREQ(find_env_var(m_env_vars, "VAR1").c_str(), "VAR1=value1");
    EXPECT_STREQ(find_env_var(m_env_vars, "VAR2").c_str(), "VAR2=value2");
    EXPECT_STREQ(find_env_var(m_env_vars, "VAR3").c_str(), "VAR3=value3");
}

TEST_F(UpdateEnvTest, NullPointer_InEnvironmentVector)
{
    m_env_vars.push_back(strdup("VAR1=value1"));
    m_env_vars.push_back(nullptr);
    m_env_vars.push_back(strdup("VAR2=value2"));
    m_original_envs.insert("VAR1=value1");
    m_original_envs.insert("VAR2=value2");

    update_env(m_env_vars, "VAR2", "new_value2", update_mode::REPLACE, ":",
               m_updated_envs, m_original_envs);

    EXPECT_STREQ(find_env_var(m_env_vars, "VAR2").c_str(), "VAR2=new_value2");
}

TEST_F(UpdateEnvTest, LongPath_Append)
{
    std::string long_path = "/very/long/path/to/some/directory/with/many/subdirectories/"
                            "that/might/be/used/in/real/world";
    m_env_vars.push_back(strdup("PATH=/usr/bin:/bin"));
    m_original_envs.insert("PATH=/usr/bin:/bin");

    update_env(m_env_vars, "PATH", long_path, update_mode::APPEND, ":", m_updated_envs,
               m_original_envs);

    std::string expected = "PATH=/usr/bin:/bin:" + long_path;
    EXPECT_STREQ(m_env_vars[0], expected.c_str());
}

TEST_F(UpdateEnvTest, SpecialCharacters_InValue)
{
    update_env(m_env_vars, "SPECIAL_VAR", "value-with_special.chars:123",
               update_mode::REPLACE, ":", m_updated_envs, m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_STREQ(m_env_vars[0], "SPECIAL_VAR=value-with_special.chars:123");
}

TEST_F(UpdateEnvTest, IntegerValues_Positive)
{
    update_env(m_env_vars, "INT_VAR", 12345, update_mode::REPLACE, ":", m_updated_envs,
               m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_STREQ(m_env_vars[0], "INT_VAR=12345");
}

TEST_F(UpdateEnvTest, IntegerValues_Negative)
{
    update_env(m_env_vars, "NEGATIVE_VAR", -999, update_mode::REPLACE, ":",
               m_updated_envs, m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_STREQ(m_env_vars[0], "NEGATIVE_VAR=-999");
}

TEST_F(UpdateEnvTest, IntegerValues_Zero)
{
    update_env(m_env_vars, "ZERO_VAR", 0, update_mode::REPLACE, ":", m_updated_envs,
               m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_STREQ(m_env_vars[0], "ZERO_VAR=0");
}

TEST_F(UpdateEnvTest, UpdateTracking_MultipleUpdates)
{
    update_env(m_env_vars, "VAR1", "val1", update_mode::REPLACE, ":", m_updated_envs,
               m_original_envs);
    update_env(m_env_vars, "VAR2", "val2", update_mode::REPLACE, ":", m_updated_envs,
               m_original_envs);
    update_env(m_env_vars, "VAR1", "val1_updated", update_mode::REPLACE, ":",
               m_updated_envs, m_original_envs);

    EXPECT_EQ(m_updated_envs.count("VAR1"), 1);
    EXPECT_EQ(m_updated_envs.count("VAR2"), 1);
    EXPECT_EQ(m_updated_envs.size(), 2);
}

TEST_F(UpdateEnvTest, WeakMode_SequentialUpdates)
{
    m_env_vars.push_back(strdup("CONFIG_VAR=initial"));
    m_original_envs.insert("CONFIG_VAR=initial");

    update_env(m_env_vars, "CONFIG_VAR", "weak_update", update_mode::WEAK, ":",
               m_updated_envs, m_original_envs);

    EXPECT_STREQ(m_env_vars[0], "CONFIG_VAR=weak_update");

    free(m_env_vars[0]);
    m_env_vars[0] = strdup("CONFIG_VAR=user_modified");

    update_env(m_env_vars, "CONFIG_VAR", "another_weak_update", update_mode::WEAK, ":",
               m_updated_envs, m_original_envs);

    EXPECT_STREQ(m_env_vars[0], "CONFIG_VAR=user_modified");
}

TEST_F(UpdateEnvTest, Append_MultiplePathsInSequence)
{
    update_env(m_env_vars, "BUILD_PATH", "/path1", update_mode::REPLACE, ":",
               m_updated_envs, m_original_envs);
    update_env(m_env_vars, "BUILD_PATH", "/path2", update_mode::APPEND, ":",
               m_updated_envs, m_original_envs);
    update_env(m_env_vars, "BUILD_PATH", "/path3", update_mode::APPEND, ":",
               m_updated_envs, m_original_envs);
    update_env(m_env_vars, "BUILD_PATH", "/path4", update_mode::APPEND, ":",
               m_updated_envs, m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_STREQ(m_env_vars[0], "BUILD_PATH=/path1:/path2:/path3:/path4");
}