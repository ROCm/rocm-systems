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

namespace
{
std::string
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
}  // namespace

class RemoveEnvTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_env_vars.clear();
        m_original_envs.clear();
    }

    void TearDown() override
    {
        for(auto* ptr : m_env_vars)
        {
            if(ptr) free(ptr);
        }
    }

    std::vector<char*>              m_env_vars;
    std::unordered_set<std::string> m_original_envs;
};

TEST_F(RemoveEnvTest, RemoveSingleVariable)
{
    m_env_vars.push_back(strdup("VAR1=value1"));
    m_env_vars.push_back(strdup("VAR2=value2"));
    m_env_vars.push_back(strdup("VAR3=value3"));

    remove_env(m_env_vars, "VAR2", m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 2);
    EXPECT_STREQ(m_env_vars[0], "VAR1=value1");
    EXPECT_STREQ(m_env_vars[1], "VAR3=value3");
}

TEST_F(RemoveEnvTest, RemoveFirstVariable)
{
    m_env_vars.push_back(strdup("FIRST_VAR=first"));
    m_env_vars.push_back(strdup("SECOND_VAR=second"));

    remove_env(m_env_vars, "FIRST_VAR", m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_STREQ(m_env_vars[0], "SECOND_VAR=second");
}

TEST_F(RemoveEnvTest, RemoveLastVariable)
{
    m_env_vars.push_back(strdup("FIRST_VAR=first"));
    m_env_vars.push_back(strdup("LAST_VAR=last"));

    remove_env(m_env_vars, "LAST_VAR", m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_STREQ(m_env_vars[0], "FIRST_VAR=first");
}

TEST_F(RemoveEnvTest, RemoveNonexistentVariable)
{
    m_env_vars.push_back(strdup("EXISTING_VAR=value"));

    remove_env(m_env_vars, "NONEXISTENT_VAR", m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_STREQ(m_env_vars[0], "EXISTING_VAR=value");
}

TEST_F(RemoveEnvTest, RemoveFromEmptyVector)
{
    remove_env(m_env_vars, "ANY_VAR", m_original_envs);

    EXPECT_TRUE(m_env_vars.empty());
}

TEST_F(RemoveEnvTest, RemoveOnlyVariable)
{
    m_env_vars.push_back(strdup("ONLY_VAR=only_value"));

    remove_env(m_env_vars, "ONLY_VAR", m_original_envs);

    EXPECT_TRUE(m_env_vars.empty());
}

TEST_F(RemoveEnvTest, RestoreFromOriginalEnvs)
{
    m_original_envs.insert("RESTORE_VAR=original_value");

    m_env_vars.push_back(strdup("RESTORE_VAR=modified_value"));
    m_env_vars.push_back(strdup("OTHER_VAR=other_value"));

    remove_env(m_env_vars, "RESTORE_VAR", m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 2);
    EXPECT_STREQ(find_env_var(m_env_vars, "OTHER_VAR").c_str(), "OTHER_VAR=other_value");
    EXPECT_STREQ(find_env_var(m_env_vars, "RESTORE_VAR").c_str(),
                 "RESTORE_VAR=original_value");
}

TEST_F(RemoveEnvTest, RemoveVariableNotInOriginal_NoRestore)
{
    m_env_vars.push_back(strdup("NEW_VAR=new_value"));

    remove_env(m_env_vars, "NEW_VAR", m_original_envs);

    EXPECT_TRUE(m_env_vars.empty());
}

TEST_F(RemoveEnvTest, RemoveWithSimilarPrefixes)
{
    m_env_vars.push_back(strdup("PATH=/usr/bin"));
    m_env_vars.push_back(strdup("PATH_EXTRA=/extra"));
    m_env_vars.push_back(strdup("MYPATH=/my"));

    remove_env(m_env_vars, "PATH", m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 2);
    EXPECT_STREQ(m_env_vars[0], "PATH_EXTRA=/extra");
    EXPECT_STREQ(m_env_vars[1], "MYPATH=/my");
}

TEST_F(RemoveEnvTest, RemoveWithNullEntries)
{
    m_env_vars.push_back(strdup("VAR1=value1"));
    m_env_vars.push_back(nullptr);
    m_env_vars.push_back(strdup("VAR2=value2"));
    m_env_vars.push_back(nullptr);
    m_env_vars.push_back(strdup("VAR3=value3"));

    remove_env(m_env_vars, "VAR2", m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 2);
    EXPECT_STREQ(m_env_vars[0], "VAR1=value1");
    EXPECT_STREQ(m_env_vars[1], "VAR3=value3");
}

TEST_F(RemoveEnvTest, RemoveMultipleTimes)
{
    m_env_vars.push_back(strdup("A=1"));
    m_env_vars.push_back(strdup("B=2"));
    m_env_vars.push_back(strdup("C=3"));
    m_env_vars.push_back(strdup("D=4"));

    remove_env(m_env_vars, "B", m_original_envs);
    ASSERT_EQ(m_env_vars.size(), 3);

    remove_env(m_env_vars, "D", m_original_envs);
    ASSERT_EQ(m_env_vars.size(), 2);

    EXPECT_STREQ(m_env_vars[0], "A=1");
    EXPECT_STREQ(m_env_vars[1], "C=3");
}

TEST_F(RemoveEnvTest, RealWorld_LD_PRELOAD)
{
    m_env_vars.push_back(strdup("LD_LIBRARY_PATH=/usr/lib"));
    m_env_vars.push_back(strdup("LD_PRELOAD=/lib/inject.so"));
    m_env_vars.push_back(strdup("PATH=/usr/bin"));

    remove_env(m_env_vars, "LD_PRELOAD", m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 2);
    EXPECT_FALSE(find_env_var(m_env_vars, "LD_PRELOAD").length() > 0);
    EXPECT_STREQ(find_env_var(m_env_vars, "LD_LIBRARY_PATH").c_str(),
                 "LD_LIBRARY_PATH=/usr/lib");
    EXPECT_STREQ(find_env_var(m_env_vars, "PATH").c_str(), "PATH=/usr/bin");
}

TEST_F(RemoveEnvTest, RealWorld_RestoreROCPROFSYS_Variable)
{
    m_original_envs.insert("ROCPROFSYS_TRACE=false");

    m_env_vars.push_back(strdup("ROCPROFSYS_TRACE=true"));
    m_env_vars.push_back(strdup("ROCPROFSYS_PROFILE=true"));

    remove_env(m_env_vars, "ROCPROFSYS_TRACE", m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 2);
    EXPECT_STREQ(find_env_var(m_env_vars, "ROCPROFSYS_TRACE").c_str(),
                 "ROCPROFSYS_TRACE=false");
    EXPECT_STREQ(find_env_var(m_env_vars, "ROCPROFSYS_PROFILE").c_str(),
                 "ROCPROFSYS_PROFILE=true");
}

TEST_F(RemoveEnvTest, EmptyVariableName)
{
    m_env_vars.push_back(strdup("VAR=value"));

    remove_env(m_env_vars, "", m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_STREQ(m_env_vars[0], "VAR=value");
}

TEST_F(RemoveEnvTest, VariableWithEmptyValue)
{
    m_env_vars.push_back(strdup("EMPTY_VALUE="));
    m_env_vars.push_back(strdup("NORMAL_VAR=value"));

    remove_env(m_env_vars, "EMPTY_VALUE", m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_STREQ(m_env_vars[0], "NORMAL_VAR=value");
}

TEST_F(RemoveEnvTest, VariableWithSpecialCharactersInValue)
{
    m_env_vars.push_back(strdup("SPECIAL=a:b:c:/path/with spaces"));
    m_env_vars.push_back(strdup("OTHER=value"));

    remove_env(m_env_vars, "SPECIAL", m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_STREQ(m_env_vars[0], "OTHER=value");
}

TEST_F(RemoveEnvTest, LongVariableName)
{
    std::string long_var_name = "VERY_LONG_ENVIRONMENT_VARIABLE_NAME_FOR_TESTING";
    std::string entry         = long_var_name + "=some_value";
    m_env_vars.push_back(strdup(entry.c_str()));
    m_env_vars.push_back(strdup("SHORT=val"));

    remove_env(m_env_vars, long_var_name, m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_STREQ(m_env_vars[0], "SHORT=val");
}

TEST_F(RemoveEnvTest, RestoreMultipleOriginalValues)
{
    m_original_envs.insert("VAR1=orig1");
    m_original_envs.insert("VAR2=orig2");
    m_original_envs.insert("VAR3=orig3");

    m_env_vars.push_back(strdup("VAR1=modified1"));
    m_env_vars.push_back(strdup("VAR2=modified2"));
    m_env_vars.push_back(strdup("VAR3=modified3"));

    remove_env(m_env_vars, "VAR2", m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 3);
    EXPECT_STREQ(find_env_var(m_env_vars, "VAR1").c_str(), "VAR1=modified1");
    EXPECT_STREQ(find_env_var(m_env_vars, "VAR2").c_str(), "VAR2=orig2");
    EXPECT_STREQ(find_env_var(m_env_vars, "VAR3").c_str(), "VAR3=modified3");
}

TEST_F(RemoveEnvTest, CaseSensitiveRemoval)
{
    m_env_vars.push_back(strdup("MyVar=value1"));
    m_env_vars.push_back(strdup("MYVAR=value2"));
    m_env_vars.push_back(strdup("myvar=value3"));

    remove_env(m_env_vars, "MYVAR", m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 2);
    EXPECT_STREQ(m_env_vars[0], "MyVar=value1");
    EXPECT_STREQ(m_env_vars[1], "myvar=value3");
}
