// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/environment.hpp"

#include <gtest/gtest.h>

#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

using namespace rocprofsys::common;

namespace
{
std::string
find_env_var(const std::vector<std::string>& env, std::string_view var_name)
{
    std::string prefix = std::string(var_name) + "=";
    for(const auto& entry : env)
    {
        if(std::string_view{ entry }.find(prefix) == 0) return entry;
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

    std::vector<std::string>        m_env_vars;
    std::unordered_set<std::string> m_original_envs;
};

TEST_F(RemoveEnvTest, RemoveSingleVariable)
{
    m_env_vars = { "VAR1=value1", "VAR2=value2", "VAR3=value3" };

    remove_env(m_env_vars, "VAR2", m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 2);
    EXPECT_EQ(m_env_vars[0], "VAR1=value1");
    EXPECT_EQ(m_env_vars[1], "VAR3=value3");
}

TEST_F(RemoveEnvTest, RemoveFirstVariable)
{
    m_env_vars = { "FIRST_VAR=first", "SECOND_VAR=second" };

    remove_env(m_env_vars, "FIRST_VAR", m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_EQ(m_env_vars[0], "SECOND_VAR=second");
}

TEST_F(RemoveEnvTest, RemoveLastVariable)
{
    m_env_vars = { "FIRST_VAR=first", "LAST_VAR=last" };

    remove_env(m_env_vars, "LAST_VAR", m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_EQ(m_env_vars[0], "FIRST_VAR=first");
}

TEST_F(RemoveEnvTest, RemoveNonexistentVariable)
{
    m_env_vars = { "EXISTING_VAR=value" };

    remove_env(m_env_vars, "NONEXISTENT_VAR", m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_EQ(m_env_vars[0], "EXISTING_VAR=value");
}

TEST_F(RemoveEnvTest, RemoveFromEmptyVector)
{
    remove_env(m_env_vars, "ANY_VAR", m_original_envs);

    EXPECT_TRUE(m_env_vars.empty());
}

TEST_F(RemoveEnvTest, RemoveOnlyVariable)
{
    m_env_vars = { "ONLY_VAR=only_value" };

    remove_env(m_env_vars, "ONLY_VAR", m_original_envs);

    EXPECT_TRUE(m_env_vars.empty());
}

TEST_F(RemoveEnvTest, RestoreFromOriginalEnvs)
{
    m_original_envs.insert("RESTORE_VAR=original_value");

    m_env_vars = { "RESTORE_VAR=modified_value", "OTHER_VAR=other_value" };

    remove_env(m_env_vars, "RESTORE_VAR", m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 2);
    EXPECT_EQ(find_env_var(m_env_vars, "OTHER_VAR"), "OTHER_VAR=other_value");
    EXPECT_EQ(find_env_var(m_env_vars, "RESTORE_VAR"), "RESTORE_VAR=original_value");
}

TEST_F(RemoveEnvTest, RemoveVariableNotInOriginal_NoRestore)
{
    m_env_vars = { "NEW_VAR=new_value" };

    remove_env(m_env_vars, "NEW_VAR", m_original_envs);

    EXPECT_TRUE(m_env_vars.empty());
}

TEST_F(RemoveEnvTest, RemoveWithSimilarPrefixes)
{
    m_env_vars = { "PATH=/usr/bin", "PATH_EXTRA=/extra", "MYPATH=/my" };

    remove_env(m_env_vars, "PATH", m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 2);
    EXPECT_EQ(m_env_vars[0], "PATH_EXTRA=/extra");
    EXPECT_EQ(m_env_vars[1], "MYPATH=/my");
}

TEST_F(RemoveEnvTest, RemoveMultipleTimes)
{
    m_env_vars = { "A=1", "B=2", "C=3", "D=4" };

    remove_env(m_env_vars, "B", m_original_envs);
    ASSERT_EQ(m_env_vars.size(), 3);

    remove_env(m_env_vars, "D", m_original_envs);
    ASSERT_EQ(m_env_vars.size(), 2);

    EXPECT_EQ(m_env_vars[0], "A=1");
    EXPECT_EQ(m_env_vars[1], "C=3");
}

TEST_F(RemoveEnvTest, RealWorld_LD_PRELOAD)
{
    m_env_vars = { "LD_LIBRARY_PATH=/usr/lib", "LD_PRELOAD=/lib/inject.so",
                   "PATH=/usr/bin" };

    remove_env(m_env_vars, "LD_PRELOAD", m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 2);
    EXPECT_FALSE(find_env_var(m_env_vars, "LD_PRELOAD").length() > 0);
    EXPECT_EQ(find_env_var(m_env_vars, "LD_LIBRARY_PATH"), "LD_LIBRARY_PATH=/usr/lib");
    EXPECT_EQ(find_env_var(m_env_vars, "PATH"), "PATH=/usr/bin");
}

TEST_F(RemoveEnvTest, RealWorld_RestoreROCPROFSYS_Variable)
{
    m_original_envs.insert("ROCPROFSYS_TRACE=false");

    m_env_vars = { "ROCPROFSYS_TRACE=true", "ROCPROFSYS_PROFILE=true" };

    remove_env(m_env_vars, "ROCPROFSYS_TRACE", m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 2);
    EXPECT_EQ(find_env_var(m_env_vars, "ROCPROFSYS_TRACE"), "ROCPROFSYS_TRACE=false");
    EXPECT_EQ(find_env_var(m_env_vars, "ROCPROFSYS_PROFILE"), "ROCPROFSYS_PROFILE=true");
}

TEST_F(RemoveEnvTest, EmptyVariableName)
{
    m_env_vars = { "VAR=value" };

    remove_env(m_env_vars, "", m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_EQ(m_env_vars[0], "VAR=value");
}

TEST_F(RemoveEnvTest, VariableWithEmptyValue)
{
    m_env_vars = { "EMPTY_VALUE=", "NORMAL_VAR=value" };

    remove_env(m_env_vars, "EMPTY_VALUE", m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_EQ(m_env_vars[0], "NORMAL_VAR=value");
}

TEST_F(RemoveEnvTest, VariableWithSpecialCharactersInValue)
{
    m_env_vars = { "SPECIAL=a:b:c:/path/with spaces", "OTHER=value" };

    remove_env(m_env_vars, "SPECIAL", m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_EQ(m_env_vars[0], "OTHER=value");
}

TEST_F(RemoveEnvTest, LongVariableName)
{
    std::string long_var_name = "VERY_LONG_ENVIRONMENT_VARIABLE_NAME_FOR_TESTING";
    m_env_vars                = { long_var_name + "=some_value", "SHORT=val" };

    remove_env(m_env_vars, long_var_name, m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_EQ(m_env_vars[0], "SHORT=val");
}

TEST_F(RemoveEnvTest, RestoreMultipleOriginalValues)
{
    m_original_envs.insert("VAR1=orig1");
    m_original_envs.insert("VAR2=orig2");
    m_original_envs.insert("VAR3=orig3");

    m_env_vars = { "VAR1=modified1", "VAR2=modified2", "VAR3=modified3" };

    remove_env(m_env_vars, "VAR2", m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 3);
    EXPECT_EQ(find_env_var(m_env_vars, "VAR1"), "VAR1=modified1");
    EXPECT_EQ(find_env_var(m_env_vars, "VAR2"), "VAR2=orig2");
    EXPECT_EQ(find_env_var(m_env_vars, "VAR3"), "VAR3=modified3");
}

TEST_F(RemoveEnvTest, CaseSensitiveRemoval)
{
    m_env_vars = { "MyVar=value1", "MYVAR=value2", "myvar=value3" };

    remove_env(m_env_vars, "MYVAR", m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 2);
    EXPECT_EQ(m_env_vars[0], "MyVar=value1");
    EXPECT_EQ(m_env_vars[1], "myvar=value3");
}
