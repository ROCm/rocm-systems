// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/environment.hpp"

#include <gtest/gtest.h>

#include <string>
#include <string_view>
#include <unordered_set>
#include <vector>

using namespace rocprofsys::common;
namespace env_vars = rocprofsys::env_vars;

static std::string
find_env_var(const std::vector<std::string>& env, std::string_view var_name)
{
    std::string prefix = std::string(var_name) + "=";
    for(const auto& entry : env)
    {
        if(std::string_view{ entry }.find(prefix) == 0) return entry;
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

    std::vector<std::string>             m_env_vars;
    std::unordered_set<std::string_view> m_updated_envs;
    std::unordered_set<std::string>      m_original_envs;
};

TEST_F(UpdateEnvTest, ReplaceMode_NewVariable)
{
    update_env(m_env_vars, "TEST_VAR", "test_value", update_mode::REPLACE, ":",
               m_updated_envs, m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_EQ(m_env_vars[0], "TEST_VAR=test_value");
    EXPECT_EQ(m_updated_envs.count("TEST_VAR"), 1);
}

TEST_F(UpdateEnvTest, ReplaceMode_ExistingVariable)
{
    m_env_vars.emplace_back("TEST_VAR=old_value");
    m_original_envs.insert("TEST_VAR=old_value");

    update_env(m_env_vars, "TEST_VAR", "new_value", update_mode::REPLACE, ":",
               m_updated_envs, m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_EQ(m_env_vars[0], "TEST_VAR=new_value");
    EXPECT_EQ(m_updated_envs.count("TEST_VAR"), 1);
}

TEST_F(UpdateEnvTest, ReplaceMode_RemovesDuplicateEntries)
{
    // Same key present twice (e.g. one from shell env, one from config file).
    // REPLACE must leave exactly one entry; otherwise consolidate_env_entries
    // later joins the parts and turns REPLACE into APPEND.
    m_env_vars.emplace_back(std::string{ env_vars::TRACE } + "=true");
    m_env_vars.emplace_back("OTHER_VAR=keep");
    m_env_vars.emplace_back(std::string{ env_vars::TRACE } + "=true");

    update_env(m_env_vars, env_vars::TRACE, false, update_mode::REPLACE, ":",
               m_updated_envs, m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 2);
    EXPECT_EQ(m_env_vars[0], std::string{ env_vars::TRACE } + "=false");
    EXPECT_EQ(m_env_vars[1], "OTHER_VAR=keep");
}

TEST_F(UpdateEnvTest, ReplaceMode_RemovesAllDuplicatesWhenManyExist)
{
    m_env_vars.emplace_back("DUP=a");
    m_env_vars.emplace_back("DUP=b");
    m_env_vars.emplace_back("DUP=c");
    m_env_vars.emplace_back("KEEP=x");

    update_env(m_env_vars, "DUP", "final", update_mode::REPLACE, ":", m_updated_envs,
               m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 2);
    EXPECT_EQ(m_env_vars[0], "DUP=final");
    EXPECT_EQ(m_env_vars[1], "KEEP=x");
}

TEST_F(UpdateEnvTest, AppendMode_NewVariable)
{
    update_env(m_env_vars, "PATH", "/new/path", update_mode::APPEND, ":", m_updated_envs,
               m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_EQ(m_env_vars[0], "PATH=/new/path");
}

TEST_F(UpdateEnvTest, AppendMode_ExistingVariable)
{
    m_env_vars.emplace_back("PATH=/old/path");
    m_original_envs.insert("PATH=/old/path");

    update_env(m_env_vars, "PATH", "/new/path", update_mode::APPEND, ":", m_updated_envs,
               m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_EQ(m_env_vars[0], "PATH=/old/path:/new/path");
    EXPECT_EQ(m_updated_envs.count("PATH"), 1);
}

TEST_F(UpdateEnvTest, PrependMode_ExistingVariable)
{
    m_env_vars.emplace_back("LD_LIBRARY_PATH=/old/lib");
    m_original_envs.insert("LD_LIBRARY_PATH=/old/lib");

    update_env(m_env_vars, "LD_LIBRARY_PATH", "/new/lib", update_mode::PREPEND, ":",
               m_updated_envs, m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_EQ(m_env_vars[0], "LD_LIBRARY_PATH=/new/lib:/old/lib");
}

TEST_F(UpdateEnvTest, WeakMode_OriginalValue)
{
    m_env_vars.emplace_back("WEAK_VAR=original");
    m_original_envs.insert("WEAK_VAR=original");

    update_env(m_env_vars, "WEAK_VAR", "new_value", update_mode::WEAK, ":",
               m_updated_envs, m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_EQ(m_env_vars[0], "WEAK_VAR=new_value");
}

TEST_F(UpdateEnvTest, WeakMode_ModifiedValue)
{
    m_env_vars.emplace_back("WEAK_VAR=original");
    m_original_envs.insert("WEAK_VAR=original");

    m_env_vars[0] = "WEAK_VAR=modified";

    update_env(m_env_vars, "WEAK_VAR", "new_value", update_mode::WEAK, ":",
               m_updated_envs, m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_EQ(m_env_vars[0], "WEAK_VAR=modified");
}

TEST_F(UpdateEnvTest, BooleanValue_True)
{
    update_env(m_env_vars, "BOOL_VAR", true, update_mode::REPLACE, ":", m_updated_envs,
               m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_EQ(m_env_vars[0], "BOOL_VAR=true");
}

TEST_F(UpdateEnvTest, BooleanValue_False)
{
    update_env(m_env_vars, "BOOL_VAR", false, update_mode::REPLACE, ":", m_updated_envs,
               m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_EQ(m_env_vars[0], "BOOL_VAR=false");
}

TEST_F(UpdateEnvTest, NumericValue)
{
    update_env(m_env_vars, "NUM_VAR", 42, update_mode::REPLACE, ":", m_updated_envs,
               m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_EQ(m_env_vars[0], "NUM_VAR=42");
}

TEST_F(UpdateEnvTest, AppendMode_AvoidsDuplicates)
{
    m_env_vars.emplace_back("PATH=/existing/path");
    m_original_envs.insert("PATH=/existing/path");

    update_env(m_env_vars, "PATH", "/existing/path", update_mode::APPEND, ":",
               m_updated_envs, m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_EQ(m_env_vars[0], "PATH=/existing/path");
}

TEST_F(UpdateEnvTest, CustomDelimiter)
{
    m_env_vars.emplace_back("VAR=a");
    m_original_envs.insert("VAR=a");

    update_env(m_env_vars, "VAR", "b", update_mode::APPEND, ",", m_updated_envs,
               m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_EQ(m_env_vars[0], "VAR=a,b");
}

TEST_F(UpdateEnvTest, RealWorld_LD_LIBRARY_PATH_Append)
{
    m_env_vars.emplace_back("LD_LIBRARY_PATH=/usr/lib:/usr/local/lib");
    m_original_envs.insert("LD_LIBRARY_PATH=/usr/lib:/usr/local/lib");

    update_env(m_env_vars, "LD_LIBRARY_PATH", "/opt/rocm/lib", update_mode::APPEND, ":",
               m_updated_envs, m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_EQ(m_env_vars[0], "LD_LIBRARY_PATH=/usr/lib:/usr/local/lib:/opt/rocm/lib");
}

TEST_F(UpdateEnvTest, RealWorld_LD_PRELOAD_Prepend)
{
    m_env_vars.emplace_back("LD_PRELOAD=/lib/existing.so");
    m_original_envs.insert("LD_PRELOAD=/lib/existing.so");

    update_env(m_env_vars, "LD_PRELOAD", "/opt/rocm/librocprof-sys-dl.so",
               update_mode::PREPEND, ":", m_updated_envs, m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_EQ(m_env_vars[0],
              "LD_PRELOAD=/opt/rocm/librocprof-sys-dl.so:/lib/existing.so");
}

TEST_F(UpdateEnvTest, RealWorld_ROCPROFSYS_Environment_Variables)
{
    update_env(m_env_vars, env_vars::TRACE, true, update_mode::REPLACE, ":",
               m_updated_envs, m_original_envs);
    update_env(m_env_vars, env_vars::PROFILE, false, update_mode::REPLACE, ":",
               m_updated_envs, m_original_envs);
    update_env(m_env_vars, env_vars::USE_SAMPLING, true, update_mode::REPLACE, ":",
               m_updated_envs, m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 3);
    EXPECT_EQ(find_env_var(m_env_vars, env_vars::TRACE),
              std::string{ env_vars::TRACE } + "=true");
    EXPECT_EQ(find_env_var(m_env_vars, env_vars::PROFILE),
              std::string{ env_vars::PROFILE } + "=false");
    EXPECT_EQ(find_env_var(m_env_vars, env_vars::USE_SAMPLING),
              std::string{ env_vars::USE_SAMPLING } + "=true");
}

TEST_F(UpdateEnvTest, RealWorld_Timing_DoubleValues)
{
    update_env(m_env_vars, env_vars::TRACE_DELAY, 1.5, update_mode::REPLACE, ":",
               m_updated_envs, m_original_envs);
    update_env(m_env_vars, env_vars::SAMPLING_FREQ, 100.0, update_mode::REPLACE, ":",
               m_updated_envs, m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 2);
    std::string delay_var = find_env_var(m_env_vars, env_vars::TRACE_DELAY);
    std::string freq_var  = find_env_var(m_env_vars, env_vars::SAMPLING_FREQ);

    EXPECT_TRUE(delay_var.find(std::string{ env_vars::TRACE_DELAY } + "=") == 0);
    EXPECT_TRUE(freq_var.find(std::string{ env_vars::SAMPLING_FREQ } + "=") == 0);
}

TEST_F(UpdateEnvTest, StringTypes_StdString)
{
    std::string value = "test_string_value";
    update_env(m_env_vars, "STRING_VAR", value, update_mode::REPLACE, ":", m_updated_envs,
               m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_EQ(m_env_vars[0], "STRING_VAR=test_string_value");
}

TEST_F(UpdateEnvTest, StringTypes_ConstCharPtr)
{
    const char* value = "const_char_value";
    update_env(m_env_vars, "CHAR_VAR", value, update_mode::REPLACE, ":", m_updated_envs,
               m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_EQ(m_env_vars[0], "CHAR_VAR=const_char_value");
}

TEST_F(UpdateEnvTest, EmptyStringValue)
{
    update_env(m_env_vars, "EMPTY_VAR", "", update_mode::REPLACE, ":", m_updated_envs,
               m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_EQ(m_env_vars[0], "EMPTY_VAR=");
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
    EXPECT_EQ(find_env_var(m_env_vars, "VAR1"), "VAR1=value1");
    EXPECT_EQ(find_env_var(m_env_vars, "VAR2"), "VAR2=value2");
    EXPECT_EQ(find_env_var(m_env_vars, "VAR3"), "VAR3=value3");
}

TEST_F(UpdateEnvTest, LongPath_Append)
{
    std::string long_path = "/very/long/path/to/some/directory/with/many/subdirectories/"
                            "that/might/be/used/in/real/world";
    m_env_vars.emplace_back("PATH=/usr/bin:/bin");
    m_original_envs.insert("PATH=/usr/bin:/bin");

    update_env(m_env_vars, "PATH", long_path, update_mode::APPEND, ":", m_updated_envs,
               m_original_envs);

    std::string expected = "PATH=/usr/bin:/bin:" + long_path;
    EXPECT_EQ(m_env_vars[0], expected);
}

TEST_F(UpdateEnvTest, SpecialCharacters_InValue)
{
    update_env(m_env_vars, "SPECIAL_VAR", "value-with_special.chars:123",
               update_mode::REPLACE, ":", m_updated_envs, m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_EQ(m_env_vars[0], "SPECIAL_VAR=value-with_special.chars:123");
}

TEST_F(UpdateEnvTest, IntegerValues_Positive)
{
    update_env(m_env_vars, "INT_VAR", 12345, update_mode::REPLACE, ":", m_updated_envs,
               m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_EQ(m_env_vars[0], "INT_VAR=12345");
}

TEST_F(UpdateEnvTest, IntegerValues_Negative)
{
    update_env(m_env_vars, "NEGATIVE_VAR", -999, update_mode::REPLACE, ":",
               m_updated_envs, m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_EQ(m_env_vars[0], "NEGATIVE_VAR=-999");
}

TEST_F(UpdateEnvTest, IntegerValues_Zero)
{
    update_env(m_env_vars, "ZERO_VAR", 0, update_mode::REPLACE, ":", m_updated_envs,
               m_original_envs);

    ASSERT_EQ(m_env_vars.size(), 1);
    EXPECT_EQ(m_env_vars[0], "ZERO_VAR=0");
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
    m_env_vars.emplace_back("CONFIG_VAR=initial");
    m_original_envs.insert("CONFIG_VAR=initial");

    update_env(m_env_vars, "CONFIG_VAR", "weak_update", update_mode::WEAK, ":",
               m_updated_envs, m_original_envs);

    EXPECT_EQ(m_env_vars[0], "CONFIG_VAR=weak_update");

    m_env_vars[0] = "CONFIG_VAR=user_modified";

    update_env(m_env_vars, "CONFIG_VAR", "another_weak_update", update_mode::WEAK, ":",
               m_updated_envs, m_original_envs);

    EXPECT_EQ(m_env_vars[0], "CONFIG_VAR=user_modified");
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
    EXPECT_EQ(m_env_vars[0], "BUILD_PATH=/path1:/path2:/path3:/path4");
}
