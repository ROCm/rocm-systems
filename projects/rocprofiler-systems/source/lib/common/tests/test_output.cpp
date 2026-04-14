// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/output.hpp"

#include <gtest/gtest.h>

#include <cstring>
#include <sstream>
#include <unordered_set>
#include <vector>

using namespace rocprofsys::common::output;

class OutputTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_old_cout_buf = std::cout.rdbuf(m_cout_stream.rdbuf());
        m_old_cerr_buf = std::cerr.rdbuf(m_cerr_stream.rdbuf());
    }

    void TearDown() override
    {
        std::cout.rdbuf(m_old_cout_buf);
        std::cerr.rdbuf(m_old_cerr_buf);

        for(auto* ptr : m_allocated)
        {
            free(ptr);
        }
        m_allocated.clear();
    }

    char* make_env(const char* str)
    {
        char* dup = strdup(str);
        m_allocated.push_back(dup);
        return dup;
    }

    std::string get_cout() { return m_cout_stream.str(); }
    std::string get_cerr() { return m_cerr_stream.str(); }

    static bool does_not_contain(const std::string& _output, const std::string& _substr)
    {
        return _output.empty() || _output.find(_substr) == std::string::npos;
    }

private:
    std::stringstream  m_cout_stream;
    std::stringstream  m_cerr_stream;
    std::streambuf*    m_old_cout_buf = nullptr;
    std::streambuf*    m_old_cerr_buf = nullptr;
    std::vector<char*> m_allocated;
};

TEST_F(OutputTest, PrintCommand_HasOutput)
{
    std::vector<char*> command_args = { make_env("./test"), make_env("arg1") };
    print_command(command_args);
    std::string output = get_cout();
    EXPECT_NE(output.find("Executing"), std::string::npos);
    EXPECT_NE(output.find("./test"), std::string::npos);
    EXPECT_NE(output.find("arg1"), std::string::npos);
}

TEST_F(OutputTest, PrintCommand_WithPrefix)
{
    std::vector<char*> command_args = { make_env("./myapp") };
    print_command(command_args, "PREFIX: ");
    std::string output = get_cout();
    EXPECT_NE(output.find("PREFIX: "), std::string::npos);
}

TEST_F(OutputTest, PrintCommand_EmptyArgv)
{
    std::vector<char*> command_args = {};
    print_command(command_args);
    std::string output = get_cout();
    EXPECT_TRUE(does_not_contain(output, "Executing"));
}

TEST_F(OutputTest, PrintEnvironment_WithUpdates)
{
    std::vector<char*>                   env     = { make_env("ROCPROFSYS_TEST=value1"),
                                                     make_env("OTHER_VAR=value2") };
    std::unordered_set<std::string_view> updated = { "ROCPROFSYS_TEST" };

    print_environment(env, updated);
    std::string output = get_cerr();
    EXPECT_NE(output.find("ROCPROFSYS_TEST=value1"), std::string::npos);
}

TEST_F(OutputTest, PrintEnvironment_WithPrefix)
{
    std::vector<char*>                   env     = { make_env("ROCPROFSYS_VAR=test") };
    std::unordered_set<std::string_view> updated = { "ROCPROFSYS_VAR" };

    print_environment(env, updated, false, "MYPREFIX: ");
    std::string output = get_cerr();
    EXPECT_NE(output.find("MYPREFIX: "), std::string::npos);
}

TEST_F(OutputTest, PrintEnvironment_SortsOutput)
{
    std::vector<char*> env = { make_env("ROCPROFSYS_Z=3"), make_env("ROCPROFSYS_A=1"),
                               make_env("ROCPROFSYS_M=2") };
    std::unordered_set<std::string_view> updated = { "ROCPROFSYS_Z", "ROCPROFSYS_A",
                                                     "ROCPROFSYS_M" };

    print_environment(env, updated);
    std::string output = get_cerr();

    auto pos_a = output.find("ROCPROFSYS_A");
    auto pos_m = output.find("ROCPROFSYS_M");
    auto pos_z = output.find("ROCPROFSYS_Z");

    EXPECT_NE(pos_a, std::string::npos);
    EXPECT_NE(pos_m, std::string::npos);
    EXPECT_NE(pos_z, std::string::npos);
    EXPECT_LT(pos_a, pos_m);
    EXPECT_LT(pos_m, pos_z);
}

TEST_F(OutputTest, PrintEnvironment_EmptyEnv)
{
    std::vector<char*>                   env     = {};
    std::unordered_set<std::string_view> updated = {};

    print_environment(env, updated);
    std::string output = get_cerr();
    EXPECT_TRUE(does_not_contain(output, "ROCPROFSYS"));
}

TEST_F(OutputTest, PrintEnvironment_NullEntries)
{
    std::vector<char*>                   env = { make_env("ROCPROFSYS_VAR=test"), nullptr,
                                                 make_env("ROCPROFSYS_OTHER=val") };
    std::unordered_set<std::string_view> updated = { "ROCPROFSYS_VAR",
                                                     "ROCPROFSYS_OTHER" };

    print_environment(env, updated);
    std::string output = get_cerr();
    EXPECT_NE(output.find("ROCPROFSYS_VAR"), std::string::npos);
    EXPECT_NE(output.find("ROCPROFSYS_OTHER"), std::string::npos);
}

TEST_F(OutputTest, PrintEnvironment_GeneralVarsIncluded)
{
    std::vector<char*>                   env     = { make_env("ROCPROFSYS_UPDATED=1"),
                                                     make_env("ROCPROFSYS_GENERAL=2") };
    std::unordered_set<std::string_view> updated = { "ROCPROFSYS_UPDATED" };

    print_environment(env, updated, true);
    std::string output = get_cerr();
    EXPECT_NE(output.find("ROCPROFSYS_UPDATED"), std::string::npos);
    EXPECT_NE(output.find("ROCPROFSYS_GENERAL"), std::string::npos);
}

TEST_F(OutputTest, PrintEnvironment_GeneralVarsExcluded)
{
    std::vector<char*>                   env     = { make_env("ROCPROFSYS_UPDATED=1"),
                                                     make_env("ROCPROFSYS_GENERAL=2") };
    std::unordered_set<std::string_view> updated = { "ROCPROFSYS_UPDATED" };

    print_environment(env, updated, false);
    std::string output = get_cerr();
    EXPECT_NE(output.find("ROCPROFSYS_UPDATED"), std::string::npos);
    EXPECT_TRUE(does_not_contain(output, "ROCPROFSYS_GENERAL"));
}

TEST_F(OutputTest, PrintEnvironment_NonRocprofsysVarsNotShown)
{
    std::vector<char*> env = { make_env("OTHER_VAR=value"), make_env("PATH=/usr/bin") };
    std::unordered_set<std::string_view> updated = {};

    print_environment(env, updated);
    std::string output = get_cerr();
    EXPECT_TRUE(does_not_contain(output, "OTHER_VAR"));
}

TEST_F(OutputTest, MatchesEnvKey_ExactMatch)
{
    EXPECT_TRUE(matches_env_key("ROCPROFSYS_VERBOSE=1", "ROCPROFSYS_VERBOSE"));
}

TEST_F(OutputTest, MatchesEnvKey_RejectsPrefixMatch)
{
    EXPECT_FALSE(matches_env_key("ROCPROFSYS_VERBOSE_LEVEL=1", "ROCPROFSYS_VERBOSE"));
}

TEST_F(OutputTest, MatchesEnvKey_EmptyKey)
{
    EXPECT_FALSE(matches_env_key("ROCPROFSYS_VERBOSE=1", ""));
}

TEST_F(OutputTest, MatchesEnvKey_EmptyEntry)
{
    EXPECT_FALSE(matches_env_key("", "ROCPROFSYS_VERBOSE"));
}

TEST_F(OutputTest, MatchesEnvKey_NoEquals)
{
    EXPECT_FALSE(matches_env_key("ROCPROFSYS_VERBOSE", "ROCPROFSYS_VERBOSE"));
}

TEST_F(OutputTest, StartsWithRocprofsys)
{
    EXPECT_TRUE(starts_with_rocprofsys("ROCPROFSYS_TRACE=ON"));
    EXPECT_FALSE(starts_with_rocprofsys("OTHER_VAR=value"));
    EXPECT_FALSE(starts_with_rocprofsys(""));
}

TEST_F(OutputTest, BuildCommandString_AllNullptr)
{
    std::vector<char*> argv   = { nullptr, nullptr };
    auto               result = build_command_string(argv);
    EXPECT_TRUE(result.empty());
}

TEST_F(OutputTest, PrintCommand_NullptrEntries)
{
    std::vector<char*> command_args = { make_env("./test"), nullptr, make_env("arg1") };
    print_command(command_args);
    std::string output = get_cout();
    EXPECT_NE(output.find("Executing"), std::string::npos);
    EXPECT_NE(output.find("./test"), std::string::npos);
    EXPECT_NE(output.find("arg1"), std::string::npos);
}
