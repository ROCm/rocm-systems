// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/common_utils.hpp"

#include <gtest/gtest.h>

#include <sstream>
#include <unordered_set>
#include <vector>

using rocprofsys::common_utils::print_command;
using rocprofsys::common_utils::print_environment;

class PrintOutputTest : public ::testing::Test
{
protected:
    void SetUp() override { m_old_cerr_buf = std::cerr.rdbuf(m_cerr_stream.rdbuf()); }

    void TearDown() override { std::cerr.rdbuf(m_old_cerr_buf); }

    std::string get_cerr() { return m_cerr_stream.str(); }

    static bool does_not_contain(const std::string& output, const std::string& substr)
    {
        return output.empty() || output.find(substr) == std::string::npos;
    }

private:
    std::stringstream m_cerr_stream;
    std::streambuf*   m_old_cerr_buf = nullptr;
};

TEST_F(PrintOutputTest, PrintCommand_HasOutput)
{
    std::vector<std::string> command_args = { "./test", "arg1" };
    print_command(command_args);
    auto output = get_cerr();
    EXPECT_NE(output.find("Executing"), std::string::npos);
    EXPECT_NE(output.find("./test"), std::string::npos);
    EXPECT_NE(output.find("arg1"), std::string::npos);
}

TEST_F(PrintOutputTest, PrintCommand_WithPrefix)
{
    std::vector<std::string> command_args = { "./myapp" };
    print_command(command_args, "PREFIX: ");
    EXPECT_NE(get_cerr().find("PREFIX: "), std::string::npos);
}

TEST_F(PrintOutputTest, PrintCommand_EmptyArgv)
{
    std::vector<std::string> command_args = {};
    print_command(command_args);
    EXPECT_TRUE(does_not_contain(get_cerr(), "Executing"));
}

TEST_F(PrintOutputTest, PrintEnvironment_WithUpdates)
{
    std::vector<std::string> env = { "ROCPROFSYS_TEST=value1", "OTHER_VAR=value2" };
    std::unordered_set<std::string_view> updated = { "ROCPROFSYS_TEST" };

    print_environment(env, updated);
    EXPECT_NE(get_cerr().find("ROCPROFSYS_TEST=value1"), std::string::npos);
}

TEST_F(PrintOutputTest, PrintEnvironment_WithPrefix)
{
    std::vector<std::string>             env     = { "ROCPROFSYS_VAR=test" };
    std::unordered_set<std::string_view> updated = { "ROCPROFSYS_VAR" };

    print_environment(env, updated, false, "MYPREFIX: ");
    EXPECT_NE(get_cerr().find("MYPREFIX: "), std::string::npos);
}

TEST_F(PrintOutputTest, PrintEnvironment_SortsOutput)
{
    std::vector<std::string>             env     = { "ROCPROFSYS_Z=3", "ROCPROFSYS_A=1",
                                                     "ROCPROFSYS_M=2" };
    std::unordered_set<std::string_view> updated = { "ROCPROFSYS_Z", "ROCPROFSYS_A",
                                                     "ROCPROFSYS_M" };

    print_environment(env, updated);
    auto output = get_cerr();

    auto pos_a = output.find("ROCPROFSYS_A");
    auto pos_m = output.find("ROCPROFSYS_M");
    auto pos_z = output.find("ROCPROFSYS_Z");

    EXPECT_NE(pos_a, std::string::npos);
    EXPECT_NE(pos_m, std::string::npos);
    EXPECT_NE(pos_z, std::string::npos);
    EXPECT_LT(pos_a, pos_m);
    EXPECT_LT(pos_m, pos_z);
}

TEST_F(PrintOutputTest, PrintEnvironment_EmptyEnv)
{
    std::vector<std::string>             env     = {};
    std::unordered_set<std::string_view> updated = {};

    print_environment(env, updated);
    EXPECT_TRUE(does_not_contain(get_cerr(), "ROCPROFSYS"));
}

TEST_F(PrintOutputTest, PrintEnvironment_GeneralVarsIncluded)
{
    std::vector<std::string> env = { "ROCPROFSYS_UPDATED=1", "ROCPROFSYS_GENERAL=2" };
    std::unordered_set<std::string_view> updated = { "ROCPROFSYS_UPDATED" };

    print_environment(env, updated, true);
    auto output = get_cerr();
    EXPECT_NE(output.find("ROCPROFSYS_UPDATED"), std::string::npos);
    EXPECT_NE(output.find("ROCPROFSYS_GENERAL"), std::string::npos);
}

TEST_F(PrintOutputTest, PrintEnvironment_GeneralVarsExcluded)
{
    std::vector<std::string> env = { "ROCPROFSYS_UPDATED=1", "ROCPROFSYS_GENERAL=2" };
    std::unordered_set<std::string_view> updated = { "ROCPROFSYS_UPDATED" };

    print_environment(env, updated, false);
    auto output = get_cerr();
    EXPECT_NE(output.find("ROCPROFSYS_UPDATED"), std::string::npos);
    EXPECT_TRUE(does_not_contain(output, "ROCPROFSYS_GENERAL"));
}

TEST_F(PrintOutputTest, PrintEnvironment_NonRocprofsysVarsNotShown)
{
    std::vector<std::string>             env     = { "OTHER_VAR=value", "PATH=/usr/bin" };
    std::unordered_set<std::string_view> updated = {};

    print_environment(env, updated);
    EXPECT_TRUE(does_not_contain(get_cerr(), "OTHER_VAR"));
}
