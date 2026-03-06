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

    void clear_streams()
    {
        m_cout_stream.str("");
        m_cout_stream.clear();
        m_cerr_stream.str("");
        m_cerr_stream.clear();
    }

private:
    std::stringstream  m_cout_stream;
    std::stringstream  m_cerr_stream;
    std::streambuf*    m_old_cout_buf = nullptr;
    std::streambuf*    m_old_cerr_buf = nullptr;
    std::vector<char*> m_allocated;
};

TEST_F(OutputTest, PrintCommand_VerboseZero_NoOutput)
{
    std::vector<char*> command_args = { make_env("./test"), make_env("arg1") };
    print_command(command_args, 0);
    std::string output = get_cout();
    EXPECT_TRUE(does_not_contain(output, "Executing"));
}

TEST_F(OutputTest, PrintCommand_VerboseOne_HasOutput)
{
    std::vector<char*> command_args = { make_env("./test"), make_env("arg1") };
    print_command(command_args, 1);
    std::string output = get_cout();
    EXPECT_NE(output.find("Executing"), std::string::npos);
    EXPECT_NE(output.find("./test"), std::string::npos);
    EXPECT_NE(output.find("arg1"), std::string::npos);
}

TEST_F(OutputTest, PrintCommand_WithPrefix)
{
    std::vector<char*> command_args = { make_env("./myapp") };
    print_command(command_args, 1, "PREFIX: ");
    std::string output = get_cout();
    EXPECT_NE(output.find("PREFIX: "), std::string::npos);
}

TEST_F(OutputTest, PrintCommand_EmptyArgv)
{
    std::vector<char*> command_args = {};
    print_command(command_args, 1);
    std::string output = get_cout();
    EXPECT_NE(output.find("Executing"), std::string::npos);
}

TEST_F(OutputTest, PrintUpdatedEnvironment_NegativeVerbose_NoOutput)
{
    std::vector<char*>                   env     = { make_env("ROCPROFSYS_TEST=1") };
    std::unordered_set<std::string_view> updated = { "ROCPROFSYS_TEST" };

    print_updated_environment(env, updated, -1);
    EXPECT_TRUE(get_cerr().empty());
}

TEST_F(OutputTest, PrintUpdatedEnvironment_WithUpdates)
{
    std::vector<char*>                   env     = { make_env("ROCPROFSYS_TEST=value1"),
                                                     make_env("OTHER_VAR=value2") };
    std::unordered_set<std::string_view> updated = { "ROCPROFSYS_TEST" };

    print_updated_environment(env, updated, 0);
    std::string output = get_cerr();
    EXPECT_NE(output.find("ROCPROFSYS_TEST=value1"), std::string::npos);
}

TEST_F(OutputTest, PrintUpdatedEnvironment_WithPrefix)
{
    std::vector<char*>                   env     = { make_env("ROCPROFSYS_VAR=test") };
    std::unordered_set<std::string_view> updated = { "ROCPROFSYS_VAR" };

    print_updated_environment(env, updated, 0, "MYPREFIX: ");
    std::string output = get_cerr();
    EXPECT_NE(output.find("MYPREFIX: "), std::string::npos);
}

TEST_F(OutputTest, PrintUpdatedEnvironment_SortsOutput)
{
    std::vector<char*> env = { make_env("ROCPROFSYS_Z=3"), make_env("ROCPROFSYS_A=1"),
                               make_env("ROCPROFSYS_M=2") };
    std::unordered_set<std::string_view> updated = { "ROCPROFSYS_Z", "ROCPROFSYS_A",
                                                     "ROCPROFSYS_M" };

    print_updated_environment(env, updated, 0);
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

TEST_F(OutputTest, PrintUpdatedEnvironment_EmptyEnv)
{
    std::vector<char*>                   env     = {};
    std::unordered_set<std::string_view> updated = {};

    print_updated_environment(env, updated, 0);
    std::string output = get_cerr();
    EXPECT_TRUE(does_not_contain(output, "ROCPROFSYS"));
}

TEST_F(OutputTest, PrintUpdatedEnvironment_NullEntries)
{
    std::vector<char*>                   env = { make_env("ROCPROFSYS_VAR=test"), nullptr,
                                                 make_env("ROCPROFSYS_OTHER=val") };
    std::unordered_set<std::string_view> updated = { "ROCPROFSYS_VAR",
                                                     "ROCPROFSYS_OTHER" };

    print_updated_environment(env, updated, 0);
    std::string output = get_cerr();
    EXPECT_NE(output.find("ROCPROFSYS_VAR"), std::string::npos);
    EXPECT_NE(output.find("ROCPROFSYS_OTHER"), std::string::npos);
}

TEST_F(OutputTest, PrintUpdatedEnvironment_GeneralVarsAtHighVerbosity)
{
    std::vector<char*>                   env     = { make_env("ROCPROFSYS_UPDATED=1"),
                                                     make_env("ROCPROFSYS_GENERAL=2") };
    std::unordered_set<std::string_view> updated = { "ROCPROFSYS_UPDATED" };

    print_updated_environment(env, updated, 1);
    std::string output = get_cerr();
    EXPECT_NE(output.find("ROCPROFSYS_UPDATED"), std::string::npos);
    EXPECT_NE(output.find("ROCPROFSYS_GENERAL"), std::string::npos);
}

TEST_F(OutputTest, PrintUpdatedEnvironment_NonRocprofsysVarsNotShown)
{
    std::vector<char*> env = { make_env("OTHER_VAR=value"), make_env("PATH=/usr/bin") };
    std::unordered_set<std::string_view> updated = {};

    print_updated_environment(env, updated, 0);
    std::string output = get_cerr();
    EXPECT_TRUE(does_not_contain(output, "OTHER_VAR"));
}

TEST_F(OutputTest, PrintUpdatedEnvironment_StringSet)
{
    std::vector<char*>              env     = { make_env("ROCPROFSYS_TEST=value") };
    std::unordered_set<std::string> updated = { "ROCPROFSYS_TEST" };

    print_updated_environment(env, updated, 0);
    std::string output = get_cerr();
    EXPECT_NE(output.find("ROCPROFSYS_TEST"), std::string::npos);
}
