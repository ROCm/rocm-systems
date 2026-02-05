// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "common/environment.hpp"

#include <gtest/gtest.h>

using namespace rocprofsys::common;

class IsPythonInterpreterTest : public ::testing::Test
{};

TEST_F(IsPythonInterpreterTest, RecognizesPython)
{
    EXPECT_TRUE(is_python_interpreter("python"));
    EXPECT_TRUE(is_python_interpreter("python3"));
    EXPECT_TRUE(is_python_interpreter("python3.8"));
    EXPECT_TRUE(is_python_interpreter("python3.9"));
    EXPECT_TRUE(is_python_interpreter("python3.10"));
    EXPECT_TRUE(is_python_interpreter("python3.11"));
    EXPECT_TRUE(is_python_interpreter("python3.12"));
    EXPECT_TRUE(is_python_interpreter("/usr/bin/python"));
    EXPECT_TRUE(is_python_interpreter("/usr/bin/python3"));
    EXPECT_TRUE(is_python_interpreter("/usr/bin/python3.10"));
    EXPECT_TRUE(is_python_interpreter("/home/user/venv/bin/python"));
    EXPECT_TRUE(is_python_interpreter("/opt/conda/bin/python3.11"));
    EXPECT_FALSE(is_python_interpreter("bash"));
    EXPECT_FALSE(is_python_interpreter("sh"));
    EXPECT_FALSE(is_python_interpreter("ruby"));
    EXPECT_FALSE(is_python_interpreter("node"));
    EXPECT_FALSE(is_python_interpreter("java"));
    EXPECT_FALSE(is_python_interpreter("/usr/bin/bash"));
    EXPECT_FALSE(is_python_interpreter("./my_app"));
    EXPECT_FALSE(is_python_interpreter("pythonista"));
    EXPECT_FALSE(is_python_interpreter("python_script.py"));
    EXPECT_FALSE(is_python_interpreter("mypython"));
    EXPECT_FALSE(is_python_interpreter("python2"));
    EXPECT_FALSE(is_python_interpreter("python3."));
    EXPECT_FALSE(is_python_interpreter("python3.a"));
    EXPECT_FALSE(is_python_interpreter("python3.10a"));
    EXPECT_FALSE(is_python_interpreter("python3x10"));
    EXPECT_FALSE(is_python_interpreter(""));
    EXPECT_FALSE(is_python_interpreter("/usr/bin/"));
}

class DuplicatedEnvironmentEntriesTest : public ::testing::Test
{};

TEST_F(DuplicatedEnvironmentEntriesTest, DuplicateEnvironmentEntries)
{
    std::vector<char*> env_vars = {
        strdup("PATH=/usr/local/bin:/usr/bin:/bin:/usr/local/bin2"),
        strdup("PATH=/usr/local/bin:/usr/bin:/bin"),
    };

    consolidate_env_entries(env_vars);

    ASSERT_EQ(env_vars.size(), 1);
    EXPECT_STREQ(env_vars[0], "PATH=/usr/local/bin:/usr/bin:/bin:/usr/local/bin2");

    for(auto* entry : env_vars)
        free(entry);
}

TEST_F(DuplicatedEnvironmentEntriesTest, HandlesEmptyVector)
{
    std::vector<char*> env_vars;
    consolidate_env_entries(env_vars);
    EXPECT_TRUE(env_vars.empty());
}

TEST_F(DuplicatedEnvironmentEntriesTest, HandlesNullEntries)
{
    std::vector<char*> env_vars = {
        strdup("PATH=/usr/bin"),
        nullptr,
        strdup("PATH=/bin"),
    };
    consolidate_env_entries(env_vars);
    ASSERT_EQ(env_vars.size(), 1);
    EXPECT_STREQ(env_vars[0], "PATH=/usr/bin:/bin");
    for(auto* entry : env_vars)
        std::free(entry);
}

TEST_F(DuplicatedEnvironmentEntriesTest, HandlesEmptyValues)
{
    std::vector<char*> env_vars = {
        strdup("EMPTY_VAR="),
        strdup("PATH=/usr/bin"),
    };
    consolidate_env_entries(env_vars);
    ASSERT_EQ(env_vars.size(), 2);

    for(auto* entry : env_vars)
        std::free(entry);
}

class AddTorchLibraryPathTest : public ::testing::Test
{
protected:
    std::unordered_set<std::string> updated_envs;
};

TEST_F(AddTorchLibraryPathTest, SkipsNonPythonExecutables)
{
    std::vector<char*> envp = {
        strdup("LD_LIBRARY_PATH=/usr/lib"),
    };
    std::vector<char*> argv = {
        strdup("/usr/bin/bash"),
    };
    add_torch_library_path(envp, argv, false, updated_envs);
    // Should not modify environment
    ASSERT_EQ(envp.size(), 1);
    EXPECT_STREQ(envp[0], "LD_LIBRARY_PATH=/usr/lib");
    for(auto* entry : envp)
        std::free(entry);
    for(auto* entry : argv)
        std::free(entry);
}

TEST_F(AddTorchLibraryPathTest, HandlesEmptyArgv)
{
    std::vector<char*> envp = {
        strdup("LD_LIBRARY_PATH=/usr/lib"),
    };
    std::vector<char*> argv;
    add_torch_library_path(envp, argv, false, updated_envs);
    ASSERT_EQ(envp.size(), 1);
    EXPECT_STREQ(envp[0], "LD_LIBRARY_PATH=/usr/lib");
    for(auto* entry : envp)
        std::free(entry);
}

TEST_F(AddTorchLibraryPathTest, HandlesNullArgvFront)
{
    std::vector<char*> envp = {
        strdup("LD_LIBRARY_PATH=/usr/lib"),
    };
    std::vector<char*> argv = { nullptr };
    add_torch_library_path(envp, argv, false, updated_envs);
    ASSERT_EQ(envp.size(), 1);
    for(auto* entry : envp)
        std::free(entry);
}
