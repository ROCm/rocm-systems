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

TEST_F(DuplicatedEnvironmentEntriesTest, PapiEventsUsesCommaDelimiter)
{
    // PAPI events contain :: in their syntax, so they should use , as delimiter
    std::vector<char*> env_vars = {
        strdup("ROCPROFSYS_PAPI_EVENTS=perf::INSTRUCTIONS"),
        strdup("ROCPROFSYS_PAPI_EVENTS=perf::CACHE_MISSES"),
    };
    consolidate_env_entries(env_vars);
    ASSERT_EQ(env_vars.size(), 1);
    EXPECT_STREQ(env_vars[0],
                 "ROCPROFSYS_PAPI_EVENTS=perf::INSTRUCTIONS,perf::CACHE_MISSES");

    for(auto* entry : env_vars)
        std::free(entry);
}

TEST_F(DuplicatedEnvironmentEntriesTest, PapiEventsPreservesColonInValue)
{
    // Single PAPI event entry should preserve the :: in the value
    std::vector<char*> env_vars = {
        strdup("ROCPROFSYS_PAPI_EVENTS=perf::PERF_COUNT_SW_CPU_CLOCK"),
    };
    consolidate_env_entries(env_vars);
    ASSERT_EQ(env_vars.size(), 1);
    EXPECT_STREQ(env_vars[0], "ROCPROFSYS_PAPI_EVENTS=perf::PERF_COUNT_SW_CPU_CLOCK");

    for(auto* entry : env_vars)
        std::free(entry);
}

TEST_F(DuplicatedEnvironmentEntriesTest, PapiEventsDeduplicates)
{
    // Duplicate PAPI events should be deduplicated
    std::vector<char*> env_vars = {
        strdup("ROCPROFSYS_PAPI_EVENTS=perf::INSTRUCTIONS"),
        strdup("ROCPROFSYS_PAPI_EVENTS=perf::CACHE_MISSES"),
        strdup("ROCPROFSYS_PAPI_EVENTS=perf::INSTRUCTIONS"),
    };
    consolidate_env_entries(env_vars);
    ASSERT_EQ(env_vars.size(), 1);
    EXPECT_STREQ(env_vars[0],
                 "ROCPROFSYS_PAPI_EVENTS=perf::INSTRUCTIONS,perf::CACHE_MISSES");

    for(auto* entry : env_vars)
        std::free(entry);
}

TEST_F(DuplicatedEnvironmentEntriesTest, SamplingOverflowEventUsesCommaDelimiter)
{
    std::vector<char*> env_vars = {
        strdup("ROCPROFSYS_SAMPLING_OVERFLOW_EVENT=perf::INSTRUCTIONS"),
        strdup("ROCPROFSYS_SAMPLING_OVERFLOW_EVENT=perf::CYCLES"),
    };
    consolidate_env_entries(env_vars);
    ASSERT_EQ(env_vars.size(), 1);
    EXPECT_STREQ(env_vars[0],
                 "ROCPROFSYS_SAMPLING_OVERFLOW_EVENT=perf::INSTRUCTIONS,perf::CYCLES");

    for(auto* entry : env_vars)
        std::free(entry);
}

TEST_F(DuplicatedEnvironmentEntriesTest, MixedDelimiterVariables)
{
    // Test that regular variables use : and PAPI uses ,
    std::vector<char*> env_vars = {
        strdup("PATH=/usr/bin"),
        strdup("ROCPROFSYS_PAPI_EVENTS=perf::INSTRUCTIONS"),
        strdup("PATH=/usr/local/bin"),
        strdup("ROCPROFSYS_PAPI_EVENTS=perf::CACHE_MISSES"),
        strdup("LD_LIBRARY_PATH=/lib"),
    };
    consolidate_env_entries(env_vars);
    ASSERT_EQ(env_vars.size(), 3);
    EXPECT_STREQ(env_vars[0], "PATH=/usr/bin:/usr/local/bin");
    EXPECT_STREQ(env_vars[1],
                 "ROCPROFSYS_PAPI_EVENTS=perf::INSTRUCTIONS,perf::CACHE_MISSES");
    EXPECT_STREQ(env_vars[2], "LD_LIBRARY_PATH=/lib");

    for(auto* entry : env_vars)
        std::free(entry);
}

TEST_F(DuplicatedEnvironmentEntriesTest, PreservesKeyOrder)
{
    std::vector<char*> env_vars = {
        strdup("ZEBRA=1"),
        strdup("ALPHA=2"),
        strdup("MIDDLE=3"),
    };
    consolidate_env_entries(env_vars);
    ASSERT_EQ(env_vars.size(), 3);
    // Order should be preserved based on first occurrence
    EXPECT_STREQ(env_vars[0], "ZEBRA=1");
    EXPECT_STREQ(env_vars[1], "ALPHA=2");
    EXPECT_STREQ(env_vars[2], "MIDDLE=3");

    for(auto* entry : env_vars)
        std::free(entry);
}

TEST_F(DuplicatedEnvironmentEntriesTest, PapiEventsWithCommaInValue)
{
    // If PAPI events are already comma-separated in a single entry, they should be split
    std::vector<char*> env_vars = {
        strdup("ROCPROFSYS_PAPI_EVENTS=perf::INSTRUCTIONS,perf::CYCLES"),
        strdup("ROCPROFSYS_PAPI_EVENTS=perf::CACHE_MISSES"),
    };
    consolidate_env_entries(env_vars);
    ASSERT_EQ(env_vars.size(), 1);
    EXPECT_STREQ(
        env_vars[0],
        "ROCPROFSYS_PAPI_EVENTS=perf::INSTRUCTIONS,perf::CYCLES,perf::CACHE_MISSES");

    for(auto* entry : env_vars)
        std::free(entry);
}

TEST_F(DuplicatedEnvironmentEntriesTest, RocmEventsUsesCommaDelimiter)
{
    // ROCM events use :device=N syntax, so they should use , as delimiter
    std::vector<char*> env_vars = {
        strdup("ROCPROFSYS_ROCM_EVENTS=SQ_WAVES:device=0"),
        strdup("ROCPROFSYS_ROCM_EVENTS=TA_TA_BUSY:device=1"),
    };
    consolidate_env_entries(env_vars);
    ASSERT_EQ(env_vars.size(), 1);
    EXPECT_STREQ(env_vars[0],
                 "ROCPROFSYS_ROCM_EVENTS=SQ_WAVES:device=0,TA_TA_BUSY:device=1");

    for(auto* entry : env_vars)
        std::free(entry);
}

TEST_F(DuplicatedEnvironmentEntriesTest, RocmEventsPreservesDeviceSyntax)
{
    // Single ROCM event entry should preserve the :device= syntax
    std::vector<char*> env_vars = {
        strdup("ROCPROFSYS_ROCM_EVENTS=GRBM_COUNT,SQ_WAVES,SQ_INSTS_VALU,TA_TA_BUSY:"
               "device=0"),
    };
    consolidate_env_entries(env_vars);
    ASSERT_EQ(env_vars.size(), 1);
    EXPECT_STREQ(
        env_vars[0],
        "ROCPROFSYS_ROCM_EVENTS=GRBM_COUNT,SQ_WAVES,SQ_INSTS_VALU,TA_TA_BUSY:device=0");

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
