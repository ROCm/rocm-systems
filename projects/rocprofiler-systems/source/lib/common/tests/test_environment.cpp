// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

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
    std::vector<std::string> env_vars = {
        "PATH=/usr/local/bin:/usr/bin:/bin:/usr/local/bin2",
        "PATH=/usr/local/bin:/usr/bin:/bin",
    };

    consolidate_env_entries(env_vars);

    ASSERT_EQ(env_vars.size(), 1);
    EXPECT_EQ(env_vars[0], "PATH=/usr/local/bin:/usr/bin:/bin:/usr/local/bin2");
}

TEST_F(DuplicatedEnvironmentEntriesTest, HandlesEmptyVector)
{
    std::vector<std::string> env_vars;
    consolidate_env_entries(env_vars);
    EXPECT_TRUE(env_vars.empty());
}

TEST_F(DuplicatedEnvironmentEntriesTest, HandlesEmptyValues)
{
    std::vector<std::string> env_vars = {
        "EMPTY_VAR=",
        "PATH=/usr/bin",
    };
    consolidate_env_entries(env_vars);
    ASSERT_EQ(env_vars.size(), 2);
}

TEST_F(DuplicatedEnvironmentEntriesTest, PapiEventsUsesCommaDelimiter)
{
    std::vector<std::string> env_vars = {
        "ROCPROFSYS_PAPI_EVENTS=perf::INSTRUCTIONS",
        "ROCPROFSYS_PAPI_EVENTS=perf::CACHE_MISSES",
    };
    consolidate_env_entries(env_vars);
    ASSERT_EQ(env_vars.size(), 1);
    EXPECT_EQ(env_vars[0],
              "ROCPROFSYS_PAPI_EVENTS=perf::INSTRUCTIONS,perf::CACHE_MISSES");
}

TEST_F(DuplicatedEnvironmentEntriesTest, PapiEventsPreservesColonInValue)
{
    std::vector<std::string> env_vars = {
        "ROCPROFSYS_PAPI_EVENTS=perf::PERF_COUNT_SW_CPU_CLOCK",
    };
    consolidate_env_entries(env_vars);
    ASSERT_EQ(env_vars.size(), 1);
    EXPECT_EQ(env_vars[0], "ROCPROFSYS_PAPI_EVENTS=perf::PERF_COUNT_SW_CPU_CLOCK");
}

TEST_F(DuplicatedEnvironmentEntriesTest, PapiEventsDeduplicates)
{
    std::vector<std::string> env_vars = {
        "ROCPROFSYS_PAPI_EVENTS=perf::INSTRUCTIONS",
        "ROCPROFSYS_PAPI_EVENTS=perf::CACHE_MISSES",
        "ROCPROFSYS_PAPI_EVENTS=perf::INSTRUCTIONS",
    };
    consolidate_env_entries(env_vars);
    ASSERT_EQ(env_vars.size(), 1);
    EXPECT_EQ(env_vars[0],
              "ROCPROFSYS_PAPI_EVENTS=perf::INSTRUCTIONS,perf::CACHE_MISSES");
}

TEST_F(DuplicatedEnvironmentEntriesTest, SamplingOverflowEventUsesCommaDelimiter)
{
    std::vector<std::string> env_vars = {
        "ROCPROFSYS_SAMPLING_OVERFLOW_EVENT=perf::INSTRUCTIONS",
        "ROCPROFSYS_SAMPLING_OVERFLOW_EVENT=perf::CYCLES",
    };
    consolidate_env_entries(env_vars);
    ASSERT_EQ(env_vars.size(), 1);
    EXPECT_EQ(env_vars[0],
              "ROCPROFSYS_SAMPLING_OVERFLOW_EVENT=perf::INSTRUCTIONS,perf::CYCLES");
}

TEST_F(DuplicatedEnvironmentEntriesTest, MixedDelimiterVariables)
{
    std::vector<std::string> env_vars = {
        "PATH=/usr/bin",        "ROCPROFSYS_PAPI_EVENTS=perf::INSTRUCTIONS",
        "PATH=/usr/local/bin",  "ROCPROFSYS_PAPI_EVENTS=perf::CACHE_MISSES",
        "LD_LIBRARY_PATH=/lib",
    };
    consolidate_env_entries(env_vars);
    ASSERT_EQ(env_vars.size(), 3);
    EXPECT_EQ(env_vars[0], "PATH=/usr/bin:/usr/local/bin");
    EXPECT_EQ(env_vars[1],
              "ROCPROFSYS_PAPI_EVENTS=perf::INSTRUCTIONS,perf::CACHE_MISSES");
    EXPECT_EQ(env_vars[2], "LD_LIBRARY_PATH=/lib");
}

TEST_F(DuplicatedEnvironmentEntriesTest, PreservesKeyOrder)
{
    std::vector<std::string> env_vars = {
        "ZEBRA=1",
        "ALPHA=2",
        "MIDDLE=3",
    };
    consolidate_env_entries(env_vars);
    ASSERT_EQ(env_vars.size(), 3);
    EXPECT_EQ(env_vars[0], "ZEBRA=1");
    EXPECT_EQ(env_vars[1], "ALPHA=2");
    EXPECT_EQ(env_vars[2], "MIDDLE=3");
}

TEST_F(DuplicatedEnvironmentEntriesTest, PapiEventsWithCommaInValue)
{
    std::vector<std::string> env_vars = {
        "ROCPROFSYS_PAPI_EVENTS=perf::INSTRUCTIONS,perf::CYCLES",
        "ROCPROFSYS_PAPI_EVENTS=perf::CACHE_MISSES",
    };
    consolidate_env_entries(env_vars);
    ASSERT_EQ(env_vars.size(), 1);
    EXPECT_EQ(
        env_vars[0],
        "ROCPROFSYS_PAPI_EVENTS=perf::INSTRUCTIONS,perf::CYCLES,perf::CACHE_MISSES");
}

TEST_F(DuplicatedEnvironmentEntriesTest, RocmEventsUsesCommaDelimiter)
{
    std::vector<std::string> env_vars = {
        "ROCPROFSYS_ROCM_EVENTS=SQ_WAVES:device=0",
        "ROCPROFSYS_ROCM_EVENTS=TA_TA_BUSY:device=1",
    };
    consolidate_env_entries(env_vars);
    ASSERT_EQ(env_vars.size(), 1);
    EXPECT_EQ(env_vars[0],
              "ROCPROFSYS_ROCM_EVENTS=SQ_WAVES:device=0,TA_TA_BUSY:device=1");
}

TEST_F(DuplicatedEnvironmentEntriesTest, RocmEventsPreservesDeviceSyntax)
{
    std::vector<std::string> env_vars = {
        "ROCPROFSYS_ROCM_EVENTS=GRBM_COUNT,SQ_WAVES,SQ_INSTS_VALU,TA_TA_BUSY:device=0",
    };
    consolidate_env_entries(env_vars);
    ASSERT_EQ(env_vars.size(), 1);
    EXPECT_EQ(
        env_vars[0],
        "ROCPROFSYS_ROCM_EVENTS=GRBM_COUNT,SQ_WAVES,SQ_INSTS_VALU,TA_TA_BUSY:device=0");
}

class AddTorchLibraryPathTest : public ::testing::Test
{
protected:
    std::unordered_set<std::string> updated_envs;
};

TEST_F(AddTorchLibraryPathTest, SkipsNonPythonExecutables)
{
    std::vector<std::string> envp = { "LD_LIBRARY_PATH=/usr/lib" };
    add_torch_library_path(envp, "/usr/bin/bash", false, updated_envs);
    ASSERT_EQ(envp.size(), 1);
    EXPECT_EQ(envp[0], "LD_LIBRARY_PATH=/usr/lib");
}

TEST_F(AddTorchLibraryPathTest, HandlesEmptyExecutable)
{
    std::vector<std::string> envp = { "LD_LIBRARY_PATH=/usr/lib" };
    add_torch_library_path(envp, std::string_view{}, false, updated_envs);
    ASSERT_EQ(envp.size(), 1);
    EXPECT_EQ(envp[0], "LD_LIBRARY_PATH=/usr/lib");
}
