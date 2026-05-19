// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/environment.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <unordered_map>

using namespace rocprofsys::common;

// ── fake_env ─────────────────────────────────────────────────────────────────
// In-memory environment backend for unit tests.
// Call fake_env::reset() in SetUp()/TearDown() to isolate tests.
struct fake_env
{
    inline static std::unordered_map<std::string, std::string> store;

    static int setenv(const char* name, const char* value, int overwrite)
    {
        if(!overwrite && store.count(name)) return 0;
        store[name] = value;
        return 0;
    }

    static char* getenv(const char* name)
    {
        auto it = store.find(name);
        return it != store.end() ? it->second.data() : nullptr;
    }

    static void reset() { store.clear(); }
};

// Convenience alias used throughout the injection tests.
using fake_environment = environment<fake_env>;

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

// ── Dependency-injection tests via fake_env ───────────────────────────────────
// These tests never touch the real process environment.

class FakeEnvGetEnvTest : public ::testing::Test
{
protected:
    void SetUp() override { fake_env::reset(); }
    void TearDown() override { fake_env::reset(); }
};

TEST_F(FakeEnvGetEnvTest, StringReturnsDefaultWhenUnset)
{
    EXPECT_EQ(fake_environment::get_env("FOO", std::string{ "default" }), "default");
}

TEST_F(FakeEnvGetEnvTest, StringReturnsValueWhenSet)
{
    fake_env::setenv("FOO", "bar", 1);
    EXPECT_EQ(fake_environment::get_env("FOO", std::string{ "default" }), "bar");
}

TEST_F(FakeEnvGetEnvTest, IntReturnsDefaultWhenUnset)
{
    EXPECT_EQ(fake_environment::get_env("FOO", 42), 42);
}

TEST_F(FakeEnvGetEnvTest, IntReturnsValueWhenSet)
{
    fake_env::setenv("FOO", "7", 1);
    EXPECT_EQ(fake_environment::get_env("FOO", 42), 7);
}

TEST_F(FakeEnvGetEnvTest, BoolTrueVariants)
{
    for(const char* v : { "1", "true", "yes", "on" })
    {
        fake_env::reset();
        fake_env::setenv("FOO", v, 1);
        EXPECT_TRUE(fake_environment::get_env("FOO", false)) << "value: " << v;
    }
}

TEST_F(FakeEnvGetEnvTest, BoolFalseVariants)
{
    for(const char* v : { "0", "false", "no", "off" })
    {
        fake_env::reset();
        fake_env::setenv("FOO", v, 1);
        EXPECT_FALSE(fake_environment::get_env("FOO", true)) << "value: " << v;
    }
}

TEST_F(FakeEnvGetEnvTest, DoubleReturnsValueWhenSet)
{
    fake_env::setenv("FOO", "3.14", 1);
    EXPECT_NEAR(fake_environment::get_env("FOO", 0.0), 3.14, 1e-9);
}

TEST_F(FakeEnvGetEnvTest, OneArgFormReturnsEmptyWhenUnset)
{
    EXPECT_EQ(fake_environment::get_env<std::string>("FOO"), "");
}

TEST_F(FakeEnvGetEnvTest, EmptyVarNameReturnsDefault)
{
    EXPECT_EQ(fake_environment::get_env("", std::string{ "fallback" }), "fallback");
}

TEST_F(FakeEnvGetEnvTest, DoesNotLeakToRealEnvironment)
{
    fake_env::setenv("ROCPROFSYS_FAKE_TEST_ISOLATION", "injected", 1);
    // The real process environment must not have been modified.
    EXPECT_EQ(::getenv("ROCPROFSYS_FAKE_TEST_ISOLATION"), nullptr);
}

class FakeEnvSetEnvTest : public ::testing::Test
{
protected:
    void SetUp() override { fake_env::reset(); }
    void TearDown() override { fake_env::reset(); }
};

TEST_F(FakeEnvSetEnvTest, SetsStringValue)
{
    fake_environment::set_env("FOO", std::string{ "hello" }, 1);
    EXPECT_EQ(fake_environment::get_env("FOO", std::string{}), "hello");
}

TEST_F(FakeEnvSetEnvTest, SetsIntValue)
{
    fake_environment::set_env("FOO", 99, 1);
    EXPECT_EQ(fake_environment::get_env("FOO", 0), 99);
}

TEST_F(FakeEnvSetEnvTest, OverrideZeroDoesNotOverwrite)
{
    fake_env::setenv("FOO", "original", 1);
    fake_environment::set_env("FOO", std::string{ "new" }, 0);
    EXPECT_EQ(fake_environment::get_env("FOO", std::string{}), "original");
}

TEST_F(FakeEnvSetEnvTest, OverrideOneOverwrites)
{
    fake_env::setenv("FOO", "original", 1);
    fake_environment::set_env("FOO", std::string{ "new" }, 1);
    EXPECT_EQ(fake_environment::get_env("FOO", std::string{}), "new");
}

class FakeEnvGetEnvChoiceTest : public ::testing::Test
{
protected:
    void SetUp() override { fake_env::reset(); }
    void TearDown() override { fake_env::reset(); }
};

TEST_F(FakeEnvGetEnvChoiceTest, ReturnsDefaultWhenUnset)
{
    auto result = fake_environment::get_env_choice<std::string>(
        "FOO", "trace", { "trace", "sampling", "causal" });
    EXPECT_EQ(result, "trace");
}

TEST_F(FakeEnvGetEnvChoiceTest, ReturnsValueWhenValidChoiceSet)
{
    fake_env::setenv("FOO", "sampling", 1);
    auto result = fake_environment::get_env_choice<std::string>(
        "FOO", "trace", { "trace", "sampling", "causal" });
    EXPECT_EQ(result, "sampling");
}

TEST_F(FakeEnvGetEnvChoiceTest, ReturnsDefaultWhenInvalidChoiceSet)
{
    fake_env::setenv("FOO", "bad_value", 1);
    auto result = fake_environment::get_env_choice<std::string>(
        "FOO", "trace", { "trace", "sampling", "causal" });
    EXPECT_EQ(result, "trace");
}

class FakeEnvConfigTest : public ::testing::Test
{
protected:
    void SetUp() override { fake_env::reset(); }
    void TearDown() override { fake_env::reset(); }
};

TEST_F(FakeEnvConfigTest, OperatorSetsValue)
{
    env_config<fake_env> cfg;
    cfg.env_name  = "FOO";
    cfg.env_value = "injected";
    cfg.override  = 1;
    cfg();
    EXPECT_EQ(fake_environment::get_env("FOO", std::string{}), "injected");
}

TEST_F(FakeEnvConfigTest, OperatorRespectsOverrideZero)
{
    fake_env::setenv("FOO", "original", 1);
    env_config<fake_env> cfg;
    cfg.env_name  = "FOO";
    cfg.env_value = "new";
    cfg.override  = 0;
    cfg();
    EXPECT_EQ(fake_environment::get_env("FOO", std::string{}), "original");
}

TEST_F(FakeEnvConfigTest, EmptyNameIsNoop)
{
    env_config<fake_env> cfg;
    cfg.env_name  = "";
    cfg.env_value = "ignored";
    cfg.override  = 1;
    EXPECT_EQ(cfg(), -1);
    EXPECT_TRUE(fake_env::store.empty());
}
