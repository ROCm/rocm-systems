// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/environment.hpp"
#include <cstdint>

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
    add_torch_library_path(envp, "/usr/bin/bash", updated_envs);
    ASSERT_EQ(envp.size(), 1);
    EXPECT_EQ(envp[0], "LD_LIBRARY_PATH=/usr/lib");
}

TEST_F(AddTorchLibraryPathTest, HandlesEmptyExecutable)
{
    std::vector<std::string> envp = { "LD_LIBRARY_PATH=/usr/lib" };
    add_torch_library_path(envp, std::string_view{}, updated_envs);
    ASSERT_EQ(envp.size(), 1);
    EXPECT_EQ(envp[0], "LD_LIBRARY_PATH=/usr/lib");
}

TEST_F(AddTorchLibraryPathTest, PythonExecutableWithoutTorchLeavesEnvUnchanged)
{
    // "/nonexistent/python3" is recognised as a Python interpreter but
    // discover_torch_libpath returns "" → early return, env unchanged.
    std::vector<std::string> envp = { "LD_LIBRARY_PATH=/usr/lib" };
    add_torch_library_path(envp, "/nonexistent/python3", updated_envs);
    ASSERT_EQ(envp.size(), 1);
    EXPECT_EQ(envp[0], "LD_LIBRARY_PATH=/usr/lib");
}

// ── discover_torch_libpath direct tests (L384–L442) ──────────────────────────

TEST(DiscoverTorchLibpathTest, EmptyBinaryReturnsEmpty)
{
    EXPECT_EQ(discover_torch_libpath(""), "");
}

TEST(DiscoverTorchLibpathTest, UnsafeCharInPathReturnsEmpty)
{
    // ';' hits default: return false in is_safe_executable_path
    EXPECT_EQ(discover_torch_libpath("/usr/bin/python;injected"), "");
}

TEST(DiscoverTorchLibpathTest, SpaceInPathReturnsEmpty)
{
    EXPECT_EQ(discover_torch_libpath("/path with space/python3"), "");
}

TEST(DiscoverTorchLibpathTest, DollarSignInPathReturnsEmpty)
{
    EXPECT_EQ(discover_torch_libpath("$HOME/python3"), "");
}

TEST(DiscoverTorchLibpathTest, SafePathWithAllowedCharsAndNoTorchReturnsEmpty)
{
    // Path uses all allowed non-alnum chars (/ _ - + .) to exercise every
    // switch case in is_safe_executable_path. Shell exits non-zero (not found)
    // → popen succeeds, status != 0 → returns empty string.
    EXPECT_EQ(discover_torch_libpath("/nonexistent_dir/python-3.11+safe.bin"), "");
}

// ── posix_env + forwarding free functions (L77, L79, L269–L279) ──────────────
// Calling get_env / set_env without the fake_environment:: prefix exercises
// posix_env::getenv / setenv through the real environment<posix_env>.

TEST(FreeFunctionEnvTest, GetEnvReturnsDefaultForUnsetVar)
{
    // Unset variable → default returned; covers posix_env::getenv + free-func body
    EXPECT_EQ(get_env("ROCPROFSYS_FREE_FUNC_TEST_UNSET_99887766", std::string{ "dflt" }),
              "dflt");
}

TEST(FreeFunctionEnvTest, GetEnvOneArgReturnsEmptyForUnsetVar)
{
    EXPECT_EQ(get_env<std::string>("ROCPROFSYS_FREE_FUNC_TEST_UNSET_99887766"), "");
}

TEST(FreeFunctionEnvTest, SetEnvAndGetViaFreeFunction)
{
    const char* name = "ROCPROFSYS_FREE_FUNC_SETENV_TEST_99887766";
    set_env(std::string{ name }, std::string{ "free_val" }, 1);
    EXPECT_EQ(get_env(name, std::string{}), "free_val");
    ::unsetenv(name);
}

// ── Enum used by the dispatch tests below ────────────────────────────────────
enum class test_color : int
{
    red   = 0,
    green = 1,
    blue  = 2,
};

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

TEST_F(FakeEnvGetEnvTest, StringViewEnvIdResolvesViaShim)
{
    // A non-null-terminated string_view name must still resolve correctly.
    fake_env::setenv("FOO", "bar", 1);
    const std::string_view name = std::string_view{ "FOOBAR" }.substr(0, 3);
    EXPECT_EQ(fake_environment::get_env(name, std::string{ "default" }), "bar");
}

TEST_F(FakeEnvGetEnvTest, StdStringEnvIdResolvesViaShim)
{
    fake_env::setenv("FOO", "7", 1);
    const std::string name{ "FOO" };
    EXPECT_EQ(fake_environment::get_env(name, 42), 7);
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

TEST_F(FakeEnvSetEnvTest, StringViewEnvVarResolvesViaShim)
{
    // A non-null-terminated string_view name must be materialised before setenv.
    const std::string_view name = std::string_view{ "FOOBAR" }.substr(0, 3);
    fake_environment::set_env(name, std::string{ "baz" }, 1);
    EXPECT_EQ(fake_environment::get_env("FOO", std::string{}), "baz");
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

// ── Bool branch gaps ──────────────────────────────────────────────────────────

TEST_F(FakeEnvGetEnvTest, BoolEmptyEnvIdReturnsDefault)
{
    EXPECT_FALSE(fake_environment::get_env("", false));
    EXPECT_TRUE(fake_environment::get_env("", true));
}

TEST_F(FakeEnvGetEnvTest, BoolEmptyValueThrows)
{
    fake_env::setenv("FOO", "", 1);
    EXPECT_THROW(fake_environment::get_env("FOO", false), std::runtime_error);
}

TEST_F(FakeEnvGetEnvTest, BoolNumericGreaterThanOneIsTrue)
{
    fake_env::setenv("FOO", "2", 1);
    EXPECT_TRUE(fake_environment::get_env("FOO", false));
}

TEST_F(FakeEnvGetEnvTest, BoolNumericOverflowIsTrueAndDoesNotThrow)
{
    // An all-digit value past UINT64_MAX must not throw (std::stoi would) and
    // is still truthy.
    fake_env::setenv("FOO", "99999999999999999999999", 1);
    EXPECT_NO_THROW({ EXPECT_TRUE(fake_environment::get_env("FOO", false)); });
}

TEST_F(FakeEnvGetEnvTest, BoolUpperCaseFalseVariants)
{
    for(const char* v : { "FALSE", "NO", "OFF", "F", "N" })
    {
        fake_env::reset();
        fake_env::setenv("FOO", v, 1);
        EXPECT_FALSE(fake_environment::get_env("FOO", true)) << "value: " << v;
    }
}

// ── Float branch gaps ─────────────────────────────────────────────────────────

TEST_F(FakeEnvGetEnvTest, FloatEmptyEnvIdReturnsDefault)
{
    EXPECT_NEAR(fake_environment::get_env("", 1.5), 1.5, 1e-9);
}

TEST_F(FakeEnvGetEnvTest, FloatInvalidStringReturnsDefault)
{
    fake_env::setenv("FOO", "not_a_float", 1);
    EXPECT_NEAR(fake_environment::get_env("FOO", 9.9), 9.9, 1e-9);
}

TEST_F(FakeEnvGetEnvTest, FloatTypeReturnsValue)
{
    fake_env::setenv("FOO", "1.5", 1);
    EXPECT_NEAR(fake_environment::get_env("FOO", 0.0f), 1.5f, 1e-6f);
}

TEST_F(FakeEnvGetEnvTest, FloatSurroundingWhitespaceParses)
{
    fake_env::setenv("FOO", "  2.5  ", 1);
    EXPECT_NEAR(fake_environment::get_env("FOO", 0.0), 2.5, 1e-9);
}

TEST_F(FakeEnvGetEnvTest, FloatTrailingGarbageReturnsDefault)
{
    fake_env::setenv("FOO", "1.5abc", 1);
    EXPECT_NEAR(fake_environment::get_env("FOO", 9.9), 9.9, 1e-9);
}

// ── Integral branch gaps ──────────────────────────────────────────────────────

TEST_F(FakeEnvGetEnvTest, IntEmptyEnvIdReturnsDefault)
{
    EXPECT_EQ(fake_environment::get_env("", 99), 99);
}

TEST_F(FakeEnvGetEnvTest, IntInvalidStringReturnsDefault)
{
    fake_env::setenv("FOO", "not_an_int", 1);
    EXPECT_EQ(fake_environment::get_env("FOO", 42), 42);
}

TEST_F(FakeEnvGetEnvTest, LongReturnsValueWhenSet)
{
    fake_env::setenv("FOO", "123456789012", 1);
    EXPECT_EQ(fake_environment::get_env("FOO", 0L), 123456789012L);
}

TEST_F(FakeEnvGetEnvTest, SizeTReturnsValueWhenSet)
{
    fake_env::setenv("FOO", "42", 1);
    EXPECT_EQ(fake_environment::get_env("FOO", std::size_t{ 0 }), std::size_t{ 42 });
}

TEST_F(FakeEnvGetEnvTest, UnsignedNegativeInputReturnsDefault)
{
    // "-1" must not wrap to UINT64_MAX for an unsigned target; fall back instead.
    fake_env::setenv("FOO", "-1", 1);
    EXPECT_EQ(fake_environment::get_env("FOO", std::uint64_t{ 7 }), std::uint64_t{ 7 });
}

TEST_F(FakeEnvGetEnvTest, UnsignedOutOfRangeInputReturnsDefault)
{
    // 70000 does not fit in std::uint16_t -> fall back to the default.
    fake_env::setenv("FOO", "70000", 1);
    EXPECT_EQ(fake_environment::get_env("FOO", std::uint16_t{ 5 }), std::uint16_t{ 5 });
}

TEST_F(FakeEnvGetEnvTest, UnsignedLargeValuePastInt64MaxParses)
{
    // Value > INT64_MAX previously threw out_of_range via stoll; it must now
    // parse correctly for an unsigned 64-bit target.
    constexpr auto big = std::uint64_t{ 18446744073709551610ULL };  // UINT64_MAX - 5
    fake_env::setenv("FOO", "18446744073709551610", 1);
    EXPECT_EQ(fake_environment::get_env("FOO", std::uint64_t{ 0 }), big);
}

TEST_F(FakeEnvGetEnvTest, SignedNegativeInputParses)
{
    // Signed targets must still accept negative values (e.g. ROCPROFSYS_VERBOSE).
    fake_env::setenv("FOO", "-3", 1);
    EXPECT_EQ(fake_environment::get_env("FOO", 0), -3);
}

TEST_F(FakeEnvGetEnvTest, SignedOutOfRangeInputReturnsDefault)
{
    // 70000 does not fit in std::int16_t -> fall back to the default.
    fake_env::setenv("FOO", "70000", 1);
    EXPECT_EQ(fake_environment::get_env("FOO", std::int16_t{ 9 }), std::int16_t{ 9 });
}

TEST_F(FakeEnvGetEnvTest, IntegralTrailingGarbageReturnsDefault)
{
    fake_env::setenv("FOO", "42abc", 1);
    EXPECT_EQ(fake_environment::get_env("FOO", 1), 1);
}

TEST_F(FakeEnvGetEnvTest, IntegralSurroundingWhitespaceParses)
{
    fake_env::setenv("FOO", "  42  ", 1);
    EXPECT_EQ(fake_environment::get_env("FOO", 0), 42);
}

// ── to_env_string ─────────────────────────────────────────────────────────────

TEST(ToEnvStringTest, BoolTrue) { EXPECT_EQ(to_env_string(true), "true"); }

TEST(ToEnvStringTest, BoolFalse) { EXPECT_EQ(to_env_string(false), "false"); }

TEST(ToEnvStringTest, Int) { EXPECT_EQ(to_env_string(42), "42"); }

TEST(ToEnvStringTest, Double) { EXPECT_EQ(to_env_string(3.14), std::to_string(3.14)); }

TEST(ToEnvStringTest, StringPassthrough)
{
    EXPECT_EQ(to_env_string(std::string{ "hello" }), "hello");
}

TEST(ToEnvStringTest, ConstCharPtrPassthrough)
{
    EXPECT_EQ(to_env_string("world"), std::string{ "world" });
}

// ── consolidate_env_entries: no-'=' entry ─────────────────────────────────────

TEST_F(DuplicatedEnvironmentEntriesTest, SkipsEntryWithoutEqualsSign)
{
    std::vector<std::string> env_vars = { "NO_EQUALS", "PATH=/usr/bin" };
    consolidate_env_entries(env_vars);
    ASSERT_EQ(env_vars.size(), 1);
    EXPECT_EQ(env_vars[0], "PATH=/usr/bin");
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
    cfg.m_env_name  = "FOO";
    cfg.m_env_value = "injected";
    cfg.m_override  = 1;
    cfg();
    EXPECT_EQ(fake_environment::get_env("FOO", std::string{}), "injected");
}

TEST_F(FakeEnvConfigTest, OperatorRespectsOverrideZero)
{
    fake_env::setenv("FOO", "original", 1);
    env_config<fake_env> cfg;
    cfg.m_env_name  = "FOO";
    cfg.m_env_value = "new";
    cfg.m_override  = 0;
    cfg();
    EXPECT_EQ(fake_environment::get_env("FOO", std::string{}), "original");
}

TEST_F(FakeEnvConfigTest, OperatorReturnsZeroOnSuccess)
{
    env_config<fake_env> cfg;
    cfg.m_env_name  = "FOO";
    cfg.m_env_value = "some_value";
    cfg.m_override  = 1;
    EXPECT_EQ(cfg(), 0);
    EXPECT_EQ(fake_environment::get_env("FOO", std::string{}), "some_value");
}

TEST_F(FakeEnvConfigTest, EmptyNameIsNoop)
{
    env_config<fake_env> cfg;
    cfg.m_env_name  = "";
    cfg.m_env_value = "ignored";
    cfg.m_override  = 1;
    EXPECT_EQ(cfg(), -1);
    EXPECT_TRUE(fake_env::store.empty());
}

// ── Enum dispatch (get_env<Tp> where is_enum_v<Tp>) ──────────────────────────

class FakeEnvEnumTest : public ::testing::Test
{
protected:
    void SetUp() override { fake_env::reset(); }
    void TearDown() override { fake_env::reset(); }
};

TEST_F(FakeEnvEnumTest, EnumReturnsDefaultWhenUnset)
{
    EXPECT_EQ(fake_environment::get_env("FOO", test_color::green), test_color::green);
}

TEST_F(FakeEnvEnumTest, EnumReturnsValueWhenSet)
{
    fake_env::setenv("FOO", "2", 1);
    EXPECT_EQ(fake_environment::get_env("FOO", test_color::red), test_color::blue);
}

TEST_F(FakeEnvEnumTest, EnumEmptyEnvIdReturnsDefault)
{
    EXPECT_EQ(fake_environment::get_env("", test_color::green), test_color::green);
}

// ── get_env_choice with non-string type ───────────────────────────────────────

class FakeEnvIntChoiceTest : public ::testing::Test
{
protected:
    void SetUp() override { fake_env::reset(); }
    void TearDown() override { fake_env::reset(); }
};

TEST_F(FakeEnvIntChoiceTest, ReturnsDefaultWhenUnset)
{
    auto result = fake_environment::get_env_choice("FOO", 1, std::set<int>{ 1, 2, 3 });
    EXPECT_EQ(result, 1);
}

TEST_F(FakeEnvIntChoiceTest, ReturnsValueWhenValidChoiceSet)
{
    fake_env::setenv("FOO", "3", 1);
    auto result = fake_environment::get_env_choice("FOO", 1, std::set<int>{ 1, 2, 3 });
    EXPECT_EQ(result, 3);
}

TEST_F(FakeEnvIntChoiceTest, ReturnsDefaultWhenInvalidChoiceSet)
{
    fake_env::setenv("FOO", "99", 1);
    auto result = fake_environment::get_env_choice("FOO", 1, std::set<int>{ 1, 2, 3 });
    EXPECT_EQ(result, 1);
}
