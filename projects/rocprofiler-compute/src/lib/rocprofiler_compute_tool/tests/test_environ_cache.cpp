// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "test_environ_cache.h"

#include "environ_cache.h"
#include "input_parameters.h"

#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <string_view>

using namespace rocprofiler_compute_tool;

TEST_F(TestEnvironCache, Instance_ReturnsSameSingleton_AcrossCalls)
{
    const auto first  = EnvironCache::instance();
    const auto second = EnvironCache::instance();

    // Both calls must return the same instance; the singleton is constructed
    // once at load time.
    EXPECT_EQ(first.get(), second.get());
}

TEST_F(TestEnvironCache, RocprofKeyInEnviron_ReturnsValue)
{
    Envp         envp{{"ROCPROF_OUTPUT_PATH=/tmp/x"}};
    EnvironCache cache{envp.data()};
    EXPECT_EQ(cache.get("ROCPROF_OUTPUT_PATH"),
              std::optional<std::string_view>{std::string_view{"/tmp/x"}});
}

TEST_F(TestEnvironCache, MissingRocprofKey_ReturnsNullopt)
{
    Envp         envp{{}};
    EnvironCache cache{envp.data()};
    EXPECT_EQ(cache.get("ROCPROF_OUTPUT_PATH"), std::nullopt);
}

TEST_F(TestEnvironCache, NonRocprofKeyInEnviron_NotCaptured)
{
    Envp         envp{{"FOO=bar"}};
    EnvironCache cache{envp.data()};
    EXPECT_EQ(cache.get("FOO"), std::nullopt);
}

TEST_F(TestEnvironCache, EmptyValueRocprofKey_ReturnsEmptyView)
{
    Envp         envp{{"ROCPROF_OUTPUT_PATH="}};
    EnvironCache cache{envp.data()};
    EXPECT_EQ(cache.get("ROCPROF_OUTPUT_PATH"), std::optional<std::string_view>{std::string_view{}});
}

TEST_F(TestEnvironCache, PrefixOnlyMatchInEnviron_ReturnsNullopt)
{
    Envp         envp{{"ROCPROF_OUTPUT_PATH_EXTRA=junk"}};
    EnvironCache cache{envp.data()};
    EXPECT_EQ(cache.get("ROCPROF_OUTPUT_PATH"), std::nullopt);
}

TEST_F(TestEnvironCache, MultipleRocprofKeys_AllCaptured)
{
    Envp         envp{{"ROCPROF_OUTPUT_PATH=/tmp/p",
                       "ROCPROF_COUNTERS=SQ_WAVES",
                       "ROCPROF_ITERATION_MULTIPLEXING=kernel",
                       "ROCPROF_KERNEL_FILTER_INCLUDE_REGEX=.*conv.*",
                       "ROCPROF_KERNEL_FILTER_RANGE=1-5"}};
    EnvironCache cache{envp.data()};
    EXPECT_EQ(cache.get("ROCPROF_OUTPUT_PATH"),
              std::optional<std::string_view>{std::string_view{"/tmp/p"}});
    EXPECT_EQ(cache.get("ROCPROF_COUNTERS"),
              std::optional<std::string_view>{std::string_view{"SQ_WAVES"}});
    EXPECT_EQ(cache.get("ROCPROF_ITERATION_MULTIPLEXING"),
              std::optional<std::string_view>{std::string_view{"kernel"}});
    EXPECT_EQ(cache.get("ROCPROF_KERNEL_FILTER_INCLUDE_REGEX"),
              std::optional<std::string_view>{std::string_view{".*conv.*"}});
    EXPECT_EQ(cache.get("ROCPROF_KERNEL_FILTER_RANGE"),
              std::optional<std::string_view>{std::string_view{"1-5"}});
}

TEST_F(TestEnvironCache, DuplicateEnvironEntries_FirstWins)
{
    Envp         envp{{"ROCPROF_OUTPUT_PATH=/tmp/first", "ROCPROF_OUTPUT_PATH=/tmp/second"}};
    EnvironCache cache{envp.data()};
    EXPECT_EQ(cache.get("ROCPROF_OUTPUT_PATH"),
              std::optional<std::string_view>{std::string_view{"/tmp/first"}});
}

TEST_F(TestEnvironCache, EnvInputParametersInjectedCache_ReturnsInjectedValues)
{
    Envp               envp{{"ROCPROF_OUTPUT_PATH=/tmp/injected",
                             "ROCPROF_COUNTERS=SQ_WAVES",
                             "ROCPROF_ITERATION_MULTIPLEXING=kernel",
                             "ROCPROF_KERNEL_FILTER_INCLUDE_REGEX=.*gemm.*",
                             "ROCPROF_KERNEL_FILTER_RANGE=3-7"}};
    EnvInputParameters input_parameters{std::make_shared<EnvironCache>(envp.data())};

    EXPECT_EQ(input_parameters.get_output_path(), std::string_view{"/tmp/injected"});
    EXPECT_EQ(input_parameters.get_requested_counters(), std::string_view{"SQ_WAVES"});
    EXPECT_EQ(input_parameters.get_iteration_multiplexing_mode(), std::string_view{"kernel"});
    EXPECT_EQ(input_parameters.get_kernel_filter_include_regex(), std::string_view{".*gemm.*"});
    EXPECT_EQ(input_parameters.get_kernel_filter_range(), std::string_view{"3-7"});
}
