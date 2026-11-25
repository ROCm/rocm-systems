// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocsys/schema.hpp"

#include "common/env_vars.hpp"
#include "parser/configuration_parser.hpp"
#include "parser/schema.hpp"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace rocprofsys::rocsys::testing
{
namespace
{
/// What --traces writes when it lists nothing: no ROCm domain and every switch off.
parser::value_map_t
nothing_traced()
{
    return { { env_vars::ROCM_DOMAINS, "" },   { env_vars::USE_MPIP, "false" },
             { env_vars::USE_OMPT, "false" },  { env_vars::USE_OPENSHMEM, "false" },
             { env_vars::USE_RCCLP, "false" }, { env_vars::USE_UCX, "false" } };
}

/// The domain of every --traces ROCm API name, as an SDK that supports them all lists
/// them.
std::vector<std::string>
every_domain()
{
    return { "hip_api",     "hsa_api",           "marker_api",  "kernel_dispatch",
             "memory_copy", "memory_allocation", "rocshmem_api" };
}

/// "id arg env" for each option, in help order.
std::vector<std::string>
row_summaries(const parser::schema& catalog)
{
    std::vector<std::string> rows;
    for(const auto& param : catalog.params())
    {
        rows.push_back(fmt::format("{} {} {}", param.id, param.arg.value_or(""),
                                   param.env.value_or("-")));
    }
    return rows;
}

TEST(rocsys_schema, catalog_is_valid_with_no_defaults_or_short_spellings)
{
    parser::configuration_parser config;
    EXPECT_TRUE(config.initialize(make_schema())) << config.failure();
    const auto catalog = make_schema();
    for(const auto& param : catalog.params())
    {
        EXPECT_FALSE(param.default_value.has_value()) << param.id;
        EXPECT_TRUE(param.arg_aliases.empty()) << param.id;
    }
}

TEST(rocsys_schema, options_have_their_spelling_and_variable)
{
    EXPECT_EQ(row_summaries(make_schema()),
              (std::vector<std::string>{
                  "traces --traces -", "output_path --output-path ROCPROFSYS_OUTPUT_PATH",
                  "log_level --log-level ROCPROFSYS_LOG_LEVEL" }));
}

TEST(rocsys_schema, trace_names_are_the_documented_ones)
{
    const auto  catalog = make_schema();
    const auto* traces  = catalog.find_param("traces");
    ASSERT_NE(traces, nullptr);
    EXPECT_EQ(traces->choices,
              (std::vector<std::string>{ "rccl", "mpi", "openmp", "ucx", "openshmem" }));
    EXPECT_EQ(traces->backend, "rocprofiler-sdk");
    EXPECT_EQ(make_trace_backend(every_domain())->choices(*traces),
              (std::vector<std::string>{ "hip-api", "hip", "hsa-api", "hsa", "roctx",
                                         "kernel-dispatch", "memory-copy",
                                         "memory-allocation", "rocshmem" }));
}

TEST(rocsys_schema, the_backend_lists_the_names_whose_domain_the_sdk_supports)
{
    const auto  catalog = make_schema();
    const auto* traces  = catalog.find_param("traces");
    ASSERT_NE(traces, nullptr);
    const auto backend = make_trace_backend({ "marker_api", "hip_api", "ompt" });
    EXPECT_EQ(backend->name(), "rocprofiler-sdk");
    EXPECT_EQ(backend->choices(*traces),
              (std::vector<std::string>{ "hip-api", "hip", "roctx" }));
    EXPECT_TRUE(make_trace_backend({})->choices(*traces).empty());
}

TEST(rocsys_schema, every_trace_name_sets_something)
{
    const auto  catalog = make_schema();
    const auto* traces  = catalog.find_param("traces");
    ASSERT_NE(traces, nullptr);
    auto names = make_trace_backend(every_domain())->choices(*traces);
    names.insert(names.end(), traces->choices.begin(), traces->choices.end());
    for(const auto& name : names)
    {
        EXPECT_NE(traces_environment({ name }), nothing_traced()) << name;
    }
}

TEST(rocsys_schema, an_alias_selects_its_row)
{
    EXPECT_EQ(traces_environment({ "hip" }), traces_environment({ "hip-api" }));
    EXPECT_EQ(traces_environment({ "hsa" }), traces_environment({ "hsa-api" }));
    EXPECT_EQ(traces_environment({ "hip", "hip-api" }).at(env_vars::ROCM_DOMAINS),
              "hip_api");
}

TEST(rocsys_schema, traces_environment_is_the_whole_selection)
{
    auto expected                    = nothing_traced();
    expected[env_vars::ROCM_DOMAINS] = "hip_api,kernel_dispatch";
    expected[env_vars::USE_MPIP]     = "true";
    EXPECT_EQ(traces_environment({ "kernel-dispatch", "mpi", "hip-api" }), expected);
    EXPECT_EQ(traces_environment({}), nothing_traced());
}
}  // namespace
}  // namespace rocprofsys::rocsys::testing
