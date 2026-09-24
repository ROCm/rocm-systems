// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "avail/catalog.hpp"
#include "avail/traces.hpp"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys::avail
{
namespace
{

struct fake_tracing_config
{
    struct operation_options_env_names
    {
        std::string operations_include_env_name;
    };

    struct operation_setting_spec
    {
        operation_options_env_names env_names;
        std::vector<std::string>    operation_choices;
    };

    static std::vector<std::string> get_domain_choices()
    {
        return { "hip_api", "hip_runtime_api", "kernel_dispatch", "marker_api" };
    }

    static std::string get_domain_defaults() { return "marker_api,kernel_dispatch"; }

    static std::string get_domain_description(std::string_view name)
    {
        if(name == "hip_api")
        {
            return "Alias for HIP runtime and compiler API tracing";
        }
        if(name == "kernel_dispatch")
        {
            return "GPU kernel dispatch tracing";
        }
        return {};
    }

    static std::vector<std::string> get_domain_members(std::string_view name)
    {
        if(name == "hip_api")
        {
            return { "hip_runtime_api" };
        }
        return {};
    }

    static std::vector<operation_setting_spec> get_operation_settings()
    {
        return {
            operation_setting_spec{
                .env_names         = { .operations_include_env_name =
                                           "ROCPROFSYS_ROCM_MARKER_API_OPERATIONS" },
                .operation_choices = { "roctxMarkA", "roctxRangePushA" },
            },
            operation_setting_spec{
                .env_names         = { .operations_include_env_name =
                                           "ROCPROFSYS_ROCM_MARKER_API_OPERATIONS" },
                .operation_choices = { "roctxMarkA" },
            },
        };
    }
};

TEST(avail_traces, domain_from_operations_env_extracts_lowercase_name)
{
    const auto domain =
        domain_from_operations_env("ROCPROFSYS_ROCM_MARKER_API_OPERATIONS");
    EXPECT_EQ(domain, std::optional<std::string>{ "marker_api" });
}

TEST(avail_traces, domain_from_operations_env_rejects_exclude_suffix)
{
    EXPECT_FALSE(
        domain_from_operations_env("ROCPROFSYS_ROCM_MARKER_API_OPERATIONS_EXCLUDE")
            .has_value());
}

TEST(avail_traces, inventory_maps_domain_choices_to_trace_records)
{
    const auto listed  = inventory::traces<fake_tracing_config>();
    const auto records = listed.records;
    ASSERT_EQ(records.size(), 4u);
    EXPECT_EQ(records[0].name, "hip_api");
    EXPECT_EQ(records[1].name, "hip_runtime_api");
    EXPECT_EQ(records[2].name, "kernel_dispatch");
    EXPECT_EQ(records[3].name, "marker_api");
}

TEST(avail_traces, inventory_marks_defaults_descriptions_and_group_aliases)
{
    const auto listed = inventory::traces<fake_tracing_config>();

    ASSERT_EQ(listed.defaults.size(), 2u);
    EXPECT_EQ(listed.defaults[0], "marker_api");
    EXPECT_EQ(listed.defaults[1], "kernel_dispatch");

    const auto& hip_api = listed.records[0];
    EXPECT_FALSE(hip_api.is_default);
    EXPECT_EQ(hip_api.description, "Alias for HIP runtime and compiler API tracing");
    ASSERT_EQ(hip_api.aliases.size(), 1u);
    EXPECT_EQ(hip_api.aliases.front(), "hip_runtime_api");

    const auto& hip_runtime = listed.records[1];
    EXPECT_THAT(hip_runtime.aliases, ::testing::Contains("hip_api"));

    EXPECT_TRUE(listed.records[2].is_default);
    EXPECT_EQ(listed.records[2].description, "GPU kernel dispatch tracing");
    EXPECT_TRUE(listed.records[3].is_default);
}

TEST(avail_traces, inventory_lists_operations_for_matching_domain)
{
    const auto result = inventory::operations<fake_tracing_config>("MARKER_API");
    EXPECT_FALSE(result.issue.has_value());
    ASSERT_EQ(result.records.size(), 2u);
    EXPECT_EQ(result.records[0].trace_name, "marker_api");
    EXPECT_EQ(result.records[0].name, "roctxMarkA");
    EXPECT_EQ(result.records[1].name, "roctxRangePushA");
}

[[nodiscard]] std::string
issue_message(const operations_listing_result& result)
{
    if(!result.issue.has_value())
    {
        return {};
    }
    return result.issue->message;
}

TEST(avail_traces, inventory_unknown_domain_is_not_a_stub)
{
    const auto result = inventory::operations<fake_tracing_config>("roctx");
    EXPECT_TRUE(result.records.empty());
    ASSERT_TRUE(result.issue.has_value());
    EXPECT_EQ(issue_message(result), unknown_trace_message("roctx"));
    EXPECT_NE(issue_message(result), k_not_implemented_message);
}

TEST(avail_traces, inventory_known_domain_without_operations)
{
    const auto result = inventory::operations<fake_tracing_config>("hip_api");
    EXPECT_TRUE(result.records.empty());
    ASSERT_TRUE(result.issue.has_value());
    EXPECT_EQ(issue_message(result), trace_has_no_operations_message("hip_api"));
}

}  // namespace
}  // namespace rocprofsys::avail
