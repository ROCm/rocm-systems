// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "avail/traces.hpp"

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rocprofsys::avail
{
namespace
{
struct fake_traces
{
    std::vector<std::string>                                  callback   = {};
    std::vector<std::string>                                  buffered   = {};
    std::unordered_map<std::string, std::vector<std::string>> operations = {};
    std::string                                               failure    = {};

    std::vector<trace_kind_entry> as_entries(const std::vector<std::string>& names) const
    {
        auto entries = std::vector<trace_kind_entry>{};
        entries.reserve(names.size());
        for(const auto& name : names)
        {
            auto entry = trace_kind_entry{ .name = name };
            auto itr   = operations.find(utility::string::to_lower(name));
            if(itr == operations.end()) itr = operations.find(name);
            if(itr != operations.end()) entry.operations = itr->second;
            entries.push_back(std::move(entry));
        }
        return entries;
    }

    std::vector<trace_kind_entry> callback_domains() const
    {
        if(!failure.empty()) throw std::runtime_error{ failure };
        return as_entries(callback);
    }

    std::vector<trace_kind_entry> buffered_domains() const
    {
        if(!failure.empty()) throw std::runtime_error{ failure };
        return as_entries(buffered);
    }
};

static_assert(trace_inventory_source<fake_traces>);

const trace_domain_record*
find_trace(const std::vector<trace_domain_record>& traces, std::string_view name)
{
    for(const auto& trace : traces)
    {
        if(trace.name == name) return &trace;
    }
    return nullptr;
}

TEST(avail_traces, lists_callback_and_buffered_sdk_domains)
{
    auto source     = fake_traces{};
    source.callback = { "HIP_RUNTIME_API", "HSA_CORE_API" };
    source.buffered = { "KERNEL_DISPATCH", "MEMORY_COPY" };

    const auto result = collect_traces(source);

    ASSERT_NE(find_trace(result.traces, "hip_runtime_api"), nullptr);
    EXPECT_TRUE(find_trace(result.traces, "hip_runtime_api")->callback);
    EXPECT_FALSE(find_trace(result.traces, "hip_runtime_api")->buffered);

    ASSERT_NE(find_trace(result.traces, "kernel_dispatch"), nullptr);
    EXPECT_TRUE(find_trace(result.traces, "kernel_dispatch")->buffered);
    EXPECT_FALSE(find_trace(result.traces, "kernel_dispatch")->callback);
    EXPECT_TRUE(result.diagnostics.empty());
}

TEST(avail_traces, a_name_in_both_tables_is_one_domain)
{
    auto source     = fake_traces{};
    source.callback = { "hip_runtime_api" };
    source.buffered = { "hip_runtime_api" };

    const auto  result = collect_traces(source);
    const auto* hip    = find_trace(result.traces, "hip_runtime_api");

    ASSERT_NE(hip, nullptr);
    EXPECT_TRUE(hip->callback);
    EXPECT_TRUE(hip->buffered);
}

TEST(avail_traces, drops_internal_sdk_kinds)
{
    auto source     = fake_traces{};
    source.callback = { "none", "code_object", "marker_core_api" };
    source.buffered = { "correlation_id_retirement" };

    const auto result = collect_traces(source);

    EXPECT_EQ(find_trace(result.traces, "none"), nullptr);
    EXPECT_EQ(find_trace(result.traces, "code_object"), nullptr);
    EXPECT_EQ(find_trace(result.traces, "marker_core_api"), nullptr);
    EXPECT_EQ(find_trace(result.traces, "correlation_id_retirement"), nullptr);
}

TEST(avail_traces, drops_sdk_domains_this_tool_does_not_trace)
{
    auto source     = fake_traces{};
    source.callback = { "runtime_initialization", "hip_stream", "hip_graph",
                        "hip_runtime_api_ext", "marker_core_range_api" };

    const auto result = collect_traces(source);

    EXPECT_EQ(find_trace(result.traces, "runtime_initialization"), nullptr);
    EXPECT_EQ(find_trace(result.traces, "hip_stream"), nullptr);
    EXPECT_EQ(find_trace(result.traces, "hip_graph"), nullptr);
    EXPECT_EQ(find_trace(result.traces, "hip_runtime_api_ext"), nullptr);
    EXPECT_EQ(find_trace(result.traces, "marker_core_range_api"), nullptr);
}

TEST(avail_traces, adds_user_facing_aliases)
{
    auto source     = fake_traces{};
    source.callback = { "hip_runtime_api" };
    source.buffered = { "kernel_dispatch" };

    const auto result = collect_traces(source);

    for(const char* alias : { "hip_api", "hsa_api", "marker_api", "roctx" })
    {
        const auto* record = find_trace(result.traces, alias);
        ASSERT_NE(record, nullptr) << alias;
        EXPECT_TRUE(record->alias) << alias;
        EXPECT_FALSE(record->alias_members.empty()) << alias;
    }

    const auto* hip = find_trace(result.traces, "hip_api");
    ASSERT_NE(hip, nullptr);
    EXPECT_EQ(hip->alias_members,
              (std::vector<std::string>{ "hip_compiler_api", "hip_runtime_api" }));

    const auto* marker = find_trace(result.traces, "marker_api");
    ASSERT_NE(marker, nullptr);
    EXPECT_EQ(marker->alias_members, (std::vector<std::string>{ "marker_core_api" }));
}

TEST(avail_traces, kfd_events_alias_only_when_kfd_domains_exist)
{
    auto without_kfd     = fake_traces{};
    without_kfd.buffered = { "kernel_dispatch" };
    EXPECT_EQ(find_trace(collect_traces(without_kfd).traces, "kfd_events"), nullptr);

    auto with_kfd     = fake_traces{};
    with_kfd.buffered = { "kfd_page_fault", "kfd_queue" };
    const auto* alias = find_trace(collect_traces(with_kfd).traces, "kfd_events");
    ASSERT_NE(alias, nullptr);
    EXPECT_TRUE(alias->alias);
    EXPECT_EQ(alias->alias_members,
              (std::vector<std::string>{ "kfd_page_fault", "kfd_queue" }));
}

TEST(avail_traces, marks_the_profiler_default_domains)
{
    auto source     = fake_traces{};
    source.callback = { "hip_runtime_api" };
    source.buffered = { "kernel_dispatch", "memory_copy", "scratch_memory" };

    const auto result = collect_traces(source);

    EXPECT_TRUE(find_trace(result.traces, "hip_runtime_api")->is_default);
    EXPECT_TRUE(find_trace(result.traces, "marker_api")->is_default);
    EXPECT_TRUE(find_trace(result.traces, "kernel_dispatch")->is_default);
    EXPECT_FALSE(find_trace(result.traces, "hip_api")->is_default);
}

TEST(avail_traces, empty_tables_still_report_aliases)
{
    auto source = fake_traces{};

    const auto result = collect_traces(source);

    EXPECT_GE(result.traces.size(), 4U);
    EXPECT_TRUE(result.diagnostics.empty());
}

TEST(avail_traces, backend_failure_is_a_diagnostic)
{
    auto source    = fake_traces{};
    source.failure = "sdk names unavailable";

    const auto result = collect_traces(source);

    EXPECT_TRUE(result.traces.empty());
    ASSERT_EQ(result.diagnostics.size(), 1U);
    EXPECT_EQ(result.diagnostics[0].source, source_id::rocprofiler_sdk);
    EXPECT_EQ(result.diagnostics[0].message, "sdk names unavailable");
}

TEST(avail_traces, domains_are_sorted_by_name)
{
    auto source     = fake_traces{};
    source.buffered = { "scratch_memory", "kernel_dispatch" };

    const auto result = collect_traces(source);
    ASSERT_GE(result.traces.size(), 2U);
    for(std::size_t i = 1; i < result.traces.size(); ++i)
        EXPECT_LT(result.traces[i - 1].name, result.traces[i].name);
}

TEST(avail_traces, attaches_operations_and_unions_them_on_aliases)
{
    auto source     = fake_traces{};
    source.callback = { "hip_runtime_api", "hip_compiler_api", "marker_core_api" };
    source.operations["hip_runtime_api"]  = { "hipMalloc", "none", "hipFree" };
    source.operations["hip_compiler_api"] = { "__hipPushCallConfiguration" };
    source.operations["marker_core_api"]  = { "roctxRangePush" };

    const auto  result = collect_traces(source);
    const auto* hip    = find_trace(result.traces, "hip_runtime_api");
    ASSERT_NE(hip, nullptr);
    EXPECT_EQ(hip->operations, (std::vector<std::string>{ "hipFree", "hipMalloc" }));

    const auto* hip_api = find_trace(result.traces, "hip_api");
    ASSERT_NE(hip_api, nullptr);
    EXPECT_EQ(hip_api->operations,
              (std::vector<std::string>{ "__hipPushCallConfiguration", "hipFree",
                                         "hipMalloc" }));

    const auto* marker = find_trace(result.traces, "marker_api");
    ASSERT_NE(marker, nullptr);
    EXPECT_EQ(marker->operations, (std::vector<std::string>{ "roctxRangePush" }));
    EXPECT_EQ(find_trace(result.traces, "marker_core_api"), nullptr);
}

TEST(avail_traces, injects_host_runtime_tokens)
{
    auto source = fake_traces{};

    const auto result = collect_traces(source);
    for(const char* name : { "mpi", "ucx", "oshmem", "kokkos", "ompt", "vaapi", "osrt" })
        ASSERT_NE(find_trace(result.traces, name), nullptr) << name;
}

TEST(avail_traces, classifies_sections_and_hides_alias_members_from_listing)
{
    EXPECT_EQ(section_for_trace("hip_api"), trace_section::gpu_rocm);
    EXPECT_EQ(section_for_trace("ompt"), trace_section::host);
    EXPECT_EQ(section_for_trace("mpi"), trace_section::host);
    EXPECT_EQ(section_for_trace("vaapi"), trace_section::other);
    EXPECT_EQ(section_for_trace("hip_stream"), trace_section::other);
    EXPECT_EQ(section_for_trace("runtime_initialization"), trace_section::other);

    auto source        = fake_traces{};
    source.callback    = { "hip_runtime_api", "hip_compiler_api" };
    const auto  result = collect_traces(source);
    const auto* hip_rt = find_trace(result.traces, "hip_runtime_api");
    ASSERT_NE(hip_rt, nullptr);
    EXPECT_TRUE(is_hidden_alias_member(*hip_rt, result.traces));
    EXPECT_FALSE(
        is_hidden_alias_member(*find_trace(result.traces, "hip_api"), result.traces));
}
}  // namespace
}  // namespace rocprofsys::avail
