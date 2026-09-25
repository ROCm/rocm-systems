// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "avail/catalog.hpp"
#include "avail/records.hpp"
#include "avail/traces.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <string>
#include <string_view>

namespace rocprofsys::avail
{
namespace
{

struct expected_diagnostic
{
    capability_kind capability;
    source_id       source;
};

const diagnostic*
find_diagnostic(const catalog_snapshot& snapshot, capability_kind capability)
{
    const auto itr =
        std::ranges::find_if(snapshot.diagnostics, [capability](const diagnostic& entry) {
            return entry.capability == capability;
        });
    return (itr == snapshot.diagnostics.end()) ? nullptr : &*itr;
}

void
expect_stub_diagnostic(const catalog_snapshot& snapshot, capability_kind capability,
                       source_id source)
{
    EXPECT_TRUE(snapshot.was_queried(capability));
    const auto* diagnostic_entry = find_diagnostic(snapshot, capability);
    ASSERT_NE(diagnostic_entry, nullptr);
    EXPECT_EQ(diagnostic_entry->source, source);
    EXPECT_EQ(diagnostic_entry->message, k_not_implemented_message);
}

TEST(avail_catalog, empty_request_performs_no_queries)
{
    const auto snapshot = query(query_request{});

    EXPECT_TRUE(snapshot.queried.empty());
    EXPECT_TRUE(snapshot.diagnostics.empty());
    EXPECT_FALSE(snapshot.degraded());
}

TEST(avail_catalog, requested_stub_has_stable_message_and_source)
{
    query_request request = {};
    request.cpu_counters  = true;

    const auto snapshot = query(request);

    ASSERT_EQ(snapshot.queried.size(), 1u);
    ASSERT_EQ(snapshot.diagnostics.size(), 1u);
    expect_stub_diagnostic(snapshot, capability_kind::cpu_counters, source_id::papi);
    EXPECT_TRUE(snapshot.cpu_counters.empty());
    EXPECT_TRUE(snapshot.degraded());
}

TEST(avail_catalog, remaining_stub_queries_keep_not_implemented_message)
{
    query_request request   = {};
    request.cpu_devices     = true;
    request.nic_devices     = true;
    request.cpu_counters    = true;
    request.cpu_metrics     = true;
    request.gpu_metrics     = true;
    request.nic_metrics     = true;
    request.storage_metrics = true;

    constexpr auto k_expected = std::array{
        expected_diagnostic{ .capability = capability_kind::cpu_devices,
                             .source     = source_id::procfs },
        expected_diagnostic{ .capability = capability_kind::nic_devices,
                             .source     = source_id::amd_smi },
        expected_diagnostic{ .capability = capability_kind::cpu_counters,
                             .source     = source_id::papi },
        expected_diagnostic{ .capability = capability_kind::cpu_metrics,
                             .source     = source_id::procfs },
        expected_diagnostic{ .capability = capability_kind::gpu_metrics,
                             .source     = source_id::amd_smi },
        expected_diagnostic{ .capability = capability_kind::nic_metrics,
                             .source     = source_id::amd_smi },
        expected_diagnostic{ .capability = capability_kind::storage_metrics,
                             .source     = source_id::storage },
    };

    const auto snapshot = query(request);

    ASSERT_EQ(snapshot.queried.size(), k_expected.size());
    ASSERT_EQ(snapshot.diagnostics.size(), k_expected.size());
    for(const auto& entry : k_expected)
    {
        expect_stub_diagnostic(snapshot, entry.capability, entry.source);
    }
}

TEST(avail_catalog, gpu_devices_query_is_not_a_stub)
{
    query_request request = {};
    request.gpu_devices   = true;

    const auto snapshot = query(request);

    EXPECT_TRUE(snapshot.was_queried(capability_kind::gpu_devices));
    const auto* diagnostic_entry =
        find_diagnostic(snapshot, capability_kind::gpu_devices);
    if(diagnostic_entry != nullptr)
    {
        EXPECT_NE(diagnostic_entry->message, k_not_implemented_message);
    }
#if !defined(ROCPROFSYS_AVAIL_HAS_SDK)
    ASSERT_NE(diagnostic_entry, nullptr);
    EXPECT_EQ(diagnostic_entry->message, k_sdk_unavailable_message);
    EXPECT_TRUE(snapshot.gpu_devices.empty());
#endif
}

TEST(avail_catalog, gpu_counters_query_is_not_a_stub)
{
    query_request request = {};
    request.gpu_counters  = true;

    const auto snapshot = query(request);

    EXPECT_TRUE(snapshot.was_queried(capability_kind::gpu_counters));
    const auto* diagnostic_entry =
        find_diagnostic(snapshot, capability_kind::gpu_counters);
    if(diagnostic_entry != nullptr)
    {
        EXPECT_NE(diagnostic_entry->message, k_not_implemented_message);
    }
#if !defined(ROCPROFSYS_AVAIL_HAS_SDK)
    ASSERT_NE(diagnostic_entry, nullptr);
    EXPECT_EQ(diagnostic_entry->message, k_sdk_unavailable_message);
    EXPECT_TRUE(snapshot.gpu_counters.empty());
#endif
}

[[nodiscard]] const trace_record*
find_trace(const catalog_snapshot& snapshot, std::string_view name)
{
    const auto itr =
        std::ranges::find_if(snapshot.traces, [name](const trace_record& entry) {
            return entry.name == name;
        });
    return (itr == snapshot.traces.end()) ? nullptr : &*itr;
}

void
expect_csv_contains(const std::string& csv, std::string_view token)
{
    EXPECT_NE(csv.find(token), std::string::npos);
}

#if defined(ROCPROFSYS_AVAIL_HAS_SDK)
TEST(avail_catalog, traces_query_lists_sdk_inventory)
{
    query_request request = {};
    request.traces        = true;

    const auto snapshot = query(request);

    EXPECT_TRUE(snapshot.was_queried(capability_kind::traces));
    EXPECT_EQ(find_diagnostic(snapshot, capability_kind::traces), nullptr);
    EXPECT_FALSE(snapshot.traces.empty());
    EXPECT_FALSE(snapshot.default_traces.empty());
    expect_csv_contains(snapshot.default_traces_csv(), "hip_runtime_api");
    expect_csv_contains(snapshot.default_traces_csv(), "marker_api");
    expect_csv_contains(snapshot.available_traces_csv(), "hip_api");

    const auto* hip_api = find_trace(snapshot, "hip_api");
    ASSERT_NE(hip_api, nullptr);
    EXPECT_FALSE(hip_api->description.empty());
    EXPECT_FALSE(hip_api->aliases.empty());

    const auto* kernel = find_trace(snapshot, "kernel_dispatch");
    ASSERT_NE(kernel, nullptr);
    EXPECT_TRUE(kernel->is_default);
}
#else
TEST(avail_catalog, traces_query_reports_sdk_unavailable)
{
    query_request request = {};
    request.traces        = true;

    const auto snapshot = query(request);

    EXPECT_TRUE(snapshot.was_queried(capability_kind::traces));
    const auto* diagnostic_entry = find_diagnostic(snapshot, capability_kind::traces);
    ASSERT_NE(diagnostic_entry, nullptr);
    EXPECT_EQ(diagnostic_entry->source, source_id::rocprofiler_sdk);
    EXPECT_EQ(diagnostic_entry->message, k_sdk_unavailable_message);
    EXPECT_TRUE(snapshot.traces.empty());
}
#endif

TEST(avail_catalog, marker_api_operations_are_not_a_stub)
{
    query_request request  = {};
    request.operations_for = "marker_api";

    const auto snapshot = query(request);

    EXPECT_TRUE(snapshot.was_queried(capability_kind::trace_operations));
    const auto* diagnostic_entry =
        find_diagnostic(snapshot, capability_kind::trace_operations);
    if(diagnostic_entry != nullptr)
    {
        EXPECT_NE(diagnostic_entry->message, k_not_implemented_message);
    }
#if defined(ROCPROFSYS_AVAIL_HAS_SDK)
    EXPECT_EQ(diagnostic_entry, nullptr);
    EXPECT_FALSE(snapshot.trace_operations.empty());
    EXPECT_EQ(snapshot.trace_operations.front().trace_name, "marker_api");
#else
    ASSERT_NE(diagnostic_entry, nullptr);
    EXPECT_EQ(diagnostic_entry->message, k_sdk_unavailable_message);
    EXPECT_TRUE(snapshot.trace_operations.empty());
#endif
}

TEST(avail_catalog, unknown_operations_query_is_not_a_stub)
{
    query_request request  = {};
    request.operations_for = "not_a_real_trace_domain";

    const auto snapshot = query(request);

    EXPECT_TRUE(snapshot.was_queried(capability_kind::trace_operations));
    EXPECT_TRUE(snapshot.trace_operations.empty());
    const auto* diagnostic_entry =
        find_diagnostic(snapshot, capability_kind::trace_operations);
    ASSERT_NE(diagnostic_entry, nullptr);
    EXPECT_NE(diagnostic_entry->message, k_not_implemented_message);
    EXPECT_EQ(diagnostic_entry->source, source_id::rocprofiler_sdk);
#if defined(ROCPROFSYS_AVAIL_HAS_SDK)
    EXPECT_EQ(diagnostic_entry->message,
              unknown_trace_message("not_a_real_trace_domain"));
#else
    EXPECT_EQ(diagnostic_entry->message, k_sdk_unavailable_message);
#endif
}

TEST(avail_catalog, unrequested_capabilities_are_not_reported)
{
    query_request request = {};
    request.gpu_metrics   = true;

    const auto snapshot = query(request);

    EXPECT_TRUE(snapshot.was_queried(capability_kind::gpu_metrics));
    EXPECT_FALSE(snapshot.was_queried(capability_kind::cpu_metrics));
    EXPECT_EQ(find_diagnostic(snapshot, capability_kind::cpu_metrics), nullptr);
}

TEST(avail_records, enum_names_are_stable)
{
    EXPECT_EQ(to_string(source_id::rocprofiler_sdk), "rocprofiler-sdk");
    EXPECT_EQ(to_string(capability_kind::trace_operations), "trace-operations");
}

}  // namespace
}  // namespace rocprofsys::avail
