// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "avail/catalog.hpp"
#include "avail/records.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <string>

namespace rocprofsys::avail
{
namespace
{

struct expected_diagnostic
{
    capability_kind capability;
    source_id       source;
};

constexpr auto k_expected_v1_stubs = std::array{
    expected_diagnostic{ .capability = capability_kind::gpu_devices,
                         .source     = source_id::rocprofiler_sdk },
    expected_diagnostic{ .capability = capability_kind::cpu_devices,
                         .source     = source_id::procfs },
    expected_diagnostic{ .capability = capability_kind::nic_devices,
                         .source     = source_id::amd_smi },
    expected_diagnostic{ .capability = capability_kind::traces,
                         .source     = source_id::rocprofiler_sdk },
    expected_diagnostic{ .capability = capability_kind::trace_operations,
                         .source     = source_id::rocprofiler_sdk },
    expected_diagnostic{ .capability = capability_kind::gpu_counters,
                         .source     = source_id::rocprofiler_sdk },
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

const diagnostic*
find_diagnostic(const catalog_snapshot& snapshot, capability_kind capability)
{
    const auto itr =
        std::ranges::find_if(snapshot.diagnostics, [capability](const diagnostic& entry) {
            return entry.capability == capability;
        });
    return (itr == snapshot.diagnostics.end()) ? nullptr : &*itr;
}

[[nodiscard]] std::string
stub_mismatch(const catalog_snapshot& snapshot)
{
    if(snapshot.queried.size() != k_expected_v1_stubs.size())
    {
        return "queried size mismatch";
    }
    if(snapshot.diagnostics.size() != k_expected_v1_stubs.size())
    {
        return "diagnostics size mismatch";
    }

    for(const auto& entry : k_expected_v1_stubs)
    {
        if(!snapshot.was_queried(entry.capability))
        {
            return "missing queried capability";
        }
        const auto* diagnostic_entry = find_diagnostic(snapshot, entry.capability);
        if(diagnostic_entry == nullptr)
        {
            return "missing diagnostic";
        }
        if(diagnostic_entry->source != entry.source)
        {
            return "diagnostic source mismatch";
        }
        if(diagnostic_entry->message != "not implemented yet")
        {
            return "diagnostic message mismatch";
        }
    }

    return {};
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
    EXPECT_TRUE(snapshot.was_queried(capability_kind::cpu_counters));
    ASSERT_EQ(snapshot.diagnostics.size(), 1u);
    EXPECT_EQ(snapshot.diagnostics.front().capability, capability_kind::cpu_counters);
    EXPECT_EQ(snapshot.diagnostics.front().source, source_id::papi);
    EXPECT_EQ(snapshot.diagnostics.front().message, "not implemented yet");
    EXPECT_TRUE(snapshot.cpu_counters.empty());
    EXPECT_TRUE(snapshot.degraded());
}

TEST(avail_catalog, every_v1_query_is_represented_by_a_stub)
{
    const auto snapshot = query(query_request{
        .gpu_devices     = true,
        .cpu_devices     = true,
        .nic_devices     = true,
        .traces          = true,
        .operations_for  = "marker_api",
        .gpu_counters    = true,
        .cpu_counters    = true,
        .cpu_metrics     = true,
        .gpu_metrics     = true,
        .nic_metrics     = true,
        .storage_metrics = true,
    });

    EXPECT_EQ(stub_mismatch(snapshot), "");
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
