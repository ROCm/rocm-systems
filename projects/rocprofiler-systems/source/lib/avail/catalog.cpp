// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "avail/catalog.hpp"
#include "avail/records.hpp"

#include <algorithm>
#include <array>
#include <string>
#include <string_view>

namespace rocprofsys::avail
{
namespace
{

constexpr std::string_view k_not_implemented_message = "not implemented yet";

struct stub_flag
{
    bool query_request::*requested;
    capability_kind      capability;
    source_id            source;
};

constexpr auto k_stub_flags = std::array{
    stub_flag{ .requested  = &query_request::gpu_devices,
               .capability = capability_kind::gpu_devices,
               .source     = source_id::rocprofiler_sdk },
    stub_flag{ .requested  = &query_request::cpu_devices,
               .capability = capability_kind::cpu_devices,
               .source     = source_id::procfs },
    stub_flag{ .requested  = &query_request::nic_devices,
               .capability = capability_kind::nic_devices,
               .source     = source_id::amd_smi },
    stub_flag{ .requested  = &query_request::traces,
               .capability = capability_kind::traces,
               .source     = source_id::rocprofiler_sdk },
    stub_flag{ .requested  = &query_request::gpu_counters,
               .capability = capability_kind::gpu_counters,
               .source     = source_id::rocprofiler_sdk },
    stub_flag{ .requested  = &query_request::cpu_counters,
               .capability = capability_kind::cpu_counters,
               .source     = source_id::papi },
    stub_flag{ .requested  = &query_request::cpu_metrics,
               .capability = capability_kind::cpu_metrics,
               .source     = source_id::procfs },
    stub_flag{ .requested  = &query_request::gpu_metrics,
               .capability = capability_kind::gpu_metrics,
               .source     = source_id::amd_smi },
    stub_flag{ .requested  = &query_request::nic_metrics,
               .capability = capability_kind::nic_metrics,
               .source     = source_id::amd_smi },
    stub_flag{ .requested  = &query_request::storage_metrics,
               .capability = capability_kind::storage_metrics,
               .source     = source_id::storage },
};

void
append_stub(catalog_snapshot& snapshot, capability_kind capability, source_id source)
{
    snapshot.queried.emplace_back(capability);
    snapshot.diagnostics.emplace_back(
        diagnostic{ .capability = capability,
                    .source     = source,
                    .message    = std::string{ k_not_implemented_message } });
}

}  // namespace

bool
catalog_snapshot::was_queried(capability_kind capability) const noexcept
{
    return std::ranges::find(queried, capability) != queried.end();
}

catalog_snapshot
query(const query_request& request)
{
    catalog_snapshot result = {};

    for(const auto& flag : k_stub_flags)
    {
        if(request.*(flag.requested))
        {
            append_stub(result, flag.capability, flag.source);
        }
    }

    if(request.operations_for)
    {
        append_stub(result, capability_kind::trace_operations,
                    source_id::rocprofiler_sdk);
    }

    return result;
}

}  // namespace rocprofsys::avail
