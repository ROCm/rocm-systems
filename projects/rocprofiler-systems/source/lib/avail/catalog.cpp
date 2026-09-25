// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "avail/catalog.hpp"
#include "avail/devices.hpp"
#include "avail/gpu_counters.hpp"
#include "avail/gpu_metrics.hpp"
#include "avail/nic.hpp"
#include "avail/records.hpp"
#include "avail/traces.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace rocprofsys::avail
{
namespace
{

struct stub_spec
{
    bool query_request::*flag;
    capability_kind      capability;
    source_id            source;
};

constexpr auto k_boolean_stubs = std::array{
    stub_spec{ .flag       = &query_request::cpu_devices,
               .capability = capability_kind::cpu_devices,
               .source     = source_id::procfs },
    stub_spec{ .flag       = &query_request::cpu_counters,
               .capability = capability_kind::cpu_counters,
               .source     = source_id::papi },
    stub_spec{ .flag       = &query_request::cpu_metrics,
               .capability = capability_kind::cpu_metrics,
               .source     = source_id::procfs },
    stub_spec{ .flag       = &query_request::storage_metrics,
               .capability = capability_kind::storage_metrics,
               .source     = source_id::storage },
};

void
append_stub(catalog_snapshot& snapshot, capability_kind capability, source_id source)
{
    snapshot.queried.emplace_back(capability);
    snapshot.diagnostics.emplace_back(diagnostic{
        .capability = capability,
        .source     = source,
        .message    = std::string{ k_not_implemented_message },
    });
}

void
append_listing_diagnostic(catalog_snapshot& snapshot, std::optional<diagnostic> entry)
{
    if(entry)
    {
        snapshot.diagnostics.push_back(std::move(*entry));
    }
}

std::string
join_csv(const std::vector<std::string>& names)
{
    std::string joined;
    for(std::size_t idx = 0; idx < names.size(); ++idx)
    {
        if(idx != 0)
        {
            joined.push_back(',');
        }
        joined += names[idx];
    }
    return joined;
}

}  // namespace

bool
catalog_snapshot::was_queried(capability_kind capability) const noexcept
{
    return std::ranges::find(queried, capability) != queried.end();
}

std::string
catalog_snapshot::default_traces_csv() const
{
    return join_csv(default_traces);
}

std::string
catalog_snapshot::available_traces_csv() const
{
    std::vector<std::string> names;
    names.reserve(traces.size());
    for(const auto& entry : traces)
    {
        names.push_back(entry.name);
    }
    return join_csv(names);
}

catalog_snapshot
query(const query_request& request)
{
    catalog_snapshot result;

    for(const auto& stub : k_boolean_stubs)
    {
        if(request.*(stub.flag))
        {
            append_stub(result, stub.capability, stub.source);
        }
    }

    if(request.gpu_devices)
    {
        result.queried.emplace_back(capability_kind::gpu_devices);
        auto listed        = query_gpu_devices();
        result.gpu_devices = std::move(listed.records);
        append_listing_diagnostic(result, std::move(listed.issue));
    }

    if(request.gpu_counters)
    {
        result.queried.emplace_back(capability_kind::gpu_counters);
        auto listed         = query_gpu_counters();
        result.gpu_counters = std::move(listed.records);
        append_listing_diagnostic(result, std::move(listed.issue));
    }

    if(request.gpu_metrics)
    {
        result.queried.emplace_back(capability_kind::gpu_metrics);
        auto listed        = query_gpu_metrics();
        result.gpu_metrics = std::move(listed.records);
        append_listing_diagnostic(result, std::move(listed.issue));
    }

    if(request.nic_devices || request.nic_metrics)
    {
        if(request.nic_devices)
        {
            result.queried.emplace_back(capability_kind::nic_devices);
        }
        if(request.nic_metrics)
        {
            result.queried.emplace_back(capability_kind::nic_metrics);
        }
        auto listed = query_nic_inventory(request.nic_devices, request.nic_metrics);
        result.nic_devices = std::move(listed.devices);
        result.nic_metrics = std::move(listed.metrics);
        append_listing_diagnostic(result, std::move(listed.issue));
    }

    if(request.traces)
    {
        result.queried.emplace_back(capability_kind::traces);
        auto listed           = query_traces();
        result.traces         = std::move(listed.records);
        result.default_traces = std::move(listed.defaults);
        append_listing_diagnostic(result, std::move(listed.issue));
    }

    if(request.operations_for)
    {
        result.queried.emplace_back(capability_kind::trace_operations);
        auto listed             = query_operations(*request.operations_for);
        result.trace_operations = std::move(listed.records);
        append_listing_diagnostic(result, std::move(listed.issue));
    }

    return result;
}

}  // namespace rocprofsys::avail
