// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "avail/catalog.hpp"

#include "avail/devices.hpp"
#include "avail/gpu_counters.hpp"
#include "avail/traces.hpp"

#include <iterator>
#include <utility>

namespace rocprofsys::avail
{
catalog_snapshot
query_catalog(const query_request& request)
{
    auto snapshot = catalog_snapshot{};

    // Counters are enumerated per agent, so they imply the device query.
    // Tracing domains use the SDK name tables and do not need GPU inventory.
    const bool needs_devices = request.devices || request.gpu_counters;
    if(needs_devices)
    {
        auto devices             = query_devices();
        snapshot.devices         = std::move(devices.devices);
        snapshot.diagnostics     = std::move(devices.diagnostics);
        snapshot.devices_queried = true;
    }

    if(request.gpu_counters)
    {
        snapshot.counters_queried = true;
        if(!snapshot.devices.empty())
        {
            auto counters           = query_gpu_counters(snapshot.devices);
            snapshot.counter_groups = std::move(counters.devices);
            snapshot.diagnostics.insert(
                snapshot.diagnostics.end(),
                std::make_move_iterator(counters.diagnostics.begin()),
                std::make_move_iterator(counters.diagnostics.end()));
        }
    }

    if(request.traces)
    {
        auto traces             = query_traces();
        snapshot.traces         = std::move(traces.traces);
        snapshot.traces_queried = true;
        snapshot.diagnostics.insert(snapshot.diagnostics.end(),
                                    std::make_move_iterator(traces.diagnostics.begin()),
                                    std::make_move_iterator(traces.diagnostics.end()));
    }

    return snapshot;
}
}  // namespace rocprofsys::avail
