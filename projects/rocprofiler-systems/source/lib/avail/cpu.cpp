// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "avail/cpu.hpp"
#include "avail/records.hpp"

#include "backends/procfs/backend.hpp"

#include <cstddef>
#include <exception>
#include <string>
#include <unistd.h>
#include <utility>

namespace rocprofsys::avail
{
namespace
{

[[nodiscard]] cpu_listing_result
cpu_query_failure(capability_kind capability, std::string message)
{
    return cpu_listing_result{
        .issue =
            diagnostic{
                .capability = capability,
                .source     = source_id::procfs,
                .message    = std::move(message),
            },
    };
}

}  // namespace

cpu_listing_result
query_cpu_inventory(bool include_devices, bool include_metrics)
{
    const auto online_cpu_count = sysconf(_SC_NPROCESSORS_ONLN);
    if(online_cpu_count <= 0)
    {
        const auto capability =
            include_devices ? capability_kind::cpu_devices : capability_kind::cpu_metrics;
        return cpu_query_failure(capability, "could not determine the online CPU count");
    }

    try
    {
        backends::procfs::backend backend{ static_cast<std::size_t>(online_cpu_count) };
        return inventory::cpu_inventory(backend, include_devices, include_metrics);
    } catch(const std::exception& err)
    {
        const auto capability =
            include_devices ? capability_kind::cpu_devices : capability_kind::cpu_metrics;
        return cpu_query_failure(capability, err.what());
    }
}

}  // namespace rocprofsys::avail
