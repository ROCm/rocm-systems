// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "avail/records.hpp"

#include "backends/procfs/backend.hpp"
#include "backends/procfs/metric_tokens.hpp"

#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace rocprofsys::avail
{

struct cpu_listing_result
{
    std::vector<device_record> devices;
    std::vector<metric_record> metrics;
    std::optional<diagnostic>  issue;
};

namespace inventory
{

using cpu_frequency_map = std::map<std::size_t, float>;
using cpu_jiffies_map   = std::map<std::size_t, backends::procfs::cpu_jiffies>;

struct cpu_metric_support
{
    const cpu_frequency_map&                 frequencies;
    const cpu_jiffies_map&                   jiffies;
    const backends::procfs::rusage_snapshot& rusage;
};

inline void
append_cpu_devices(std::vector<device_record>&                devices,
                   const backends::procfs::socket_topology_t& topology)
{
    for(const auto& [socket_id, cpu_ids] : topology)
    {
        const auto parent_id = "cpu-socket-" + std::to_string(socket_id);
        devices.emplace_back(device_record{
            .id            = parent_id,
            .kind          = device_kind::cpu_socket,
            .name          = "CPU socket " + std::to_string(socket_id),
            .logical_index = socket_id,
        });
        for(const auto cpu_id : cpu_ids)
        {
            devices.emplace_back(device_record{
                .id            = "cpu-" + std::to_string(cpu_id),
                .kind          = device_kind::cpu_core,
                .name          = "CPU " + std::to_string(cpu_id),
                .logical_index = cpu_id,
                .parent_id     = parent_id,
            });
        }
    }
}

[[nodiscard]] inline availability
logical_cpu_metric_status(backends::procfs::cpu::metric_kind kind, std::size_t cpu_id,
                          const cpu_metric_support& support)
{
    if(kind == backends::procfs::cpu::metric_kind::frequency)
    {
        return support.frequencies.contains(cpu_id) ? availability::available
                                                    : availability::unavailable;
    }
    return support.jiffies.contains(cpu_id) ? availability::available
                                            : availability::unavailable;
}

[[nodiscard]] inline availability
process_metric_status(backends::procfs::cpu::metric_kind kind,
                      const cpu_metric_support&          support) noexcept
{
    if(kind == backends::procfs::cpu::metric_kind::peak_rss &&
       support.rusage.peak_rss <= 0)
    {
        return availability::unavailable;
    }
    return availability::available;
}

[[nodiscard]] inline metric_record
make_cpu_metric_record(const backends::procfs::cpu::metric_token& token,
                       availability                               status,
                       std::optional<std::string> device_id = std::nullopt)
{
    return metric_record{
        .name        = std::string{ token.name },
        .description = std::string{ token.description },
        .unit        = std::string{ token.unit },
        .source      = source_id::procfs,
        .status      = status,
        .device_id   = std::move(device_id),
    };
}

inline void
append_logical_cpu_metric_record(std::vector<metric_record>&                records,
                                 const backends::procfs::cpu::metric_token& token,
                                 std::size_t cpu_id, const cpu_metric_support& support)
{
    records.emplace_back(make_cpu_metric_record(
        token, logical_cpu_metric_status(token.kind, cpu_id, support),
        "cpu-" + std::to_string(cpu_id)));
}

inline void
append_logical_cpu_metric(std::vector<metric_record>&                records,
                          const backends::procfs::cpu::metric_token& token,
                          const backends::procfs::socket_topology_t& topology,
                          const cpu_metric_support&                  support)
{
    for(const auto& topology_entry : topology)
    {
        for(const auto cpu_id : topology_entry.second)
        {
            append_logical_cpu_metric_record(records, token, cpu_id, support);
        }
    }
}

inline void
append_process_metric(std::vector<metric_record>&                records,
                      const backends::procfs::cpu::metric_token& token,
                      const cpu_metric_support&                  support)
{
    records.emplace_back(
        make_cpu_metric_record(token, process_metric_status(token.kind, support)));
}

inline void
append_cpu_metrics(std::vector<metric_record>&                records,
                   const backends::procfs::socket_topology_t& topology,
                   const cpu_metric_support&                  support)
{
    for(const auto& token : backends::procfs::cpu::k_metric_tokens)
    {
        if(token.scope == backends::procfs::cpu::metric_scope::process)
        {
            append_process_metric(records, token, support);
            continue;
        }
        append_logical_cpu_metric(records, token, topology, support);
    }
}

template <typename Backend>
[[nodiscard]] cpu_listing_result
cpu_inventory(Backend& backend, bool include_devices, bool include_metrics)
{
    cpu_listing_result result;
    const auto&        topology = backend.get_socket_topology();
    if(include_devices)
    {
        append_cpu_devices(result.devices, topology);
    }
    if(include_metrics)
    {
        const auto frequencies = backend.read_cpu_frequencies();
        const auto jiffies     = backend.read_proc_stat();
        const auto rusage      = backend.read_rusage();
        append_cpu_metrics(result.metrics, topology,
                           cpu_metric_support{ .frequencies = frequencies,
                                               .jiffies     = jiffies,
                                               .rusage      = rusage });
    }
    return result;
}

}  // namespace inventory

[[nodiscard]] cpu_listing_result
query_cpu_inventory(bool include_devices, bool include_metrics);

}  // namespace rocprofsys::avail
