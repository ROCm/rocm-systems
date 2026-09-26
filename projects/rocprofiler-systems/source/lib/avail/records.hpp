// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace rocprofsys::avail
{

enum class source_id : std::uint8_t
{
    catalog,
    rocprofiler_sdk,
    amd_smi,
    procfs,
    papi,
    storage,
};

enum class capability_kind : std::uint8_t
{
    gpu_devices,
    cpu_devices,
    nic_devices,
    traces,
    trace_operations,
    gpu_counters,
    cpu_counters,
    cpu_metrics,
    gpu_metrics,
    nic_metrics,
    storage_metrics,
};

enum class device_kind : std::uint8_t
{
    gpu,
    cpu_socket,
    cpu_core,
    nic,
    nic_port,
};

enum class availability : std::uint8_t
{
    unknown,
    available,
    unavailable,
};

[[nodiscard]] std::string_view
to_string(source_id value) noexcept;

[[nodiscard]] std::string_view
to_string(capability_kind value) noexcept;

struct diagnostic
{
    capability_kind capability = capability_kind::gpu_devices;
    source_id       source     = source_id::catalog;
    std::string     message;
};

struct device_record
{
    std::string                id;
    device_kind                kind = device_kind::gpu;
    std::string                name;
    std::optional<std::size_t> logical_index = std::nullopt;
    std::optional<std::string> vendor        = std::nullopt;
    std::optional<std::string> product       = std::nullopt;
    std::optional<std::string> pci_bdf       = std::nullopt;
    std::optional<std::string> parent_id     = std::nullopt;
    std::optional<std::size_t> numa_node     = std::nullopt;
};

struct counter_dimension
{
    std::string   name;
    std::uint64_t extent = 0;
};

struct counter_record
{
    std::string                    name;
    std::string                    description;
    source_id                      source     = source_id::catalog;
    availability                   status     = availability::unknown;
    std::optional<std::uint64_t>   id         = std::nullopt;
    std::optional<std::string>     device_id  = std::nullopt;
    std::optional<std::string>     block      = std::nullopt;
    std::optional<std::string>     expression = std::nullopt;
    std::optional<std::string>     unit       = std::nullopt;
    std::vector<std::string>       aliases;
    std::vector<counter_dimension> dimensions;
};

struct trace_record
{
    std::string              name;
    std::string              description;
    std::vector<std::string> aliases;
};

struct operation_record
{
    std::string trace_name;
    std::string name;
    std::string description;
};

struct metric_record
{
    std::string                name;
    std::string                description;
    std::string                unit;
    source_id                  source    = source_id::catalog;
    availability               status    = availability::unknown;
    std::optional<std::string> device_id = std::nullopt;
};

}  // namespace rocprofsys::avail
