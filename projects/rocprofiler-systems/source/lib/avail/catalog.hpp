// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "avail/records.hpp"

#include <optional>
#include <string>
#include <vector>

namespace rocprofsys::avail
{

struct query_request
{
    bool                       gpu_devices     = false;
    bool                       cpu_devices     = false;
    bool                       nic_devices     = false;
    bool                       traces          = false;
    std::optional<std::string> operations_for  = std::nullopt;
    bool                       gpu_counters    = false;
    bool                       cpu_counters    = false;
    bool                       cpu_metrics     = false;
    bool                       gpu_metrics     = false;
    bool                       nic_metrics     = false;
    bool                       storage_metrics = false;
};

struct catalog_snapshot
{
    std::vector<device_record>    gpu_devices;
    std::vector<device_record>    cpu_devices;
    std::vector<device_record>    nic_devices;
    std::vector<trace_record>     traces;
    std::vector<operation_record> trace_operations;
    std::vector<counter_record>   gpu_counters;
    std::vector<counter_record>   cpu_counters;
    std::vector<metric_record>    cpu_metrics;
    std::vector<metric_record>    gpu_metrics;
    std::vector<metric_record>    nic_metrics;
    std::vector<metric_record>    storage_metrics;
    std::vector<diagnostic>       diagnostics;
    std::vector<capability_kind>  queried;

    [[nodiscard]] bool was_queried(capability_kind capability) const noexcept;
    [[nodiscard]] bool degraded() const noexcept { return !diagnostics.empty(); }
};

[[nodiscard]] catalog_snapshot
query(const query_request& request);

}  // namespace rocprofsys::avail
