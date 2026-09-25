// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "avail/records.hpp"

#include <string_view>

namespace rocprofsys::avail
{

std::string_view
to_string(source_id value) noexcept
{
    switch(value)
    {
        case source_id::catalog: return "catalog";
        case source_id::rocprofiler_sdk: return "rocprofiler-sdk";
        case source_id::amd_smi: return "amd-smi";
        case source_id::procfs: return "procfs";
        case source_id::papi: return "papi";
        case source_id::storage: return "storage";
    }
    return "unknown";
}

std::string_view
to_string(capability_kind value) noexcept
{
    switch(value)
    {
        case capability_kind::gpu_devices: return "gpu-devices";
        case capability_kind::cpu_devices: return "cpu-devices";
        case capability_kind::nic_devices: return "nic-devices";
        case capability_kind::traces: return "traces";
        case capability_kind::trace_operations: return "trace-operations";
        case capability_kind::gpu_counters: return "gpu-counters";
        case capability_kind::cpu_counters: return "cpu-counters";
        case capability_kind::cpu_metrics: return "cpu-metrics";
        case capability_kind::gpu_metrics: return "gpu-metrics";
        case capability_kind::nic_metrics: return "nic-metrics";
        case capability_kind::storage_metrics: return "storage-metrics";
    }
    return "unknown";
}

}  // namespace rocprofsys::avail
