// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include "pc_sampling_collector.h"
#include "synchronized.hpp"

#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>

namespace rocm_compute
{
class sdk_callbacks_t;

enum class IterationMultiplexingMode
{
    Disabled,
    Simple,
    Kernel,
    Launch
};

struct kernel_dispatch_info_t
{
    uint64_t           kernel_id;
    uint64_t           queue_id;
    rocprofiler_dim3_t workgroup_size;
    rocprofiler_dim3_t grid_size;
    uint32_t           LDS_memory_size;

    bool operator<(const kernel_dispatch_info_t other) const
    {
        return std::tie(kernel_id,
                        queue_id,
                        workgroup_size.x,
                        workgroup_size.y,
                        workgroup_size.z,
                        grid_size.x,
                        grid_size.y,
                        grid_size.z,
                        LDS_memory_size) < std::tie(other.kernel_id,
                                                    other.queue_id,
                                                    other.workgroup_size.x,
                                                    other.workgroup_size.y,
                                                    other.workgroup_size.z,
                                                    other.grid_size.x,
                                                    other.grid_size.y,
                                                    other.grid_size.z,
                                                    other.LDS_memory_size);
    }
};

struct iteration_multiplexing_dispatch_record_t
{
    std::size_t                                   config;
    std::map<uint64_t, std::size_t>               kernel_id_to_profile_index;
    std::map<kernel_dispatch_info_t, std::size_t> kernel_params_to_profile_index;
};

struct counter_info_record_t
{
    uint64_t    dispatch_id     = 0;
    uint64_t    agent_id        = 0;
    uint64_t    kernel_id       = 0;
    uint32_t    LDS_memory_size = 0;
    uint64_t    counter_id      = 0;
    std::string counter_name;
    double      counter_value = 0.;
};

struct tool_data_t
{
    IterationMultiplexingMode iteration_multiplexing_mode{IterationMultiplexingMode::Disabled};
    PcSamplingMode                             pc_sampling_mode{PcSamplingMode::Disabled};
    std::mutex                                 mut{};
    std::string                                counters_output_filename{};
    std::string                                code_obj_output_filename{};
    std::unordered_map<uint64_t, std::string>  counter_id_name_map{};
    std::string                                requested_counters{};
    std::string                                kernel_filter_include_regex{};
    std::vector<std::pair<uint64_t, uint64_t>> kernel_filter_ranges{};
    std::vector<counter_info_record_t>         counter_records;
    std::set<uint64_t>                         target_kernel_ids{};

    synchronized_t<pc_sampling_collector_t::ptr> pc_sampling_collector;
    std::shared_ptr<sdk_callbacks_t>           sdk_callbacks{};
};
}  // namespace rocm_compute
