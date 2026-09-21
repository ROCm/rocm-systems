// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#include "pc_sampling_feature.h"
#include "sdk_wrapper.h"

#include <rocprofiler-sdk/rocprofiler.h>

#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocprofiler_compute_tool
{
enum class iteration_multiplexing_mode_t
{
    DISABLED,
    SIMPLE,
    KERNEL,
    LAUNCH
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
    std::mutex                                 mut{};
    std::string                                output_filename{};
    std::unordered_map<uint64_t, std::string>  counter_id_name_map{};
    std::string                                requested_counters{};
    std::string                                kernel_filter_include_regex{};
    std::vector<std::pair<uint64_t, uint64_t>> kernel_filter_ranges{};
    std::vector<counter_info_record_t>         counter_records;
    std::set<uint64_t>                         target_kernel_ids{};
    iteration_multiplexing_mode_t iteration_multiplexing_mode{iteration_multiplexing_mode_t::DISABLED};
    pc_sampling_feature_t pc_sampling{};
};

class SdkCallbacks
{
public:
    virtual ~SdkCallbacks() = default;

    virtual void dispatch_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,

                                   rocprofiler_counter_config_id_t* config,
                                   void*                            callback_data_args) = 0;

    virtual void record_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                                 rocprofiler_counter_record_t*                record_data,
                                 size_t                                       record_count,
                                 void* callback_data_args) = 0;

    virtual void tool_tracing_callback(rocprofiler_callback_tracing_record_t record,
                                       void*                                 callback_data) = 0;
};

class SdkCallbacksImpl : public SdkCallbacks
{
public:
    SdkCallbacksImpl(const std::shared_ptr<SdkWrapper>& sdk_wrapper);

    void dispatch_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                           rocprofiler_counter_config_id_t*             config,
                           void* callback_data_args) override;

    void record_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                         rocprofiler_counter_record_t*                record_data,
                         size_t                                       record_count,
                         void*                                        callback_data_args) override;

    void tool_tracing_callback(rocprofiler_callback_tracing_record_t record, void* callback_data) override;

private:
    static bool is_targeted_dispatch(const tool_data_t* tool, uint64_t kernel_id, uint64_t kernel_iteration);
    void create_counter_collection_profile(
        tool_data_t*           tool,
        rocprofiler_agent_id_t agent_id,
        std::unordered_map<uint64_t, std::vector<rocprofiler_counter_config_id_t>>& profile_cache) const;

    std::shared_ptr<SdkWrapper>            m_sdk_wrapper;
    std::unordered_map<uint64_t, uint64_t> m_kernel_dispatch_count_by_kernel_id{};
    std::shared_mutex                      m_kernel_id_iteration_mutex;
    std::shared_mutex                      m_mutex = {};
    std::unordered_map<uint64_t, std::vector<rocprofiler_counter_config_id_t>> m_profile_cache_per_agent = {};
    std::unordered_map<uint64_t, iteration_multiplexing_dispatch_record_t> m_iteration_multiplexing_per_agent = {};

    static std::string truncate_name(std::string_view name);
    static std::string cxa_demangle(const std::string& mangled_name, int* status);
    static std::vector<std::string> split_by_regex(const std::string& s, const std::string& regex_pattern);
};
}  // namespace rocprofiler_compute_tool
