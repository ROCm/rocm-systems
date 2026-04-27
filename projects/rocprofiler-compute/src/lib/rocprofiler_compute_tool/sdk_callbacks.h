// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#include "sdk_wrapper.h"
#include "tool_data.h"
#include <rocprofiler-sdk/rocprofiler.h>

#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocprofiler_compute_tool
{

class sdk_callbacks_t
{
public:
    virtual ~sdk_callbacks_t() = default;

    virtual void dispatch_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,

                                   rocprofiler_counter_config_id_t* config,
                                   tool_data_t&                     tool_data) = 0;

    virtual void record_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                                 rocprofiler_counter_record_t*                record_data,
                                 size_t                                       record_count,
                                 tool_data_t&                                 tool_data) = 0;

    virtual void tool_tracing_callback(rocprofiler_callback_tracing_record_t record,
                                       tool_data_t&                          tool_data) = 0;

    virtual void code_object_tracing_callback(rocprofiler_callback_tracing_record_t record,
                                              tool_data_t&                          tool_data) = 0;
};

class sdk_callbacks_impl_t : public sdk_callbacks_t
{
public:
    sdk_callbacks_impl_t(const std::shared_ptr<sdk_wrapper_t>& sdk_wrapper);

    void dispatch_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                           rocprofiler_counter_config_id_t*             config,
                           tool_data_t&                                 tool_data) override;

    void record_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                         rocprofiler_counter_record_t*                record_data,
                         size_t                                       record_count,
                         tool_data_t&                                 tool_data) override;

    void tool_tracing_callback(rocprofiler_callback_tracing_record_t record,
                               tool_data_t&                          tool_data) override;

    void code_object_tracing_callback(rocprofiler_callback_tracing_record_t record,
                                      tool_data_t&                          tool_data) override;

private:
    static bool is_targeted_dispatch(const tool_data_t& tool, uint64_t kernel_id, uint64_t kernel_iteration);
    void create_counter_collection_profile(
        tool_data_t&           tool,
        rocprofiler_agent_id_t agent_id,
        std::unordered_map<uint64_t, std::vector<rocprofiler_counter_config_id_t>>& profile_cache) const;

    std::shared_ptr<sdk_wrapper_t>         m_sdk_wrapper;
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
