// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "mocks.h"

#include "gsl_assert.h"

#include <utility>

std::string_view MockInputParameters::get_output_path()
{
    if (!m_output_path_set || m_output_path.empty())
        return rocprofiler_compute_tool::EnvInputParameters::kDefaultOutputPath;
    return std::string_view{m_output_path};
}

std::string_view MockInputParameters::get_requested_counters()
{
    if (!m_requested_counters_set || m_requested_counters.empty())
        return rocprofiler_compute_tool::EnvInputParameters::kDefaultRequestedCounters;
    return std::string_view{m_requested_counters};
}

std::string_view MockInputParameters::get_iteration_multiplexing_mode()
{
    if (!m_iteration_multiplexing_mode_set || m_iteration_multiplexing_mode.empty())
        return rocprofiler_compute_tool::EnvInputParameters::kDefaultIterationMultiplexingMode;
    return std::string_view{m_iteration_multiplexing_mode};
}

std::string_view MockInputParameters::get_kernel_filter_include_regex()
{
    if (!m_kernel_filter_include_regex_set || m_kernel_filter_include_regex.empty())
        return rocprofiler_compute_tool::EnvInputParameters::kDefaultKernelFilterIncludeRegex;
    return std::string_view{m_kernel_filter_include_regex};
}

std::string_view MockInputParameters::get_kernel_filter_range()
{
    if (!m_kernel_filter_range_set || m_kernel_filter_range.empty())
        return rocprofiler_compute_tool::EnvInputParameters::kDefaultKernelFilterRange;
    return std::string_view{m_kernel_filter_range};
}

void MockInputParameters::set_output_path(const std::string& output_path)
{
    m_output_path     = output_path;
    m_output_path_set = true;
}

void MockInputParameters::set_requested_counters(const std::string& counters)
{
    m_requested_counters     = counters;
    m_requested_counters_set = true;
}

void MockInputParameters::set_iteration_multiplexing_mode(const std::string& mode)
{
    m_iteration_multiplexing_mode     = mode;
    m_iteration_multiplexing_mode_set = true;
}

void MockInputParameters::set_kernel_filter_include_regex(const std::string& regex)
{
    m_kernel_filter_include_regex     = regex;
    m_kernel_filter_include_regex_set = true;
}

void MockInputParameters::set_kernel_filter_range(const std::string& range)
{
    m_kernel_filter_range     = range;
    m_kernel_filter_range_set = true;
}

void MockInputParameters::unset_output_path()
{
    m_output_path_set = false;
}

void MockInputParameters::unset_requested_counters()
{
    m_requested_counters_set = false;
}

void MockInputParameters::unset_iteration_multiplexing_mode()
{
    m_iteration_multiplexing_mode_set = false;
}

void MockInputParameters::unset_kernel_filter_include_regex()
{
    m_kernel_filter_include_regex_set = false;
}

void MockInputParameters::unset_kernel_filter_range()
{
    m_kernel_filter_range_set = false;
}

/////////////////////////////////////////////////////////////////////////
// MockSdkWrapper
void MockSdkWrapper::create_context(rocprofiler_context_id_t* context_id)
{
    m_created_contexts.push_back(context_id->handle);
}

void MockSdkWrapper::configure_callback_dispatch_counting_service(
    rocprofiler_context_id_t                   context_id,
    rocprofiler_dispatch_counting_service_cb_t dispatch_callback,
    void*                                      dispatch_callback_args,
    rocprofiler_dispatch_counting_record_cb_t  record_callback,
    void*                                      record_callback_args)
{
    m_dispatch_counting_service_info.push_back(
        dispatch_counting_service_info{context_id.handle,
                                       reinterpret_cast<void*>(dispatch_callback),
                                       dispatch_callback_args,
                                       reinterpret_cast<void*>(record_callback),
                                       record_callback_args});
}

void MockSdkWrapper::configure_callback_tracing_service(rocprofiler_context_id_t context_id,
                                                        rocprofiler_callback_tracing_kind_t kind,
                                                        const rocprofiler_tracing_operation_t* operations,
                                                        size_t operations_count,
                                                        rocprofiler_callback_tracing_cb_t callback,
                                                        void* callback_args)
{
}

void MockSdkWrapper::start_context(rocprofiler_context_id_t context_id)
{
    m_started_contexts.push_back(context_id.handle);
}

void MockSdkWrapper::iterate_agent_supported_counters(rocprofiler_agent_id_t              agent_id,
                                                      rocprofiler_available_counters_cb_t cb,
                                                      void*                               user_data)
{
    auto counters = get_counters();
    cb(agent_id, counters.data(), counters.size(), user_data);
}

std::vector<rocprofiler_counter_id_t> MockSdkWrapper::get_counters() const
{
    std::vector<rocprofiler_counter_id_t> counters;
    for (uint32_t i = 0; i < m_counter_names.size(); ++i)
    {
        rocprofiler_counter_id_t counter_id{i};
        counters.push_back(counter_id);
    }
    return counters;
}

void MockSdkWrapper::query_counter_info(rocprofiler_counter_id_t              counter_id,
                                        rocprofiler_counter_info_version_id_t version,
                                        void*                                 info)
{
    Expects(counter_id.handle < m_counter_names.size());
    Expects(info);

    const auto counter_info = static_cast<rocprofiler_counter_info_v0_t*>(info);
    counter_info->id        = counter_id;
    counter_info->name      = m_counter_names[counter_id.handle].c_str();
}

void MockSdkWrapper::create_counter_config(rocprofiler_agent_id_t           agent_id,
                                           rocprofiler_counter_id_t*        counters_list,
                                           size_t                           counters_count,
                                           rocprofiler_counter_config_id_t* config_id)
{
    Expects(counters_count <= m_counter_names.size());
    create_counter_config_info info;
    for (size_t i = 0; i < counters_count; ++i)
    {
        Expects(counters_list[i].handle < m_counter_names.size());
        info.counter_names.push_back(m_counter_names[counters_list[i].handle]);
    }
    m_create_counter_config_info.push_back(info);
    config_id->handle = m_create_counter_config_info.size() - 1;
}

void MockSdkWrapper::query_record_counter_id(rocprofiler_counter_instance_id_t id,
                                             rocprofiler_counter_id_t*         counter_id)
{
    m_query_counter_record_info.push_back({id, id});
    counter_id->handle = id;
}

void MockSdkWrapper::set_available_counters(const std::vector<std::string>& counter_names)
{
    m_counter_names = counter_names;
}

const std::vector<uint64_t>& MockSdkWrapper::get_created_contexts() const
{
    return m_created_contexts;
}

const std::vector<uint64_t>& MockSdkWrapper::get_started_contexts() const
{
    return m_started_contexts;
}

const std::vector<MockSdkWrapper::dispatch_counting_service_info>&
    MockSdkWrapper::get_dispatch_counting_service_info() const
{
    return m_dispatch_counting_service_info;
}

const std::vector<MockSdkWrapper::create_counter_config_info>& MockSdkWrapper::get_create_counter_config_info() const
{
    return m_create_counter_config_info;
}

const std::vector<MockSdkWrapper::query_counter_record_info>& MockSdkWrapper::get_query_counter_record_info() const
{
    return m_query_counter_record_info;
}

/////////////////////////////////////////////////////////////////////////
// MockCountersWriter
void MockCountersWriter::write_counters(rocprofiler_compute_tool::tool_data_t* tool_data)
{
    write_counters_info args;
    for (const auto& counter : tool_data->counter_records)
    {
        args.counter_ids.push_back(counter.counter_id);
        args.kernel_id.push_back(counter.kernel_id);
    }
    m_write_counters_args.push_back(std::move(args));
}

const std::vector<MockCountersWriter::write_counters_info>& MockCountersWriter::get_write_counters_info() const
{
    return m_write_counters_args;
}
