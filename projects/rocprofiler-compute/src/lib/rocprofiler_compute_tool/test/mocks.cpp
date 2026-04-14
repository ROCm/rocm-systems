// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "mocks.h"

const char* MockInputParameters::get_output_path()
{
    return m_output_path.c_str();
}

const char* MockInputParameters::get_requested_counters()
{
    return m_requested_counters.c_str();
}

const char* MockInputParameters::get_iteration_multiplexing_mode()
{
    return m_iteration_multiplexing_mode.c_str();
}

const char* MockInputParameters::get_kernel_filter_include_regex()
{
    return m_kernel_filter_include_regex.c_str();
}

const char* MockInputParameters::get_kernel_filter_range()
{
    return m_kernel_filter_range.c_str();
}

void MockInputParameters::set_output_path(const std::string& output_path)
{
    m_output_path = output_path;
}

void MockInputParameters::set_requested_counters(const std::string& counters)
{
    m_requested_counters = counters;
}

void MockInputParameters::set_iteration_multiplexing_mode(const std::string& mode)
{
    m_iteration_multiplexing_mode = mode;
}

void MockInputParameters::set_kernel_filter_include_regex(const std::string& regex)
{
    m_kernel_filter_include_regex = regex;
}

void MockInputParameters::set_kernel_filter_range(const std::string& range)
{
    m_kernel_filter_range = range;
}

/////////////////////////////////////////////////////////////////////////
// MockSdkWrapper
void MockSdkWrapper::create_context(rocprofiler_context_id_t* context_id)
{
    m_created_contexts.push_back(*context_id);
}

void MockSdkWrapper::configure_callback_dispatch_counting_service(
    rocprofiler_context_id_t                   context_id,
    rocprofiler_dispatch_counting_service_cb_t dispatch_callback,
    void*                                      dispatch_callback_args,
    rocprofiler_dispatch_counting_record_cb_t  record_callback,
    void*                                      record_callback_args)
{
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
    m_started_contexts.push_back(context_id);
}

void MockSdkWrapper::iterate_agent_supported_counters(rocprofiler_agent_id_t              agent_id,
                                                      rocprofiler_available_counters_cb_t cb,
                                                      void*                               user_data)
{
}

void MockSdkWrapper::query_counter_info(rocprofiler_counter_id_t              counter_id,
                                        rocprofiler_counter_info_version_id_t version,
                                        void*                                 info)
{
}

void MockSdkWrapper::create_counter_config(rocprofiler_agent_id_t           agent_id,
                                           rocprofiler_counter_id_t*        counters_list,
                                           size_t                           counters_count,
                                           rocprofiler_counter_config_id_t* config_id)
{
}

void MockSdkWrapper::query_record_counter_id(rocprofiler_counter_instance_id_t id,
                                             rocprofiler_counter_id_t*         counter_id)
{
}

const std::vector<rocprofiler_context_id_t>& MockSdkWrapper::get_created_contexts() const
{
    return m_created_contexts;
}

const std::vector<rocprofiler_context_id_t>& MockSdkWrapper::get_started_contexts() const
{
    return m_started_contexts;
}
