// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "mocks.h"

#include "gsl_assert.h"
using namespace rocm_compute;

const char* mock_env_parameters_t::get_output_path()
{
    return m_output_path.c_str();
}

const char* mock_env_parameters_t::get_requested_counters()
{
    return m_requested_counters.c_str();
}

const char* mock_env_parameters_t::get_iteration_multiplexing_mode()
{
    return m_iteration_multiplexing_mode.c_str();
}

const char* mock_env_parameters_t::get_kernel_filter_include_regex()
{
    return m_kernel_filter_include_regex.c_str();
}

const char* mock_env_parameters_t::get_kernel_filter_range()
{
    return m_kernel_filter_range.c_str();
}

const char* mock_env_parameters_t::get_pc_sampling_mode() const
{
    return m_pc_sampling_mode.c_str();
}

void mock_env_parameters_t::set_output_path(const std::string& output_path)
{
    m_output_path = output_path;
}

void mock_env_parameters_t::set_requested_counters(const std::string& counters)
{
    m_requested_counters = counters;
}

void mock_env_parameters_t::set_iteration_multiplexing_mode(const std::string& mode)
{
    m_iteration_multiplexing_mode = mode;
}

void mock_env_parameters_t::set_kernel_filter_include_regex(const std::string& regex)
{
    m_kernel_filter_include_regex = regex;
}

void mock_env_parameters_t::set_kernel_filter_range(const std::string& range)
{
    m_kernel_filter_range = range;
}

void mock_env_parameters_t::set_pc_sampling_mode(const std::string& mode)
{
    m_pc_sampling_mode = mode;
}

/////////////////////////////////////////////////////////////////////////
// mock_sdk_wrapper_t
void mock_sdk_wrapper_t::create_context(rocprofiler_context_id_t* context_id)
{
    m_created_contexts.push_back(context_id->handle);
}

void mock_sdk_wrapper_t::configure_callback_dispatch_counting_service(
    rocprofiler_context_id_t                   context_id,
    rocprofiler_dispatch_counting_service_cb_t dispatch_callback,
    void*                                      dispatch_callback_args,
    rocprofiler_dispatch_counting_record_cb_t  record_callback,
    void*                                      record_callback_args)
{
    m_dispatch_counting_service_info.push_back(
        dispatch_counting_service_info_t{context_id.handle,
                                         reinterpret_cast<void*>(dispatch_callback),
                                         dispatch_callback_args,
                                         reinterpret_cast<void*>(record_callback),
                                         record_callback_args});
}

void mock_sdk_wrapper_t::configure_callback_tracing_service(rocprofiler_context_id_t context_id,
                                                            rocprofiler_callback_tracing_kind_t kind,
                                                            const rocprofiler_tracing_operation_t* operations,
                                                            size_t operations_count,
                                                            rocprofiler_callback_tracing_cb_t callback,
                                                            void* callback_args)
{
}

void mock_sdk_wrapper_t::start_context(rocprofiler_context_id_t context_id)
{
    m_started_contexts.push_back(context_id.handle);
}

void mock_sdk_wrapper_t::iterate_agent_supported_counters(rocprofiler_agent_id_t agent_id,
                                                          rocprofiler_available_counters_cb_t cb,
                                                          void* user_data)
{
    auto counters = get_counters();
    cb(agent_id, counters.data(), counters.size(), user_data);
}

std::vector<rocprofiler_counter_id_t> mock_sdk_wrapper_t::get_counters() const
{
    std::vector<rocprofiler_counter_id_t> counters;
    for (uint32_t i = 0; i < m_counter_names.size(); ++i)
    {
        rocprofiler_counter_id_t counter_id{i};
        counters.push_back(counter_id);
    }
    return counters;
}

void mock_sdk_wrapper_t::query_counter_info(rocprofiler_counter_id_t              counter_id,
                                            rocprofiler_counter_info_version_id_t version,
                                            void*                                 info)
{
    Expects(counter_id.handle < m_counter_names.size());
    Expects(info);

    const auto counter_info = static_cast<rocprofiler_counter_info_v0_t*>(info);
    counter_info->id        = counter_id;
    counter_info->name      = m_counter_names[counter_id.handle].c_str();
}

void mock_sdk_wrapper_t::create_counter_config(rocprofiler_agent_id_t           agent_id,
                                               rocprofiler_counter_id_t*        counters_list,
                                               size_t                           counters_count,
                                               rocprofiler_counter_config_id_t* config_id)
{
    Expects(counters_count <= m_counter_names.size());
    create_counter_config_info_t info;
    for (size_t i = 0; i < counters_count; ++i)
    {
        Expects(counters_list[i].handle < m_counter_names.size());
        info.counter_names.push_back(m_counter_names[counters_list[i].handle]);
    }
    m_create_counter_config_info.push_back(info);
    config_id->handle = m_create_counter_config_info.size() - 1;
}

void mock_sdk_wrapper_t::query_record_counter_id(rocprofiler_counter_instance_id_t id,
                                                 rocprofiler_counter_id_t*         counter_id)
{
    m_query_counter_record_info.push_back({id, id});
    counter_id->handle = id;
}

void mock_sdk_wrapper_t::set_available_counters(const std::vector<std::string>& counter_names)
{
    m_counter_names = counter_names;
}

const std::vector<uint64_t>& mock_sdk_wrapper_t::get_created_contexts() const
{
    return m_created_contexts;
}

const std::vector<uint64_t>& mock_sdk_wrapper_t::get_started_contexts() const
{
    return m_started_contexts;
}

const std::vector<mock_sdk_wrapper_t::dispatch_counting_service_info_t>&
    mock_sdk_wrapper_t::get_dispatch_counting_service_info() const
{
    return m_dispatch_counting_service_info;
}

const std::vector<mock_sdk_wrapper_t::create_counter_config_info_t>&
    mock_sdk_wrapper_t::get_create_counter_config_info() const
{
    return m_create_counter_config_info;
}

const std::vector<mock_sdk_wrapper_t::query_counter_record_info_t>&
    mock_sdk_wrapper_t::get_query_counter_record_info() const
{
    return m_query_counter_record_info;
}

/////////////////////////////////////////////////////////////////////////
// mock_counters_writer_t
void mock_counters_writer_t::write_counters(const std::string& output_file, const std::vector<counter_info_record_t>& records)
{
    write_counters_info_t args;
    for (const auto& counter : records)
    {
        args.counter_ids.push_back(counter.counter_id);
        args.kernel_id.push_back(counter.kernel_id);
    }
    m_write_counters_args.push_back(std::move(args));
}

const std::vector<mock_counters_writer_t::write_counters_info_t>& mock_counters_writer_t::get_write_counters_info() const
{
    return m_write_counters_args;
}

/////////////////////////////////////////////////////////////////////////
// mock_sdk_callbacks_t
void mock_sdk_callbacks_t::dispatch_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                                             rocprofiler_counter_config_id_t* config,
                                             tool_data_t& /*tool_data*/)
{
    m_dispatch_callback_info.push_back({dispatch_data, config});
}

void mock_sdk_callbacks_t::record_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                                           rocprofiler_counter_record_t* record_data,
                                           size_t                        record_count,
                                           tool_data_t& /*tool_data*/)
{
    m_record_callback_info.push_back({dispatch_data, record_data, record_count});
}

void mock_sdk_callbacks_t::tool_tracing_callback(rocprofiler_callback_tracing_record_t record,
                                                 tool_data_t& /*tool_data*/)
{
    m_tracing_callback_info.push_back({record});
}

const std::vector<mock_sdk_callbacks_t::dispatch_callback_info_t>& mock_sdk_callbacks_t::get_dispatch_callback_info() const
{
    return m_dispatch_callback_info;
}

const std::vector<mock_sdk_callbacks_t::record_callback_info_t>& mock_sdk_callbacks_t::get_record_callback_info() const
{
    return m_record_callback_info;
}

const std::vector<mock_sdk_callbacks_t::tracing_callback_info_t>& mock_sdk_callbacks_t::get_tracing_callback_info() const
{
    return m_tracing_callback_info;
}

void mock_pc_sampling_collector_t::on_code_object_load(
    const rocprofiler_callback_tracing_code_object_load_data_t& info)
{
    m_on_code_object_load_info.push_back(info);
}

void mock_pc_sampling_collector_t::write(code_object_writer_t& writer) {}

const std::vector<rocprofiler_callback_tracing_code_object_load_data_t>&
    mock_pc_sampling_collector_t::get_on_code_object_load_info() const
{
    return m_on_code_object_load_info;
}
