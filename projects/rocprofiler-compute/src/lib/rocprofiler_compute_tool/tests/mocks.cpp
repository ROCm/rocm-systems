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

std::string_view MockInputParameters::get_pc_sampling_method()
{
    return std::string_view{m_pc_sampling_method};
}

std::string_view MockInputParameters::get_pc_sampling_beta_enabled()
{
    return std::string_view{m_pc_sampling_beta_enabled};
}

std::string_view MockInputParameters::get_pc_sampling_interval()
{
    return std::string_view{m_pc_sampling_interval};
}

void MockInputParameters::set_pc_sampling_interval(const std::string& interval)
{
    m_pc_sampling_interval = interval;
}

void MockInputParameters::set_pc_sampling_method(const std::string& method)
{
    m_pc_sampling_method = method;
}

void MockInputParameters::set_pc_sampling_beta_enabled(const std::string& value)
{
    m_pc_sampling_beta_enabled = value;
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

void MockSdkWrapper::at_intercept_table_registration_hsa(rocprofiler_intercept_library_cb_t callback,
                                                         void* user_data)
{
    m_hsa_intercept_registration_info.push_back({callback, user_data});
}

const std::vector<MockSdkWrapper::hsa_intercept_registration_info>&
    MockSdkWrapper::get_hsa_intercept_registration_info() const
{
    return m_hsa_intercept_registration_info;
}

const std::vector<MockSdkWrapper::query_counter_record_info>& MockSdkWrapper::get_query_counter_record_info() const
{
    return m_query_counter_record_info;
}

void MockSdkWrapper::query_available_gpu_agents(std::vector<rocprofiler_agent_id_t>& out_gpu_agents)
{
    out_gpu_agents = m_gpu_agents;
}

void MockSdkWrapper::query_pc_sampling_configs(rocprofiler_agent_id_t /*agent_id*/,
                                               rocprofiler_available_pc_sampling_configurations_cb_t cb,
                                               void* user_data)
{
    if (!m_pc_sampling_config_set)
        return;

    rocprofiler_pc_sampling_configuration_t config{};
    config.size         = sizeof(rocprofiler_pc_sampling_configuration_t);
    config.method       = m_pc_sampling_config.method;
    config.unit         = m_pc_sampling_config.unit;
    config.min_interval = m_pc_sampling_config.min_interval;
    config.max_interval = m_pc_sampling_config.max_interval;
    config.flags        = 0;
    cb(&config, 1, user_data);
}

void MockSdkWrapper::create_buffer(rocprofiler_context_id_t context_id,
                                   size_t                   size,
                                   size_t                   watermark,
                                   rocprofiler_buffer_policy_t /*policy*/,
                                   rocprofiler_buffer_tracing_cb_t /*callback*/,
                                   void* /*callback_data*/,
                                   rocprofiler_buffer_id_t* buffer_id)
{
    const uint64_t assigned = m_next_buffer_id++;
    if (buffer_id != nullptr)
        buffer_id->handle = assigned;
    m_create_buffer_info.push_back(create_buffer_info{context_id.handle, size, watermark, assigned});
}

rocprofiler_status_t MockSdkWrapper::configure_pc_sampling_service(rocprofiler_context_id_t /*context_id*/,
                                                                   rocprofiler_agent_id_t agent_id,
                                                                   rocprofiler_pc_sampling_method_t method,
                                                                   rocprofiler_pc_sampling_unit_t unit,
                                                                   uint64_t interval,
                                                                   rocprofiler_buffer_id_t buffer_id,
                                                                   int /*flags*/)
{
    m_configure_pc_sampling_info.push_back(
        configure_pc_sampling_info{agent_id, method, unit, interval, buffer_id.handle});
    return m_configure_pc_sampling_status;
}

void MockSdkWrapper::flush_buffer(rocprofiler_buffer_id_t buffer_id)
{
    m_flush_buffer_info.push_back(flush_buffer_info{buffer_id.handle});
}

void MockSdkWrapper::configure_buffer_tracing_service(rocprofiler_context_id_t          context_id,
                                                      rocprofiler_buffer_tracing_kind_t kind,
                                                      rocprofiler_buffer_id_t           buffer_id)
{
    m_buffer_tracing_service_info.push_back(
        buffer_tracing_service_info{context_id.handle, kind, buffer_id.handle});
}

void MockSdkWrapper::query_agent_records(std::vector<rocprofiler_compute_tool::agent_record_t>& out_agents)
{
    out_agents = m_agent_records;
}

void MockSdkWrapper::set_available_gpu_agents(std::vector<rocprofiler_agent_id_t> agents)
{
    m_gpu_agents = std::move(agents);
}

void MockSdkWrapper::set_agent_records(std::vector<rocprofiler_compute_tool::agent_record_t> agents)
{
    m_agent_records = std::move(agents);
}

const std::vector<MockSdkWrapper::buffer_tracing_service_info>& MockSdkWrapper::get_buffer_tracing_service_info() const
{
    return m_buffer_tracing_service_info;
}

void MockSdkWrapper::set_pc_sampling_config(size_t                           min_interval,
                                            size_t                           max_interval,
                                            rocprofiler_pc_sampling_method_t method,
                                            rocprofiler_pc_sampling_unit_t   unit)
{
    m_pc_sampling_config     = pc_sampling_config_t{min_interval, max_interval, method, unit};
    m_pc_sampling_config_set = true;
}

void MockSdkWrapper::set_configure_pc_sampling_status(rocprofiler_status_t status)
{
    m_configure_pc_sampling_status = status;
}

const std::vector<MockSdkWrapper::create_buffer_info>& MockSdkWrapper::get_create_buffer_info() const
{
    return m_create_buffer_info;
}

const std::vector<MockSdkWrapper::configure_pc_sampling_info>& MockSdkWrapper::get_configure_pc_sampling_info() const
{
    return m_configure_pc_sampling_info;
}

const std::vector<MockSdkWrapper::flush_buffer_info>& MockSdkWrapper::get_flush_buffer_info() const
{
    return m_flush_buffer_info;
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

void MockPcSamplingCollector::on_code_object_load(
    const rocprofiler_callback_tracing_code_object_load_data_t& /*info*/)
{
    ++load_count;
}

void MockPcSamplingCollector::write(rocprofiler_compute_tool::code_object_writer_t& /*writer*/) {}

void MockPcSamplingCollector::append_sample(const rocprofiler_compute_tool::pc_sample_record_t& record)
{
    ++append_sample_count;
    appended_samples.push_back(record);
}

void MockPcSamplingCollector::add_kernel_symbol(uint64_t           code_object_id,
                                                const std::string& formatted_kernel_name,
                                                uint64_t           kernel_id)
{
    added_kernel_symbols.emplace_back(code_object_id, formatted_kernel_name);
    added_kernel_ids.push_back(kernel_id);
}

void MockPcSamplingCollector::add_agent(const rocprofiler_compute_tool::agent_record_t& agent)
{
    ++add_agent_count;
    added_agents.push_back(agent);
}

void MockPcSamplingCollector::append_kernel_dispatch(
    const rocprofiler_compute_tool::kernel_dispatch_record_t& record)
{
    ++append_kernel_dispatch_count;
    appended_kernel_dispatches.push_back(record);
}

void MockPcSamplingCollector::write_samples(rocprofiler_compute_tool::pc_sample_writer_t& /*writer*/)
{
    ++write_samples_count;
}

size_t MockPcSamplingCollector::snapshot_sources(const std::filesystem::path& /*output_root*/)
{
    ++snapshot_sources_count;
    return 0;
}
