// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include "counters_writer.h"
#include "env_parameters.h"
#include "pc_sampling_collector.h"
#include "sdk_callbacks.h"
#include "sdk_wrapper.h"

class mock_env_parameters_t : public rocprofiler_compute_tool::env_parameters_t
{
public:
    const char* get_output_path() override;
    const char* get_requested_counters() override;
    const char* get_iteration_multiplexing_mode() override;
    const char* get_kernel_filter_include_regex() override;
    const char* get_kernel_filter_range() override;
    const char* get_pc_sampling_mode() const override;

    void set_output_path(const std::string& output_path);
    void set_requested_counters(const std::string& counters);
    void set_iteration_multiplexing_mode(const std::string& mode);
    void set_kernel_filter_include_regex(const std::string& regex);
    void set_kernel_filter_range(const std::string& range);
    void set_pc_sampling_mode(const std::string& mode);

private:
    const char* m_non_empty_str               = "non empty string";
    std::string m_output_path                 = m_non_empty_str;
    std::string m_requested_counters          = m_non_empty_str;
    std::string m_iteration_multiplexing_mode = m_non_empty_str;
    std::string m_kernel_filter_include_regex = m_non_empty_str;
    std::string m_kernel_filter_range         = m_non_empty_str;
    std::string m_pc_sampling_mode            = m_non_empty_str;
};

class mock_sdk_wrapper_t : public rocprofiler_compute_tool::sdk_wrapper_t
{
public:
    struct dispatch_counting_service_info_t
    {
        uint64_t context                = 0;
        void*    dispatch_callback      = nullptr;
        void*    dispatch_callback_args = nullptr;
        void*    record_callback        = nullptr;
        void*    record_callback_args   = nullptr;
    };

    struct create_counter_config_info_t
    {
        std::vector<std::string> counter_names;
    };

    struct query_counter_record_info_t
    {
        uint64_t counter_instance_id = 0;
        uint64_t counter_id          = 0;
    };

    ~mock_sdk_wrapper_t() override = default;
    void create_context(rocprofiler_context_id_t* context_id) override;
    void configure_callback_dispatch_counting_service(
        rocprofiler_context_id_t                   context_id,
        rocprofiler_dispatch_counting_service_cb_t dispatch_callback,
        void*                                      dispatch_callback_args,
        rocprofiler_dispatch_counting_record_cb_t  record_callback,
        void*                                      record_callback_args) override;
    void configure_callback_tracing_service(rocprofiler_context_id_t               context_id,
                                            rocprofiler_callback_tracing_kind_t    kind,
                                            const rocprofiler_tracing_operation_t* operations,
                                            size_t                                 operations_count,
                                            rocprofiler_callback_tracing_cb_t      callback,
                                            void* callback_args) override;
    void start_context(rocprofiler_context_id_t context_id) override;
    void iterate_agent_supported_counters(rocprofiler_agent_id_t              agent_id,
                                          rocprofiler_available_counters_cb_t cb,
                                          void*                               user_data) override;
    void query_counter_info(rocprofiler_counter_id_t              counter_id,
                            rocprofiler_counter_info_version_id_t version,
                            void*                                 info) override;
    void create_counter_config(rocprofiler_agent_id_t           agent_id,
                               rocprofiler_counter_id_t*        counters_list,
                               size_t                           counters_count,
                               rocprofiler_counter_config_id_t* config_id) override;
    void query_record_counter_id(rocprofiler_counter_instance_id_t id,
                                 rocprofiler_counter_id_t*         counter_id) override;

    // Test functions
    void set_available_counters(const std::vector<std::string>& counter_names);
    const std::vector<uint64_t>&                         get_created_contexts() const;
    const std::vector<uint64_t>&                         get_started_contexts() const;
    const std::vector<dispatch_counting_service_info_t>& get_dispatch_counting_service_info() const;
    const std::vector<create_counter_config_info_t>&     get_create_counter_config_info() const;
    const std::vector<query_counter_record_info_t>&      get_query_counter_record_info() const;

private:
    std::vector<rocprofiler_counter_id_t> get_counters() const;

    std::vector<uint64_t>                         m_created_contexts;
    std::vector<uint64_t>                         m_started_contexts;
    std::vector<dispatch_counting_service_info_t> m_dispatch_counting_service_info;
    std ::vector<create_counter_config_info_t>    m_create_counter_config_info;
    std::vector<query_counter_record_info_t>      m_query_counter_record_info;
    std::vector<std::string>                      m_counter_names;
};

class mock_counters_writer_t : public rocprofiler_compute_tool::counters_writer_t
{
public:
    struct write_counters_info_t
    {
        std::vector<uint64_t> counter_ids;
        std::vector<uint64_t> kernel_id;
    };

    void write_counters(const rocprofiler_compute_tool::tool_data_t& tool_data) override;
    const std::vector<write_counters_info_t>& get_write_counters_info() const;

private:
    std::vector<write_counters_info_t> m_write_counters_args;
};

class mock_sdk_callbacks_t : public rocprofiler_compute_tool::sdk_callbacks_t
{
public:
    struct dispatch_callback_info_t
    {
        rocprofiler_dispatch_counting_service_data_t dispatch_data{};
        rocprofiler_counter_config_id_t*             config = nullptr;
    };

    struct record_callback_info_t
    {
        rocprofiler_dispatch_counting_service_data_t dispatch_data{};
        rocprofiler_counter_record_t*                record_data  = nullptr;
        size_t                                       record_count = 0;
    };

    struct tracing_callback_info_t
    {
        rocprofiler_callback_tracing_record_t record{};
    };

    void dispatch_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                           rocprofiler_counter_config_id_t*             config,
                           rocprofiler_compute_tool::tool_data_t&       tool_data) override;

    void record_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                         rocprofiler_counter_record_t*                record_data,
                         size_t                                       record_count,
                         rocprofiler_compute_tool::tool_data_t&       tool_data) override;

    void tool_tracing_callback(rocprofiler_callback_tracing_record_t  record,
                               rocprofiler_compute_tool::tool_data_t& tool_data) override;

    // Test accessors
    const std::vector<dispatch_callback_info_t>& get_dispatch_callback_info() const;
    const std::vector<record_callback_info_t>&   get_record_callback_info() const;
    const std::vector<tracing_callback_info_t>&  get_tracing_callback_info() const;

private:
    std::vector<dispatch_callback_info_t> m_dispatch_callback_info;
    std::vector<record_callback_info_t>   m_record_callback_info;
    std::vector<tracing_callback_info_t>  m_tracing_callback_info;
};

class mock_pc_sampling_collector_t : public pc_sampling_collector::pc_sampling_collector_t
{
public:
    void on_code_object_load(const rocprofiler_callback_tracing_code_object_load_data_t& info) override;
    const std::vector<rocprofiler_callback_tracing_code_object_load_data_t>& get_on_code_object_load_info() const;

private:
    std::vector<rocprofiler_callback_tracing_code_object_load_data_t> m_on_code_object_load_info = {};
};