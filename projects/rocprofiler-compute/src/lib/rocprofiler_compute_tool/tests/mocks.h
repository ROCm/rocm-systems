// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include "counters_writer.h"
#include "input_parameters.h"
#include "sdk_callbacks.h"
#include "sdk_wrapper.h"

#include <gmock/gmock.h>

#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

class MockInputParameters : public rocprofiler_compute_tool::InputParameters
{
public:
    std::string_view get_output_path() override;
    std::string_view get_requested_counters() override;
    std::string_view get_iteration_multiplexing_mode() override;
    std::string_view get_kernel_filter_include_regex() override;
    std::string_view get_kernel_filter_range() override;
    std::string_view get_pc_sampling_method() override;
    std::string_view get_pc_sampling_beta_enabled() override;
    std::string_view get_pc_sampling_interval() override;

    void set_output_path(const std::string& output_path);
    void set_requested_counters(const std::string& counters);
    void set_iteration_multiplexing_mode(const std::string& mode);
    void set_kernel_filter_include_regex(const std::string& regex);
    void set_kernel_filter_range(const std::string& range);
    void set_pc_sampling_method(const std::string& method);
    void set_pc_sampling_beta_enabled(const std::string& value);
    void set_pc_sampling_interval(const std::string& interval);

    void unset_output_path();
    void unset_requested_counters();
    void unset_iteration_multiplexing_mode();
    void unset_kernel_filter_include_regex();
    void unset_kernel_filter_range();

private:
    const char* m_non_empty_str               = "non empty string";
    std::string m_output_path                 = m_non_empty_str;
    std::string m_requested_counters          = m_non_empty_str;
    std::string m_iteration_multiplexing_mode = m_non_empty_str;
    std::string m_kernel_filter_include_regex = m_non_empty_str;
    std::string m_kernel_filter_range         = m_non_empty_str;
    std::string m_pc_sampling_method;
    std::string m_pc_sampling_beta_enabled;
    std::string m_pc_sampling_interval;

    bool m_output_path_set                 = true;
    bool m_requested_counters_set          = true;
    bool m_iteration_multiplexing_mode_set = true;
    bool m_kernel_filter_include_regex_set = true;
    bool m_kernel_filter_range_set         = true;
};

class MockSdkWrapper : public rocprofiler_compute_tool::SdkWrapper
{
public:
    struct dispatch_counting_service_info
    {
        uint64_t context                = 0;
        void*    dispatch_callback      = nullptr;
        void*    dispatch_callback_args = nullptr;
        void*    record_callback        = nullptr;
        void*    record_callback_args   = nullptr;
    };

    struct create_counter_config_info
    {
        std::vector<std::string> counter_names;
    };

    struct query_counter_record_info
    {
        uint64_t counter_instance_id = 0;
        uint64_t counter_id          = 0;
    };

    ~MockSdkWrapper() override = default;
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

    void at_intercept_table_registration_hsa(rocprofiler_intercept_library_cb_t callback,
                                             void*                              user_data) override;

    void query_available_gpu_agents(std::vector<rocprofiler_agent_id_t>& out_gpu_agents) override;
    void query_pc_sampling_configs(rocprofiler_agent_id_t                                agent_id,
                                   rocprofiler_available_pc_sampling_configurations_cb_t cb,
                                   void* user_data) override;
    void create_buffer(rocprofiler_context_id_t        context_id,
                       size_t                          size,
                       size_t                          watermark,
                       rocprofiler_buffer_policy_t     policy,
                       rocprofiler_buffer_tracing_cb_t callback,
                       void*                           callback_data,
                       rocprofiler_buffer_id_t*        buffer_id) override;
    rocprofiler_status_t configure_pc_sampling_service(rocprofiler_context_id_t         context_id,
                                                       rocprofiler_agent_id_t           agent_id,
                                                       rocprofiler_pc_sampling_method_t method,
                                                       rocprofiler_pc_sampling_unit_t   unit,
                                                       uint64_t                         interval,
                                                       rocprofiler_buffer_id_t          buffer_id,
                                                       int flags) override;
    void                 flush_buffer(rocprofiler_buffer_id_t buffer_id) override;
    void configure_buffer_tracing_service(rocprofiler_context_id_t          context_id,
                                          rocprofiler_buffer_tracing_kind_t kind,
                                          rocprofiler_buffer_id_t           buffer_id) override;
    void query_agent_records(std::vector<rocprofiler_compute_tool::agent_record_t>& out_agents) override;

    struct buffer_tracing_service_info
    {
        uint64_t                          context   = 0;
        rocprofiler_buffer_tracing_kind_t kind      = ROCPROFILER_BUFFER_TRACING_NONE;
        uint64_t                          buffer_id = 0;
    };

    struct hsa_intercept_registration_info
    {
        rocprofiler_intercept_library_cb_t callback  = nullptr;
        void*                              user_data = nullptr;
    };

    struct create_buffer_info
    {
        uint64_t context   = 0;
        size_t   size      = 0;
        size_t   watermark = 0;
        uint64_t buffer_id = 0;
    };

    struct configure_pc_sampling_info
    {
        rocprofiler_agent_id_t           agent{};
        rocprofiler_pc_sampling_method_t method    = ROCPROFILER_PC_SAMPLING_METHOD_NONE;
        rocprofiler_pc_sampling_unit_t   unit      = ROCPROFILER_PC_SAMPLING_UNIT_NONE;
        uint64_t                         interval  = 0;
        uint64_t                         buffer_id = 0;
    };

    struct flush_buffer_info
    {
        uint64_t buffer_id = 0;
    };

    struct pc_sampling_config_t
    {
        size_t                           min_interval = 0;
        size_t                           max_interval = 0;
        rocprofiler_pc_sampling_method_t method       = ROCPROFILER_PC_SAMPLING_METHOD_NONE;
        rocprofiler_pc_sampling_unit_t   unit         = ROCPROFILER_PC_SAMPLING_UNIT_NONE;
    };

    // Test functions
    void set_available_counters(const std::vector<std::string>& counter_names);
    const std::vector<uint64_t>&                        get_created_contexts() const;
    const std::vector<uint64_t>&                        get_started_contexts() const;
    const std::vector<dispatch_counting_service_info>&  get_dispatch_counting_service_info() const;
    const std::vector<create_counter_config_info>&      get_create_counter_config_info() const;
    const std::vector<query_counter_record_info>&       get_query_counter_record_info() const;
    const std::vector<hsa_intercept_registration_info>& get_hsa_intercept_registration_info() const;

    void set_available_gpu_agents(std::vector<rocprofiler_agent_id_t> agents);
    void set_agent_records(std::vector<rocprofiler_compute_tool::agent_record_t> agents);
    const std::vector<buffer_tracing_service_info>& get_buffer_tracing_service_info() const;
    void set_pc_sampling_config(size_t                           min_interval,
                                size_t                           max_interval,
                                rocprofiler_pc_sampling_method_t method,
                                rocprofiler_pc_sampling_unit_t   unit);
    // Append an additional config; query_pc_sampling_configs delivers every
    // configured entry in one callback so the multi-config selection path can be
    // exercised.
    void add_pc_sampling_config(size_t                           min_interval,
                                size_t                           max_interval,
                                rocprofiler_pc_sampling_method_t method,
                                rocprofiler_pc_sampling_unit_t   unit);
    void set_configure_pc_sampling_status(rocprofiler_status_t status);

    const std::vector<create_buffer_info>&         get_create_buffer_info() const;
    const std::vector<configure_pc_sampling_info>& get_configure_pc_sampling_info() const;
    const std::vector<flush_buffer_info>&          get_flush_buffer_info() const;

private:
    std::vector<rocprofiler_counter_id_t> get_counters() const;

    std::vector<uint64_t>                        m_created_contexts;
    std::vector<uint64_t>                        m_started_contexts;
    std::vector<dispatch_counting_service_info>  m_dispatch_counting_service_info;
    std ::vector<create_counter_config_info>     m_create_counter_config_info;
    std::vector<query_counter_record_info>       m_query_counter_record_info;
    std::vector<std::string>                     m_counter_names;
    std::vector<hsa_intercept_registration_info> m_hsa_intercept_registration_info;

    std::vector<rocprofiler_agent_id_t>                   m_gpu_agents;
    std::vector<rocprofiler_compute_tool::agent_record_t> m_agent_records;
    std::vector<buffer_tracing_service_info>              m_buffer_tracing_service_info;
    pc_sampling_config_t                                  m_pc_sampling_config{};
    bool                                                  m_pc_sampling_config_set = false;
    std::vector<pc_sampling_config_t>                     m_extra_pc_sampling_configs;
    rocprofiler_status_t            m_configure_pc_sampling_status = ROCPROFILER_STATUS_SUCCESS;
    uint64_t                        m_next_buffer_id               = 1;
    std::vector<create_buffer_info> m_create_buffer_info;
    std::vector<configure_pc_sampling_info> m_configure_pc_sampling_info;
    std::vector<flush_buffer_info>          m_flush_buffer_info;
};

class MockCountersWriter : public rocprofiler_compute_tool::CountersWriter
{
public:
    struct write_counters_info
    {
        std::vector<uint64_t> counter_ids;
        std::vector<uint64_t> kernel_id;
    };

    void write_counters(rocprofiler_compute_tool::tool_data_t* tool_data) override;
    const std::vector<write_counters_info>& get_write_counters_info() const;

private:
    std::vector<write_counters_info> m_write_counters_args;
};

class MockPcSamplingCollector : public rocprofiler_compute_tool::pc_sampling_collector_t
{
public:
    void on_code_object_load(const rocprofiler_callback_tracing_code_object_load_data_t& info) override;
    void write(rocprofiler_compute_tool::code_object_writer_t& writer) override;

    void append_sample(const rocprofiler_compute_tool::pc_sample_record_t& record) override;
    void add_kernel_symbol(uint64_t           code_object_id,
                           const std::string& formatted_kernel_name,
                           uint64_t           kernel_id) override;
    void add_agent(const rocprofiler_compute_tool::agent_record_t& agent) override;
    void append_kernel_dispatch(const rocprofiler_compute_tool::kernel_dispatch_record_t& record) override;
    void   write_samples(rocprofiler_compute_tool::pc_sample_writer_t& writer) override;
    size_t snapshot_sources(const std::filesystem::path& output_root) override;

    int                                                       load_count                   = 0;
    int                                                       append_sample_count          = 0;
    int                                                       write_samples_count          = 0;
    int                                                       snapshot_sources_count       = 0;
    int                                                       add_agent_count              = 0;
    int                                                       append_kernel_dispatch_count = 0;
    std::vector<rocprofiler_compute_tool::pc_sample_record_t> appended_samples;
    std::vector<std::pair<uint64_t, std::string>>             added_kernel_symbols;
    std::vector<uint64_t>                                     added_kernel_ids;
    std::vector<rocprofiler_compute_tool::agent_record_t>     added_agents;
    std::vector<rocprofiler_compute_tool::kernel_dispatch_record_t> appended_kernel_dispatches;
};
