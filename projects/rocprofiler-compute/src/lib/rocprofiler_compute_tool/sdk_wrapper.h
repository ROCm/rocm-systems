// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include <rocprofiler-sdk/rocprofiler.h>

namespace rocprofiler_compute_tool
{
class SdkWrapper
{
public:
    virtual ~SdkWrapper()                                             = default;
    virtual void create_context(rocprofiler_context_id_t* context_id) = 0;
    virtual void configure_callback_dispatch_counting_service(
        rocprofiler_context_id_t                   context_id,
        rocprofiler_dispatch_counting_service_cb_t dispatch_callback,
        void*                                      dispatch_callback_args,
        rocprofiler_dispatch_counting_record_cb_t  record_callback,
        void*                                      record_callback_args) = 0;

    virtual void configure_callback_tracing_service(rocprofiler_context_id_t            context_id,
                                                    rocprofiler_callback_tracing_kind_t kind,
                                                    const rocprofiler_tracing_operation_t* operations,
                                                    size_t operations_count,
                                                    rocprofiler_callback_tracing_cb_t callback,
                                                    void* callback_args) = 0;

    virtual void start_context(rocprofiler_context_id_t context_id) = 0;

    virtual void iterate_agent_supported_counters(rocprofiler_agent_id_t              agent_id,
                                                  rocprofiler_available_counters_cb_t cb,
                                                  void* user_data) = 0;

    virtual void query_counter_info(rocprofiler_counter_id_t              counter_id,
                                    rocprofiler_counter_info_version_id_t version,
                                    void*                                 info) = 0;

    virtual void create_counter_config(rocprofiler_agent_id_t           agent_id,
                                       rocprofiler_counter_id_t*        counters_list,
                                       size_t                           counters_count,
                                       rocprofiler_counter_config_id_t* config_id) = 0;

    virtual void query_record_counter_id(rocprofiler_counter_instance_id_t id,
                                         rocprofiler_counter_id_t*         counter_id) = 0;

    virtual void at_intercept_table_registration_hsa(rocprofiler_intercept_library_cb_t callback,
                                                     void* user_data) = 0;
};

class SdkWrapperImpl : public SdkWrapper
{
public:
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
};
}  // namespace rocprofiler_compute_tool
