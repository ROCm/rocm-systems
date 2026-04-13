// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include <gmock/gmock.h>
#include "input_parameters.h"
#include "sdk_callbacks.h"
#include "sdk_wrapper.h"

class MockInputParameters : public rocprofiler_compute_tool::InputParameters
{
public:
    MOCK_METHOD(char*, get_output_path, (), (override));
    MOCK_METHOD(char*, get_requested_counters, (), (override));
    MOCK_METHOD(char*, get_iteration_multiplexing_mode, (), (override));
    MOCK_METHOD(char*, get_kernel_filter_include_regex, (), (override));
    MOCK_METHOD(char*, get_kernel_filter_range, (), (override));
};

class MockSdkCallbacks : public rocprofiler_compute_tool::SdkCallbacks
{
public:
    MOCK_METHOD(void,
                dispatch_callback,
                (rocprofiler_dispatch_counting_service_data_t dispatch_data,
                 rocprofiler_counter_config_id_t*             config,
                 void*                                        callback_data_args),
                (override));
    MOCK_METHOD(void,
                record_callback,
                (rocprofiler_dispatch_counting_service_data_t dispatch_data,
                 rocprofiler_counter_record_t*                record_data,
                 size_t                                       record_count,
                 void*                                        callback_data_args),
                (override));
    MOCK_METHOD(void,
                tool_tracing_callback,
                (rocprofiler_callback_tracing_record_t record, void* callback_data),
                (override));
};

class MockSdkWrapper : public rocprofiler_compute_tool::SdkWrapper
{
public:
    MOCK_METHOD(void, create_context, (rocprofiler_context_id_t* context_id), (override));
    MOCK_METHOD(void,
                configure_callback_dispatch_counting_service,
                (rocprofiler_context_id_t                   context_id,
                 rocprofiler_dispatch_counting_service_cb_t dispatch_callback,
                 void*                                      dispatch_callback_args,
                 rocprofiler_dispatch_counting_record_cb_t  record_callback,
                 void*                                      record_callback_args),
                (override));
    MOCK_METHOD(void,
                configure_callback_tracing_service,
                (rocprofiler_context_id_t               context_id,
                 rocprofiler_callback_tracing_kind_t    kind,
                 const rocprofiler_tracing_operation_t* operations,
                 size_t                                 operations_count,
                 rocprofiler_callback_tracing_cb_t      callback,
                 void*                                  callback_args),
                (override));
    MOCK_METHOD(void, start_context, (rocprofiler_context_id_t context_id), (override));
    MOCK_METHOD(void,
                iterate_agent_supported_counters,
                (rocprofiler_agent_id_t              agent_id,
                 rocprofiler_available_counters_cb_t cb,
                 void*                               user_data),
                (override));
    MOCK_METHOD(void,
                query_counter_info,
                (rocprofiler_counter_id_t              counter_id,
                 rocprofiler_counter_info_version_id_t version,
                 void*                                 info),
                (override));
    MOCK_METHOD(void,
                create_counter_config,
                (rocprofiler_agent_id_t           agent_id,
                 rocprofiler_counter_id_t*        counters_list,
                 size_t                           counters_count,
                 rocprofiler_counter_config_id_t* config_id),
                (override));
    MOCK_METHOD(void,
                query_record_counter_id,
                (rocprofiler_counter_instance_id_t id, rocprofiler_counter_id_t* counter_id),
                (override));
};


