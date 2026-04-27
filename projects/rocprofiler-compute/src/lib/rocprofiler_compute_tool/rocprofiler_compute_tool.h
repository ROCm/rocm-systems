// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include "counters_writer.h"
#include "input_parameters.h"
#include "sdk_wrapper.h"

#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <memory>

// Tool entry point. Therefore, must be outside of namespace.
rocprofiler_tool_configure_result_t* rocprofiler_configure(uint32_t                 version,
                                                           const char*              runtime_version,
                                                           uint32_t                 priority,
                                                           rocprofiler_client_id_t* id);

namespace rocprofiler_compute_tool
{
void dispatch_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                       rocprofiler_counter_config_id_t*             config,
                       rocprofiler_user_data_t*                     user_data,
                       void*                                        callback_data_args);

void record_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                     rocprofiler_counter_record_t*                record_data,
                     size_t                                       record_count,
                     rocprofiler_user_data_t                      user_data,
                     void*                                        callback_data_args);

void tool_tracing_callback(rocprofiler_callback_tracing_record_t record,
                           rocprofiler_user_data_t*              user_data,
                           void*                                 callback_data);

void code_object_tracing_callback(rocprofiler_callback_tracing_record_t record,
                                                        rocprofiler_user_data_t* user_data,
                                                        void*                    data);
}  // namespace rocprofiler_compute_tool

namespace rocprofiler_compute_tool::test_knobs
{
void set_input_parameters(const std::shared_ptr<input_parameters_t>& parameters);
void set_sdk_wrapper(const std::shared_ptr<sdk_wrapper_t>& sdk_wrapper);
void set_csv_writer(const std::shared_ptr<counters_writer_t>& csv_writer);
void reset_cfg();
}  // namespace rocprofiler_compute_tool::test_knobs