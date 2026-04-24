// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include "counters_writer.h"
#include "input_parameters.h"
#include "sdk_wrapper.h"

#include <rocprofiler-sdk/registration.h>

#include <memory>

rocprofiler_tool_configure_result_t* rocprofiler_configure(uint32_t                 version,
                                                           const char*              runtime_version,
                                                           uint32_t                 priority,
                                                           rocprofiler_client_id_t* id);

namespace rocprofiler_compute_tool::test_knobs
{
void set_input_parameters(const std::shared_ptr<input_parameters_t>& parameters);
void set_sdk_wrapper(const std::shared_ptr<sdk_wrapper_t>& sdk_wrapper);
void set_csv_writer(const std::shared_ptr<counters_writer_t>& csv_writer);
void reset_cfg();
}  // namespace rocprofiler_compute_tool::test_knobs