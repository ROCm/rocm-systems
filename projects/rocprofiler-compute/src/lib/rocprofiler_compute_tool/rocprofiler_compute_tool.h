// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include "counters_writer.h"
#include "input_parameters.h"
#include "sdk_wrapper.h"
#include "tool_setup.h"

#include <rocprofiler-sdk/registration.h>

#include <memory>

rocprofiler_tool_configure_result_t* rocprofiler_configure(uint32_t                 version,
                                                           const char*              runtime_version,
                                                           uint32_t                 priority,
                                                           rocprofiler_client_id_t* id);

namespace rocprofiler_compute_tool::test_knobs
{
void set_input_parameters(const std::shared_ptr<InputParameters>& parameters);
void set_sdk_wrapper(const std::shared_ptr<SdkWrapper>& sdk_wrapper);
void set_csv_writer(const std::shared_ptr<CountersWriter>& csv_writer);
void set_tool_setup(const std::shared_ptr<ToolSetUp>& tool_setup);
void reset_cfg();
void reset_tool_setup();
}  // namespace rocprofiler_compute_tool::test_knobs