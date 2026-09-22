// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include "counters_writer.h"
#include "input_parameters.h"
#include "sdk_wrapper.h"

#include <rocprofiler-sdk/registration.h>

#include <memory>
#include <string_view>

rocprofiler_tool_configure_result_t* rocprofiler_configure(uint32_t                 version,
                                                           const char*              runtime_version,
                                                           uint32_t                 priority,
                                                           rocprofiler_client_id_t* id);

namespace rocprofiler_compute_tool
{
iteration_multiplexing_mode_t iteration_multiplexing_mode(std::string_view mode);
}  // namespace rocprofiler_compute_tool

namespace rocprofiler_compute_tool::test_knobs
{
void set_input_parameters(const std::shared_ptr<InputParameters>& parameters);
void set_sdk_wrapper(const std::shared_ptr<SdkWrapper>& sdk_wrapper);
/// Call after rocprofiler_configure, which is where the writers are
/// registered. Returns false when no writer of that name is registered.
bool replace_writer(std::string_view name, const std::shared_ptr<OutputWriter>& writer);
void reset_cfg();
}  // namespace rocprofiler_compute_tool::test_knobs
