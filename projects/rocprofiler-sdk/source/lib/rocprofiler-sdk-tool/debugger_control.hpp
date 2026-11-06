// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
#pragma once

#include <rocprofiler-sdk/experimental/thread_trace.h>

namespace rocprofiler::tool::debugger_control
{
bool
requested();
rocprofiler_context_id_t
configure(rocprofiler_thread_trace_shader_data_callback_t output);
bool
owns(rocprofiler_context_id_t context);
void
start_worker();
// Called by tool_detach/tool_fini, after the SDK has stopped client contexts.
void
finalize();
void
shader_data_callback(rocprofiler_thread_trace_shader_data_t data, rocprofiler_user_data_t userdata);
}  // namespace rocprofiler::tool::debugger_control
