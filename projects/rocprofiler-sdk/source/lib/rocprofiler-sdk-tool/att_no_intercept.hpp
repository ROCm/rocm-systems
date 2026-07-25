// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/experimental/thread-trace/agent.h>
#include <rocprofiler-sdk/experimental/thread-trace/core.h>
#include <rocprofiler-sdk/fwd.h>

#include <cstddef>
#include <cstdint>
#include <unordered_set>

namespace rocprofiler
{
namespace tool
{
namespace att_no_intercept
{
using kernel_symbol_t = rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;

struct agent_trace_config_t
{
    rocprofiler_context_id_t context  = {};
    rocprofiler_user_data_t  userdata = {};
};

using shader_data_forwarder_t = void (*)(rocprofiler_thread_trace_shader_data_t,
                                         rocprofiler_user_data_t);

bool
is_supported();

void
configure(shader_data_forwarder_t    shader_data_forwarder,
          std::unordered_set<size_t> kernel_filter_range);

agent_trace_config_t
configure_agent(rocprofiler_agent_id_t id, uint64_t consecutive_kernels);

void
shader_data_callback(rocprofiler_thread_trace_shader_data_t shader_data,
                     rocprofiler_user_data_t                userdata);

void
code_object_load(const rocprofiler_callback_tracing_code_object_load_data_t& data);

void
kernel_symbol_load(const kernel_symbol_t& data, bool is_targeted);

void
finalize();
}  // namespace att_no_intercept
}  // namespace tool
}  // namespace rocprofiler
