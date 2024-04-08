// MIT License
//
// Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rocprofiler-sdk/intercept_table.h>
#include <tuple>
#include "include/rocprofiler-sdk/thread_trace.h"

namespace rocprofiler
{
struct thread_trace_parameters
{
    rocprofiler_context_id_t               context_id;
    rocprofiler_att_dispatch_callback_t    dispatch_cb_fn;
    rocprofiler_att_shader_data_callback_t shader_cb_fn;
    void*                                  callback_userdata;

    // Parameters
    rocprofiler_att_parameter_flag_t flags;
    uint64_t                         buffer_size;
    uint8_t                          target_cu;
    uint8_t                          simd_select;
    uint8_t                          reserved;
    uint8_t                          vmid_mask;
    uint16_t                         perfcounter_mask;
    uint8_t                          perfcounter_ctrl;
    uint64_t                         shader_engine_mask;

    // GFX9 Only
    std::vector<std::string> perfcounters;
};

namespace hsa
{
class AQLPacket;
};

class ThreadTracer
{
public:
    ThreadTracer(std::shared_ptr<thread_trace_parameters>& _params)
    : params(_params){};
    virtual void start_context();
    virtual void stop_context();
    virtual void resource_init(const hsa::AgentCache&, const CoreApiTable&, const AmdExtTable&);
    virtual void resource_deinit(const hsa::AgentCache&);
    virtual ~ThreadTracer() = default;

    std::shared_ptr<thread_trace_parameters>                      params;
    std::mutex                                                    trace_resources_mut;
    std::unordered_map<uint64_t, std::unique_ptr<hsa::AQLPacket>> resources;
    std::unordered_map<uint64_t, std::atomic<int>>                agent_active_queues;
};  // namespace thread_trace

}  // namespace rocprofiler
