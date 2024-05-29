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

#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"

#include <rocprofiler-sdk/amd_detail/thread_trace.h>
#include <rocprofiler-sdk/intercept_table.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace rocprofiler
{
struct thread_trace_parameter_pack
{
    rocprofiler_context_id_t               context_id;
    rocprofiler_att_dispatch_callback_t    dispatch_cb_fn;
    rocprofiler_att_shader_data_callback_t shader_cb_fn;
    void*                                  callback_userdata;

    // Parameters
    uint8_t  target_cu          = 1;
    uint8_t  simd_select        = DEFAULT_SIMD;
    uint8_t  perfcounter_ctrl   = 0;
    uint64_t shader_engine_mask = DEFAULT_SE_MASK;
    uint64_t buffer_size        = DEFAULT_BUFFER_SIZE;

    // GFX9 Only
    std::vector<std::string> perfcounters;

    static constexpr size_t DEFAULT_SIMD        = 0x7;
    static constexpr size_t DEFAULT_SE_MASK     = 0x21;
    static constexpr size_t DEFAULT_BUFFER_SIZE = 0x6000000;
};

namespace hsa
{
class AQLPacket;
};

class ThreadTracer
{
public:
    ThreadTracer(std::shared_ptr<thread_trace_parameter_pack>& _params)
    : params(_params){};
    virtual void start_context();
    virtual void stop_context();
    virtual void resource_init(const hsa::AgentCache&, const CoreApiTable&, const AmdExtTable&);
    virtual void resource_deinit(const hsa::AgentCache&);
    virtual ~ThreadTracer() = default;

    std::mutex                                                    trace_resources_mut;
    std::shared_ptr<thread_trace_parameter_pack>                  params;
    std::unordered_map<uint64_t, std::unique_ptr<hsa::AQLPacket>> resources;
    std::unordered_map<uint64_t, std::atomic<int>>                agent_active_queues;
};  // namespace thread_trace

}  // namespace rocprofiler
