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

#include <rocprofiler-sdk/cxx/hash.hpp>
#include <rocprofiler-sdk/cxx/operators.hpp>
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"

#include <rocprofiler-sdk/amd_detail/thread_trace.h>
#include <rocprofiler-sdk/intercept_table.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <shared_mutex>
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
    std::vector<uint32_t> perfcounters;

    static constexpr size_t DEFAULT_SIMD                  = 0x7;
    static constexpr size_t DEFAULT_PERFCOUNTER_SIMD_MASK = 0xF;
    static constexpr size_t DEFAULT_SE_MASK               = 0x21;
    static constexpr size_t DEFAULT_BUFFER_SIZE           = 0x8000000;
    static constexpr size_t PERFCOUNTER_SIMD_MASK_SHIFT   = 28;
};

namespace hsa
{
class AQLPacket;
};

struct ThreadTraceActiveResource
{
    rocprofiler_correlation_id_t    corr_id;
    rocprofiler_queue_id_t          queue_id;
    std::unique_ptr<hsa::AQLPacket> packet{nullptr};
};

class AgentThreadTracer
{
    using code_object_id_t = uint64_t;
    struct CodeobjRecord
    {
        code_object_id_t id;
        uint64_t         addr;
        uint64_t         size;
        bool             bUnload;
    };

public:
    AgentThreadTracer(thread_trace_parameter_pack _params,
                      const hsa::AgentCache&,
                      const CoreApiTable&,
                      const AmdExtTable&);
    virtual ~AgentThreadTracer();

    void load_codeobj(code_object_id_t id, uint64_t addr, uint64_t size);
    void unload_codeobj(code_object_id_t id);

    std::unique_ptr<hsa::AQLPacket> pre_kernel_call(rocprofiler_att_control_flags_t control_flags,
                                                    rocprofiler_queue_id_t          queue_id,
                                                    rocprofiler_correlation_id_t    corr_id);

    void post_kernel_call(std::unique_ptr<hsa::AQLPacket>&& aql);

    hsa_queue_t*                    queue = nullptr;
    std::mutex                      trace_resources_mut;
    thread_trace_parameter_pack     params;
    std::unique_ptr<hsa::AQLPacket> cached_resources;
    ThreadTraceActiveResource       active_resources;
    std::atomic<int>                data_is_ready{0};
    std::atomic<int>                active_queues{1};
    std::vector<CodeobjRecord>      remaining_codeobj_record;

    std::unique_ptr<aql::ThreadTraceAQLPacketFactory> factory;

private:
    bool Submit(hsa_ext_amd_aql_pm4_packet_t* packet);

    decltype(hsa_queue_load_read_index_relaxed)* load_read_index_relaxed_fn{nullptr};
    decltype(hsa_queue_add_write_index_relaxed)* add_write_index_relaxed_fn{nullptr};
    decltype(hsa_signal_store_screlease)*        signal_store_screlease_fn{nullptr};
    decltype(hsa_queue_destroy)*                 queue_destroy_fn{nullptr};
};  // namespace thread_trace

class GlobalThreadTracer
{
    struct CodeobjAddrRange
    {
        int64_t  addr;
        uint64_t size;
    };
    using AQLPacketPtr     = std::unique_ptr<hsa::AQLPacket>;
    using inst_pkt_t       = common::container::small_vector<std::pair<AQLPacketPtr, int64_t>, 4>;
    using corr_id_map_t    = hsa::Queue::queue_info_session_t::external_corr_id_map_t;
    using code_object_id_t = uint64_t;

public:
    GlobalThreadTracer(thread_trace_parameter_pack _params)
    : params(std::move(_params)){};
    virtual void start_context();
    virtual void stop_context();
    virtual void resource_init(const hsa::AgentCache&, const CoreApiTable&, const AmdExtTable&);
    virtual void resource_deinit(const hsa::AgentCache&);
    virtual ~GlobalThreadTracer() = default;

    static void codeobj_tracing_callback(rocprofiler_callback_tracing_record_t record,
                                         rocprofiler_user_data_t*              user_data,
                                         void*                                 callback_data);

    std::unique_ptr<hsa::AQLPacket> pre_kernel_call(const hsa::Queue&              queue,
                                                    uint64_t                       kernel_id,
                                                    const context::correlation_id* corr_id);

    void post_kernel_call(inst_pkt_t& aql);

    std::map<hsa_agent_t, std::map<code_object_id_t, CodeobjAddrRange>> loaded_codeobjs;
    std::unordered_map<hsa_agent_t, std::unique_ptr<AgentThreadTracer>> agents;

    std::atomic<int>            post_move_data{0};
    std::shared_mutex           agents_map_mut;
    rocprofiler_context_id_t    codeobj_client_ctx{0};
    thread_trace_parameter_pack params;
};  // namespace thread_trace

}  // namespace rocprofiler
