/* Copyright (c) 2022 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#pragma once

#include <atomic>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <hsa/amd_hsa_kernel_code.h>
#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>
#include <hsa/hsa_ven_amd_loader.h>

#include <rocprofiler/fwd.h>
#include "lib/common/synchronized.hpp"
#include "lib/rocprofiler/hsa/agent_cache.hpp"

namespace rocprofiler
{
namespace hsa
{
/**
 * Struct containing AQL packet information. Including start/stop/read
 * packets along with allocated buffers
 */
struct AQLPacket
{
    hsa_ven_amd_aqlprofile_profile_t                  profile;
    hsa_ext_amd_aql_pm4_packet_t                      start{.header            = 0,
                                       .pm4_command       = {0},
                                       .completion_signal = {.handle = 0}};
    hsa_ext_amd_aql_pm4_packet_t                      stop{.header            = 0,
                                      .pm4_command       = {0},
                                      .completion_signal = {.handle = 0}};
    hsa_ext_amd_aql_pm4_packet_t                      read{.header            = 0,
                                      .pm4_command       = {0},
                                      .completion_signal = {.handle = 0}};
    bool                                              command_buf_mallocd{false};
    bool                                              output_buffer_malloced{false};
    std::function<decltype(hsa_amd_memory_pool_free)> free_func;
    AQLPacket(std::function<decltype(hsa_amd_memory_pool_free)> func)
    : free_func(std::move(func))
    {}

    ~AQLPacket()
    {
        if(!command_buf_mallocd)
        {
            free_func(profile.command_buffer.ptr);
        }
        else
        {
            free(profile.command_buffer.ptr);
        }

        if(!output_buffer_malloced)
        {
            free_func(profile.output_buffer.ptr);
        }
        else
        {
            free(profile.output_buffer.ptr);
        }
    }

    // Keep move constuctors (i.e. std::move())
    AQLPacket(AQLPacket&& other) = default;
    AQLPacket& operator=(AQLPacket&& other) = default;

    // Do not allow copying this class
    AQLPacket(const AQLPacket&) = delete;
    AQLPacket& operator=(const AQLPacket&) = delete;
};

using ClientID = int64_t;

// Interceptor for a single specific queue
class Queue
{
public:
    // Internal session information that is used by write interceptor
    // to track state of the intercepted kernel.
    struct queue_info_session_t
    {
        Queue&                       queue;
        std::unique_ptr<AQLPacket>   inst_pkt;
        ClientID                     inst_pkt_id;
        hsa_ext_amd_aql_pm4_packet_t kernel_pkt;
        hsa_signal_t                 interrupt_signal;
    };

    Queue(const AgentCache&  agent,
          uint32_t           size,
          hsa_queue_type32_t type,
          void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data),
          void*         data,
          uint32_t      private_segment_size,
          uint32_t      group_segment_size,
          CoreApiTable  core_api,
          AmdExtTable   ext_api,
          hsa_queue_t** queue);

    const hsa_queue_t* intercept_queue() const { return _intercept_queue; };
    const AgentCache&  get_agent() const { return _agent; }

    void create_signal(uint32_t attribute, hsa_signal_t* signal) const;
    void signal_async_handler(const hsa_signal_t& signal, Queue::queue_info_session_t* data) const;

    rocprofiler_queue_id_t get_id() const
    {
        return {.handle = reinterpret_cast<uint64_t>(intercept_queue())};
    };

    template <class Func>
    void signal_callback(Func&& func) const
    {
        _callbacks.rlock([&func](const auto& data) { func(data); });
    }

    // Fast check to see if we have any callbacks we need to notify
    int get_notifiers() const { return _notifiers; }

    // Tracks the number of in flight kernel executions we
    // are waiting on. We cannot destroy Queue until all kernels
    // have comleted.
    void async_started() { _active_async_packets++; }
    void async_complete() { _active_async_packets--; }

    ~Queue()
    {
        // Potentially replace with condition variable at some point
        // but performance may not matter here.
        while(_active_async_packets > 0)
        {}
    }

    // Function prototype used to notify consumers that a kernel has been
    // enqueued. An AQL packet can be returned that will be injected into
    // the queue.
    using QueueCB = std::function<
        std::unique_ptr<AQLPacket>(const Queue&, ClientID, const hsa_ext_amd_aql_pm4_packet_t&)>;
    // Signals the completion of the kernel packet.
    using CompletedCB = std::function<void(const Queue&,
                                           ClientID,
                                           const hsa_ext_amd_aql_pm4_packet_t&,
                                           std::unique_ptr<AQLPacket>)>;

    void register_callback(ClientID id, QueueCB enqueue_cb, CompletedCB complete_cb);
    void remove_callback(ClientID id);

    const CoreApiTable& core_api() const { return _core_api; }
    const AmdExtTable&  ext_api() const { return _ext_api; }

private:
    std::atomic<int64_t> _active_async_packets{0};
    CoreApiTable         _core_api;
    AmdExtTable          _ext_api;
    const AgentCache&    _agent;
    std::atomic<int>     _notifiers;
    rocprofiler::common::Synchronized<std::unordered_map<ClientID, std::pair<QueueCB, CompletedCB>>>
                 _callbacks;
    hsa_queue_t* _intercept_queue;
};

// Tracks and manages HSA queues
class QueueController
{
public:
    QueueController() = default;
    // Initializes the QueueInterceptor. This must be delayed until
    // HSA has been inited.
    void Init(CoreApiTable& core_table, AmdExtTable& ext_table);
    // Called to add a queue that was created by the user program
    void add_queue(hsa_queue_t*, std::unique_ptr<Queue>);
    void destory_queue(hsa_queue_t*);

    // Add callback to queues associated with the agent. Returns a client
    // id that can be used by callers to remove the callback.
    ClientID add_callback(const rocprofiler_agent_t&, Queue::QueueCB, Queue::CompletedCB);
    void     remove_callback(ClientID);

    const CoreApiTable& get_core_table() const { return _core_table; }
    const AmdExtTable&  get_ext_table() const { return _ext_table; }

    // Gets the list of supported HSA agents that can be intercepted
    const std::unordered_map<uint32_t, AgentCache>& get_supported_agents() const
    {
        return _supported_agents;
    }

    std::unordered_map<uint32_t, AgentCache>& get_supported_agents() { return _supported_agents; }

private:
    CoreApiTable _core_table;
    AmdExtTable  _ext_table;
    rocprofiler::common::Synchronized<std::unordered_map<hsa_queue_t*, std::unique_ptr<Queue>>>
        _queues;
    rocprofiler::common::Synchronized<
        std::unordered_map<ClientID,
                           std::tuple<rocprofiler_agent_t, Queue::QueueCB, Queue::CompletedCB>>>
        _callback_cache;

    std::unordered_map<uint32_t, AgentCache> _supported_agents;
};

QueueController&
get_queue_controller();

void
queue_controller_init(HsaApiTable* table);

}  // namespace hsa
}  // namespace rocprofiler
