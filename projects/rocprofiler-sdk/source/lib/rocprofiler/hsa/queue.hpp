// Copyright (c) 2018-2023 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

#include <rocprofiler/fwd.h>

#include "lib/common/synchronized.hpp"
#include "lib/rocprofiler/hsa/agent_cache.hpp"
#include "lib/rocprofiler/hsa/aql_packet.hpp"

#include <hsa/amd_hsa_kernel_code.h>
#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ven_amd_loader.h>

#include <atomic>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace rocprofiler
{
namespace hsa
{
using ClientID = int64_t;

// Interceptor for a single specific queue
class Queue
{
public:
    using callback_t = void (*)(hsa_status_t status, hsa_queue_t* source, void* data);
    // Function prototype used to notify consumers that a kernel has been
    // enqueued. An AQL packet can be returned that will be injected into
    // the queue.
    using QueueCB = std::function<
        std::unique_ptr<AQLPacket>(const Queue&, ClientID, const hsa_ext_amd_aql_pm4_packet_t&)>;
    // Signals the completion of the kernel packet.
    using CompletedCB    = std::function<void(const Queue&,
                                           ClientID,
                                           const hsa_ext_amd_aql_pm4_packet_t&,
                                           std::unique_ptr<AQLPacket>)>;
    using callback_map_t = std::unordered_map<ClientID, std::pair<QueueCB, CompletedCB>>;

    // Internal session information that is used by write interceptor
    // to track state of the intercepted kernel.
    struct queue_info_session_t
    {
        Queue&                       queue;
        std::unique_ptr<AQLPacket>   inst_pkt         = {};
        ClientID                     inst_pkt_id      = 0;
        hsa_ext_amd_aql_pm4_packet_t kernel_pkt       = null_amd_aql_pm4_packet;
        hsa_signal_t                 interrupt_signal = {};
    };

    Queue(const AgentCache&  agent,
          uint32_t           size,
          hsa_queue_type32_t type,
          callback_t         callback,
          void*              data,
          uint32_t           private_segment_size,
          uint32_t           group_segment_size,
          CoreApiTable       core_api,
          AmdExtTable        ext_api,
          hsa_queue_t**      queue);

    ~Queue();

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

    void register_callback(ClientID id, QueueCB enqueue_cb, CompletedCB complete_cb);
    void remove_callback(ClientID id);

    const CoreApiTable& core_api() const { return _core_api; }
    const AmdExtTable&  ext_api() const { return _ext_api; }

private:
    std::atomic<int>                                  _notifiers            = {0};
    std::atomic<int64_t>                              _active_async_packets = {0};
    CoreApiTable                                      _core_api             = {};
    AmdExtTable                                       _ext_api              = {};
    const AgentCache&                                 _agent;
    rocprofiler::common::Synchronized<callback_map_t> _callbacks       = {};
    hsa_queue_t*                                      _intercept_queue = nullptr;
};

}  // namespace hsa
}  // namespace rocprofiler
