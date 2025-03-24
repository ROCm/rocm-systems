// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/callback_tracing.h>
#include <rocprofiler-sdk/fwd.h>

#include "lib/common/container/small_vector.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/aql_packet.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_info_session.hpp"
#include "lib/rocprofiler-sdk/hsa/rocprofiler_packet.hpp"
#include "lib/rocprofiler-sdk/kernel_dispatch/profiling_time.hpp"

#include <hsa/amd_hsa_kernel_code.h>
#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ven_amd_aqlprofile.h>
#include <hsa/hsa_ven_amd_loader.h>

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>

namespace rocprofiler
{
namespace context
{
struct context;
struct correlation_id;
}  // namespace context
namespace hsa
{
enum class queue_state
{
    normal       = 0,
    to_destroy   = 1,
    done_destroy = 2
};

// Interceptor for a single specific queue
class Queue
{
public:
    using context_t            = context::context;
    using context_array_t      = common::container::small_vector<const context_t*>;
    using callback_t           = void (*)(hsa_status_t status, hsa_queue_t* source, void* data);
    using queue_info_session_t = queue_info_session;

    struct pkt_and_serialize_t
    {
        std::unique_ptr<AQLPacket> pkt{nullptr};
        bool                       request_serialize{false};
    };

    // Function prototype used to notify consumers that a kernel has been enqueued.
    // Pair first: An AQL packet can be returned that will be injected into the queue.
    // Pair second: Boolean flag indicating the dispatch needs to be serialized.
    using queue_cb_t =
        std::function<pkt_and_serialize_t(const Queue&,
                                          const rocprofiler_packet&,
                                          rocprofiler_kernel_id_t,
                                          rocprofiler_dispatch_id_t,
                                          rocprofiler_user_data_t*,
                                          const queue_info_session_t::external_corr_id_map_t&,
                                          const context::correlation_id*)>;
    // Signals the completion of the kernel packet.
    using completed_cb_t = std::function<void(const Queue&,
                                              const rocprofiler_packet&,
                                              std::shared_ptr<Queue::queue_info_session_t>&,
                                              inst_pkt_t&,
                                              kernel_dispatch::profiling_time)>;
    using callback_map_t = std::unordered_map<ClientID, std::pair<queue_cb_t, completed_cb_t>>;

    Queue(const AgentCache& agent, CoreApiTable table);
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
    virtual ~Queue();

    const hsa_queue_t*        intercept_queue() const { return _intercept_queue; };
    virtual const AgentCache& get_agent() const { return _agent; }

    void create_signal(uint32_t attribute, hsa_signal_t* signal) const;
    void signal_async_handler(const hsa_signal_t& signal, void* data) const;

    template <typename FuncT>
    void signal_callback(FuncT&& func) const;

    template <typename FuncT>
    void lock_queue(FuncT&& func);

    virtual rocprofiler_queue_id_t get_id() const;

    // Fast check to see if we have any callbacks we need to notify
    int get_notifiers() const { return _notifiers; }

    // Tracks the number of in flight kernel executions we
    // are waiting on. We cannot destroy Queue until all kernels
    // have comleted.
    void    async_started() { _core_api.hsa_signal_add_relaxed_fn(_active_kernels, 1); }
    void    async_complete() { _core_api.hsa_signal_subtract_relaxed_fn(_active_kernels, 1); }
    int64_t active_async_packets() const
    {
        return _core_api.hsa_signal_load_scacquire_fn(_active_kernels);
    }
    void sync() const;

    void register_callback(ClientID id, queue_cb_t enqueue_cb, completed_cb_t complete_cb);
    void remove_callback(ClientID id);

    const CoreApiTable&             core_api() const { return _core_api; }
    const AmdExtTable&              ext_api() const { return _ext_api; }
    mutable std::mutex              cv_mutex;
    mutable std::condition_variable cv_ready_signal;
    hsa_signal_t                    block_signal;
    hsa_signal_t                    ready_signal;
    queue_state                     get_state() const;
    void                            set_state(queue_state state);

private:
    std::atomic<int>                     _notifiers            = {0};
    std::atomic<int64_t>                 _active_async_packets = {0};
    CoreApiTable                         _core_api             = {};
    AmdExtTable                          _ext_api              = {};
    const AgentCache&                    _agent;
    common::Synchronized<callback_map_t> _callbacks       = {};
    hsa_queue_t*                         _intercept_queue = nullptr;
    queue_state                          _state           = queue_state::normal;
    std::mutex                           _lock_queue;
    hsa_signal_t                         _active_kernels = {.handle = 0};
};

inline rocprofiler_queue_id_t
Queue::get_id() const
{
    return {.handle = intercept_queue()->id};
};

template <typename FuncT>
inline void
Queue::signal_callback(FuncT&& func) const
{
    _callbacks.rlock([&func](const auto& data) { func(data); });
}

template <typename FuncT>
void
Queue::lock_queue(FuncT&& func)
{
    std::unique_lock<std::mutex> lock(_lock_queue);
    func();
}
}  // namespace hsa
}  // namespace rocprofiler
