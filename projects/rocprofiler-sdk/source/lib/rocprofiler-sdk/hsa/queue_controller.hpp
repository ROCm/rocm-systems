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

#include "lib/rocprofiler-sdk/hsa/profile_serializer.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/kfd/doorbell_map.hpp"

#include "lib/rocprofiler-sdk-attach/table.h"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/cxx/hash.hpp>
#include <rocprofiler-sdk/cxx/operators.hpp>

#include <cstdint>
#include <functional>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace rocprofiler
{
namespace context
{
struct context;
}
namespace hsa
{
// Tracks and manages HSA queues
class QueueController
{
public:
    using agent_callback_tuple_t = std::tuple<rocprofiler_agent_t, queue_callbacks_t>;
    using queue_iterator_cb_t    = std::function<void(const Queue*)>;
    using callback_iterator_cb_t = std::function<void(ClientID, const agent_callback_tuple_t&)>;
    using queue_map_t            = std::unordered_map<hsa_queue_t*, std::unique_ptr<Queue>>;
    using agent_cache_map_t      = std::unordered_map<uint32_t, AgentCache>;

    QueueController()  = default;
    ~QueueController() = default;
    // Initializes the QueueInterceptor. This must be delayed until
    // HSA has been inited.
    void init(CoreApiTable& core_table, AmdExtTable& ext_table);

    // Called to add a queue that was created by the user program. |is_compute|
    // gates BOTH inline QueueState creation and the signal-less ownership/window
    // bookkeeping: only a compute queue's doorbell can source a CP dispatch-log
    // record (§3.9), and a non-compute (e.g. SDMA) queue's packet format is
    // incompatible with AQL interposition. |is_attach| marks a queue adopted
    // mid-life via the attach API: its earlier slot history went unseen, so it
    // opens no window and latches the process-wide signal-less disable (§0.4,
    // §3.9). Every call site passes both explicitly.
    void add_queue(hsa_queue_t*,
                   std::unique_ptr<Queue>,
                   bool is_compute = true,
                   bool is_attach  = false);
    void destroy_queue(hsa_queue_t*);

    // Add callback to queues associated with the agent. Returns a client
    // id that can be used by callers to remove the callback. If no agent
    // is specified, callback will be applied to all agents.
    ClientID add_callback(std::optional<rocprofiler_agent_t>, queue_callbacks_t callbacks);
    void     remove_callback(ClientID);

    const CoreApiTable& get_core_table() const { return _core_table; }
    const AmdExtTable&  get_ext_table() const { return _ext_table; }

    // Gets the list of supported HSA agents that can be Pintercepted
    const agent_cache_map_t& get_supported_agents() const;

    agent_cache_map_t& get_supported_agents();

    const Queue* get_queue(const hsa_queue_t&) const;

    void iterate_queues(const queue_iterator_cb_t&) const;
    void set_queue_state(queue_state state, hsa_queue_t* hsa_queue);

    void add_dispatch_ready(const Queue* queue);

    void iterate_callbacks(const callback_iterator_cb_t&) const;

    common::Synchronized<hsa::profiler_serializer>& serializer(const Queue*);

    // An empty set means every GPU agent.
    using agent_handle_set_t = std::unordered_set<rocprofiler_agent_id_t>;

    /**
     * Enable/disable serialization for QueueController, has no effect if counter collection
     * is not in use (which defaults to no serialization mechanism).
     *
     * Serialization is reference counted per agent. Counter collection, thread trace and SPM
     * each enable it independently, so an agent stays serialized until every subsystem that
     * asked for it has released it -- otherwise whichever service stops first would silently
     * unserialize the ones still running. The no-argument overloads apply to every GPU agent,
     * which is the behavior every caller had before agents could be scoped.
     */
    void enable_serialization();
    void disable_serialization();
    void enable_serialization(const agent_handle_set_t& agents);
    void disable_serialization(const agent_handle_set_t& agents);

    // Whether the given agent currently has serialization enabled. Exposed for tests and for
    // callers that need to reason about scope rather than trigger a transition.
    bool is_serialization_enabled(rocprofiler_agent_id_t agent_id) const;

    // Prints current state of signals for queues, used for debugging. Only prints
    // serialization related signals if not compiled in debug mode.
    void print_debug_signals() const;

#if !defined(NDEBUG)
    // Tracks the creation of all signals in queues, used for debugging and disabled
    // in release mode (adds locking around signal creation).
    common::Synchronized<std::unordered_map<uint64_t, hsa_signal_t>> _debug_signals;
#endif

private:
    using client_id_map_t  = std::unordered_map<ClientID, agent_callback_tuple_t>;
    using resource_alloc_t = void(const AgentCache&, const CoreApiTable&, const AmdExtTable&);

    CoreApiTable                          _core_table         = {};
    AmdExtTable                           _ext_table          = {};
    common::Synchronized<queue_map_t>     _queues             = {};
    common::Synchronized<client_id_map_t> _callback_cache     = {};
    agent_cache_map_t                     _supported_agents   = {};
    std::atomic<bool>                     _serialized_enabled = {false};
    common::Synchronized<
        std::unordered_map<rocprofiler_agent_id_t,
                           std::shared_ptr<common::Synchronized<hsa::profiler_serializer>>>>
        _profiler_serializer;
    // How many subsystems currently want serialization, per agent. `all` counts the callers
    // that asked for every agent; it is kept separate from `per_agent` because an agent's
    // serializer is created lazily on first use and an unscoped request has to cover agents
    // that do not exist yet.
    struct serialization_refcount
    {
        int64_t                                             all       = 0;
        std::unordered_map<rocprofiler_agent_id_t, int64_t> per_agent = {};

        int64_t count(rocprofiler_agent_id_t agent_id) const
        {
            auto itr = per_agent.find(agent_id);
            return all + ((itr == per_agent.end()) ? 0 : itr->second);
        }

        bool enabled(rocprofiler_agent_id_t agent_id) const { return count(agent_id) > 0; }

        bool any() const
        {
            if(all > 0) return true;
            for(const auto& [agent_id, count] : per_agent)
            {
                if(count > 0) return true;
            }
            return false;
        }
    };

    common::Synchronized<serialization_refcount> _serialization_refcount;

    // Applies a +1/-1 to the refcount of each agent in `agents` (empty == all GPU agents) and
    // toggles only the serializers whose effective count crossed zero.
    void update_serialization(const agent_handle_set_t& agents, bool enable);
};

QueueController*
get_queue_controller();

bool
enable_queue_intercept();

bool
context_needs_queue_interposition_tracing(const context::context* ctx);

void
queue_controller_init(HsaApiTable* table);

void
queue_controller_fini();

void
queue_controller_sync();

void
queue_controller_init(RocAttachDispatchTable* table);

void
profiler_serializer_kernel_completion_signal(hsa_signal_t queue_block_signal);

// Resolve a queue's page-relative doorbell slot from its intercept queue's
// hardware doorbell pointer (§3.2, pure derivation -- no bind, no GPU/queue id).
// Capture and the reader MUST compute the identical slot for correlation to work.
// nullopt when the queue's doorbell signal is missing or is not a doorbell kind.
std::optional<uint32_t>
capture_doorbell_key(const hsa_queue_t* intercept_queue);
}  // namespace hsa
}  // namespace rocprofiler
