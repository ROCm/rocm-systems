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

#include "lib/rocprofiler-sdk/hsa/replay_window.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/hsa/rocprofiler_packet.hpp"

#include <rocprofiler-sdk/cxx/hash.hpp>
#include <rocprofiler-sdk/cxx/operators.hpp>

#include <fmt/format.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <thread>
#include <unordered_map>

namespace rocprofiler
{
namespace hsa
{
namespace
{
thread_local bool tl_passthrough = false;

struct agent_replay_state_t
{
    std::shared_mutex     mutex        = {};
    std::atomic<uint64_t> submit_epoch = {0};
};

agent_replay_state_t&
agent_replay_state(rocprofiler_agent_id_t agent_id)
{
    // No get_fini_status() guard is needed here. Every caller reaches this only when a replay
    // service is active, and that is false during finalization, so the static map below is never
    // touched after teardown.
    using state_map_t    = std::unordered_map<rocprofiler_agent_id_t, agent_replay_state_t>;
    static auto*& states = common::static_object<common::Synchronized<state_map_t>>::construct();

    // Every submit looks its agent up. Nodes are never erased and unordered_map references survive
    // rehashing, so a reference stays valid without the lock and a thread can cache its last one.
    thread_local auto                  tl_agent_id = rocprofiler_agent_id_t{.handle = UINT64_MAX};
    thread_local agent_replay_state_t* tl_state    = nullptr;
    if(tl_state != nullptr && tl_agent_id.handle == agent_id.handle) return *tl_state;

    auto* state = const_cast<agent_replay_state_t*>(states->rlock(
        [](const state_map_t& _map, auto _id) -> const agent_replay_state_t* {
            const auto itr = _map.find(_id);
            return (itr == _map.end()) ? nullptr : &itr->second;
        },
        agent_id));
    if(state == nullptr)
        state = &states->wlock(
            [](state_map_t& _map, auto _id) -> agent_replay_state_t& { return _map[_id]; },
            agent_id);

    tl_agent_id = agent_id;
    tl_state    = state;
    return *state;
}

// Async completion handlers still running on `agent`'s queues, including queues destroy_queue is
// still syncing. The map is read before the retiring set; see iterate_retiring_queues for why.
int64_t
agent_async_in_flight(const QueueController& controller, hsa_agent_t agent)
{
    int64_t    in_flight = 0;
    const auto count     = [&](const Queue* queue) {
        if(queue != nullptr && queue->get_agent().get_hsa_agent() == agent)
            in_flight += queue->active_async_packets();
    };
    controller.iterate_queues(count);
    controller.iterate_retiring_queues(count);
    return in_flight;
}

// Calls `ready` every couple of milliseconds until it returns true or `timeout` passes. Returns
// whether it did.
template <typename ReadyFn>
bool
poll_for(ReadyFn&& ready, replay_deadline_t::clock::duration timeout)
{
    using clock                  = replay_deadline_t::clock;
    constexpr auto poll_interval = std::chrono::milliseconds{2};
    const auto     poll_end      = clock::now() + timeout;
    for(;;)
    {
        if(ready()) return true;
        const auto now = clock::now();
        if(now >= poll_end) return false;
        std::this_thread::sleep_for(std::min<clock::duration>(poll_interval, poll_end - now));
    }
}

// `queue`'s ring when it takes AQL packets, else nullptr.
const hsa_queue_t*
writable_ring(const Queue& queue)
{
    if(queue.get_engine_type() != Queue::engine_type::compute) return nullptr;
    const auto* ring = queue.intercept_queue();
    return (ring == nullptr || ring->base_address == nullptr) ? nullptr : ring;
}

// Claims the next slot of `ring` only while one is free, so a caller that gives up has written
// nothing: a slot claimed with hsa_queue_add_write_index and never filled stalls the packet
// processor at its invalid header for good.
bool
try_reserve_slot(const CoreApiTable& core, const hsa_queue_t* ring, uint64_t& write_idx)
{
    write_idx = core.hsa_queue_load_write_index_relaxed_fn(ring);
    while(write_idx - core.hsa_queue_load_read_index_scacquire_fn(ring) < ring->size)
    {
        const auto seen = core.hsa_queue_cas_write_index_relaxed_fn(ring, write_idx, write_idx + 1);
        if(seen == write_idx) return true;
        write_idx = seen;
    }
    return false;
}

void
write_slot(const CoreApiTable&       core,
           const hsa_queue_t*        ring,
           uint64_t                  write_idx,
           const rocprofiler_packet& packet)
{
    const size_t offset = (write_idx % ring->size) * sizeof(rocprofiler_packet);
    // The ring buffer is a flat array of 64-byte packet slots addressed by index. Stepping a
    // byte pointer rather than casting the base address through an integer keeps this a
    // pointer-to-pointer cast, which is what it is.
    auto* slot = reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(ring->base_address) + offset);
    const auto* src = reinterpret_cast<const uint32_t*>(&packet);

    // Header last (release store): the packet processor must not see a slot whose header is
    // valid before its payload is.
    std::memcpy(&slot[1], &src[1], sizeof(rocprofiler_packet) - sizeof(uint32_t));
    reinterpret_cast<std::atomic<uint32_t>*>(slot)->store(src[0], std::memory_order_release);

    core.hsa_signal_store_screlease_fn(ring->doorbell_signal, write_idx);
}
}  // namespace

std::shared_mutex&
agent_replay_mutex(rocprofiler_agent_id_t agent_id)
{
    return agent_replay_state(agent_id).mutex;
}

std::atomic<uint64_t>&
agent_submit_epoch(rocprofiler_agent_id_t agent_id)
{
    return agent_replay_state(agent_id).submit_epoch;
}

void
replay_drain_or_fatal(const Queue& queue)
{
    replay_wait_or_fatal([&]() { return queue.sync(); },
                         "this queue's async completion handler(s)");
}

bool
replay_drain_agent(hsa_agent_t agent, const replay_deadline_t& deadline)
{
    auto* queue_controller = get_queue_controller();
    if(queue_controller == nullptr) return true;

    const auto drained = [&]() { return agent_async_in_flight(*queue_controller, agent) == 0; };
    return replay_wait_until([&](auto timeout) { return poll_for(drained, timeout); },
                             "this agent's async completion handler(s)",
                             deadline);
}

void
replay_drain_agent_or_fatal(hsa_agent_t agent)
{
    auto* queue_controller = get_queue_controller();
    if(queue_controller == nullptr) return;

    const auto drained = [&]() { return agent_async_in_flight(*queue_controller, agent) == 0; };
    if(replay_wait([&]() { return poll_for(drained, drain_slice); },
                   "this agent's async completion handler(s)"))
        return;
    ROCP_FATAL << fmt::format(
        "replay: agent-wide drain stuck (async handler(s) still active after ~{}s)",
        drain_budget_secs);
}

bool
replay_ring_submit(const Queue& queue, const rocprofiler_packet* packets, size_t count)
{
    if(count == 0) return true;
    const auto* ring = writable_ring(queue);
    if(ring == nullptr) return false;

    const auto& core = queue.core_api();
    for(size_t i = 0; i < count; ++i)
    {
        const uint64_t write_idx = core.hsa_queue_add_write_index_relaxed_fn(ring, 1);

        // The ring holds ring->size packets; a slot may only be written once the GPU has consumed
        // the packet `size` slots earlier.
        while((write_idx - core.hsa_queue_load_read_index_scacquire_fn(ring)) >= ring->size)
            std::this_thread::yield();

        write_slot(core, ring, write_idx, packets[i]);
    }

    return true;
}

barrier_wait_result_t
replay_barrier_and_wait(const Queue&             queue,
                        hsa_signal_t             completion,
                        const replay_deadline_t& deadline)
{
    const auto* ring = writable_ring(queue);
    if(ring == nullptr) return barrier_wait_result_t::not_submitted;

    const auto& core      = queue.core_api();
    auto        write_idx = uint64_t{0};
    const auto  reserved  = [&]() { return try_reserve_slot(core, ring, write_idx); };
    if(!replay_wait_until([&](auto timeout) { return poll_for(reserved, timeout); },
                          "a free slot in this queue's ring",
                          deadline))
        return barrier_wait_result_t::ring_full;

    auto barrier   = hsa_barrier_and_packet_t{};
    barrier.header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
    barrier.header |= 1 << HSA_PACKET_HEADER_BARRIER;
    barrier.completion_signal = completion;

    core.hsa_signal_store_screlease_fn(completion, 1);
    write_slot(core, ring, write_idx, rocprofiler_packet{barrier});

    const auto completed = replay_wait_until(
        [&](replay_deadline_t::clock::duration timeout) {
            return core.hsa_signal_wait_scacquire_fn(completion,
                                                     HSA_SIGNAL_CONDITION_EQ,
                                                     0,
                                                     std::chrono::nanoseconds{timeout}.count(),
                                                     HSA_WAIT_STATE_BLOCKED) == 0;
        },
        "this queue's prior GPU work",
        deadline);

    return completed ? barrier_wait_result_t::completed : barrier_wait_result_t::timed_out;
}

void
set_interceptor_passthrough(bool enabled)
{
    tl_passthrough = enabled;
}

bool
interceptor_passthrough()
{
    return tl_passthrough;
}
}  // namespace hsa
}  // namespace rocprofiler
