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
#include "lib/rocprofiler-sdk/agent.hpp"
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
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>

namespace rocprofiler
{
namespace hsa
{
namespace
{
struct copy_tickets_t
{
    std::mutex         mutex     = {};
    uint64_t           next      = 0;
    std::set<uint64_t> in_flight = {};
};

copy_tickets_t&
copy_tickets(rocprofiler_agent_id_t agent_id)
{
    using map_t       = std::unordered_map<rocprofiler_agent_id_t, copy_tickets_t>;
    static auto*& map = common::static_object<common::Synchronized<map_t>>::construct();
    return map->wlock([](map_t& _map, auto _id) -> copy_tickets_t& { return _map[_id]; }, agent_id);
}

// True while a copy registered before ticket `limit` is still in flight.
bool
copies_before(rocprofiler_agent_id_t agent_id, uint64_t limit)
{
    auto&      tickets = copy_tickets(agent_id);
    const auto lock    = std::lock_guard<std::mutex>{tickets.mutex};
    return !tickets.in_flight.empty() && *tickets.in_flight.begin() < limit;
}

uint64_t
copy_ticket_limit(rocprofiler_agent_id_t agent_id)
{
    auto&      tickets = copy_tickets(agent_id);
    const auto lock    = std::lock_guard<std::mutex>{tickets.mutex};
    return tickets.next;
}

// The ring holds hsa_queue->size packets; a slot may only be written once the GPU has consumed the
// packet `size` slots earlier.
template <typename CoreTableT>
void
write_ring(const CoreTableT&         core,
           hsa_queue_t*              hsa_queue,
           const rocprofiler_packet* packets,
           size_t                    count)
{
    for(size_t i = 0; i < count; ++i)
    {
        const uint64_t write_idx = core.hsa_queue_add_write_index_relaxed_fn(hsa_queue, 1);

        while((write_idx - core.hsa_queue_load_read_index_scacquire_fn(hsa_queue)) >=
              hsa_queue->size)
            std::this_thread::yield();

        const size_t offset = (write_idx % hsa_queue->size) * sizeof(rocprofiler_packet);
        // The ring buffer is a flat array of 64-byte packet slots addressed by index. Stepping a
        // byte pointer rather than casting the base address through an integer keeps this a
        // pointer-to-pointer cast, which is what it is.
        auto* slot =
            reinterpret_cast<uint32_t*>(static_cast<uint8_t*>(hsa_queue->base_address) + offset);
        const auto* src = reinterpret_cast<const uint32_t*>(&packets[i]);

        // Header last (release store): the packet processor must not see a slot whose header is
        // valid before its payload is.
        std::memcpy(&slot[1], &src[1], sizeof(rocprofiler_packet) - sizeof(uint32_t));
        reinterpret_cast<std::atomic<uint32_t>*>(slot)->store(src[0], std::memory_order_release);

        core.hsa_signal_store_screlease_fn(hsa_queue->doorbell_signal, write_idx);
    }
}

hsa_queue_t*
private_queue(hsa_agent_t agent)
{
    using map_t       = std::unordered_map<uint64_t, hsa_queue_t*>;
    static auto*& map = common::static_object<common::Synchronized<map_t>>::construct();

    return map->wlock([agent](map_t& _map) -> hsa_queue_t* {
        if(auto itr = _map.find(agent.handle); itr != _map.end()) return itr->second;

        // The internal table is the unwrapped one, so this is a plain hardware queue: the SDK does
        // not intercept it and ROCr takes no intercept lock on its doorbell. It lives until HSA
        // shuts down.
        auto* core = get_core_table();
        if(core == nullptr || core->hsa_queue_create_fn == nullptr) return nullptr;

        constexpr uint32_t preferred_size = 4096;
        auto               max_size       = uint32_t{0};
        if(core->hsa_agent_get_info_fn(agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &max_size) !=
               HSA_STATUS_SUCCESS ||
           max_size == 0)
            return nullptr;

        hsa_queue_t* queue = nullptr;
        if(core->hsa_queue_create_fn(agent,
                                     std::min(preferred_size, max_size),
                                     HSA_QUEUE_TYPE_MULTI,
                                     nullptr,
                                     nullptr,
                                     UINT32_MAX,
                                     UINT32_MAX,
                                     &queue) != HSA_STATUS_SUCCESS)
            queue = nullptr;

        _map.emplace(agent.handle, queue);
        return queue;
    });
}
}  // namespace

std::shared_mutex&
agent_replay_mutex(rocprofiler_agent_id_t agent_id)
{
    // No get_fini_status() guard is needed here. Every caller reaches this only when a replay
    // service is active, and that is false during finalization, so the static lock map below is
    // never touched after teardown.
    using lock_map_t    = std::unordered_map<rocprofiler_agent_id_t, std::shared_mutex>;
    static auto*& locks = common::static_object<common::Synchronized<lock_map_t>>::construct();

    return locks->wlock([](lock_map_t& _map, auto _id) -> std::shared_mutex& { return _map[_id]; },
                        agent_id);
}

void
replay_drain_or_fatal(const Queue& queue)
{
    replay_wait_or_fatal([&]() { return queue.sync(); },
                         "this queue's async completion handler(s)");
}

void
replay_drain_agent_or_fatal(hsa_agent_t agent)
{
    auto* queue_controller = get_queue_controller();
    if(queue_controller == nullptr) return;

    constexpr auto poll_interval = std::chrono::milliseconds{2};
    constexpr auto max_wait      = std::chrono::seconds{drain_budget_secs};
    const auto     deadline      = std::chrono::steady_clock::now() + max_wait;

    const auto* rocp_agent = agent::get_rocprofiler_agent(agent);
    const auto  copy_limit = rocp_agent ? copy_ticket_limit(rocp_agent->id) : uint64_t{0};

    for(;;)
    {
        int64_t in_flight = 0;
        queue_controller->iterate_queues([&](const Queue* sibling) {
            if(sibling != nullptr && sibling->get_agent().get_hsa_agent() == agent)
                in_flight += sibling->active_async_packets();
        });
        const bool copies = rocp_agent != nullptr && copies_before(rocp_agent->id, copy_limit);

        if(in_flight == 0 && !copies) return;

        ROCP_FATAL_IF(std::chrono::steady_clock::now() >= deadline) << fmt::format(
            "replay: agent-wide drain stuck ({} async handler(s){} still active after ~{}s)",
            in_flight,
            copies ? " and device-writing copies" : "",
            drain_budget_secs);

        std::this_thread::sleep_for(poll_interval);
    }
}

bool
replay_ring_submit(const Queue& queue, const rocprofiler_packet* packets, size_t count)
{
    if(count == 0) return true;
    if(queue.get_engine_type() != Queue::engine_type::compute) return false;

    const auto* hsa_queue = queue.intercept_queue();
    if(hsa_queue == nullptr || hsa_queue->base_address == nullptr) return false;

    write_ring(
        queue.core_api(),
        const_cast<hsa_queue_t*>(hsa_queue),  // NOLINT(cppcoreguidelines-pro-type-const-cast)
        packets,
        count);
    return true;
}

bool
replay_private_submit(hsa_agent_t agent, const rocprofiler_packet* packets, size_t count)
{
    if(count == 0) return true;

    auto* hsa_queue = private_queue(agent);
    if(hsa_queue == nullptr) return false;

    write_ring(*get_core_table(), hsa_queue, packets, count);
    return true;
}

uint64_t
replay_copy_submitted(rocprofiler_agent_id_t agent_id)
{
    auto&      tickets = copy_tickets(agent_id);
    const auto lock    = std::lock_guard<std::mutex>{tickets.mutex};
    const auto ticket  = tickets.next++;
    tickets.in_flight.emplace(ticket);
    return ticket;
}

void
replay_copy_retired(rocprofiler_agent_id_t agent_id, uint64_t ticket)
{
    auto&      tickets = copy_tickets(agent_id);
    const auto lock    = std::lock_guard<std::mutex>{tickets.mutex};
    tickets.in_flight.erase(ticket);
}

bool
replay_barrier_and_wait(const Queue& queue, hsa_signal_t completion)
{
    using namespace std::chrono_literals;

    auto barrier   = hsa_barrier_and_packet_t{};
    barrier.header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
    barrier.header |= 1 << HSA_PACKET_HEADER_BARRIER;
    barrier.completion_signal = completion;

    auto packet = rocprofiler_packet{barrier};

    const auto& core = queue.core_api();
    core.hsa_signal_store_screlease_fn(completion, 1);

    if(!replay_ring_submit(queue, &packet, 1)) return false;

    replay_wait_or_fatal(
        [&]() {
            return core.hsa_signal_wait_scacquire_fn(completion,
                                                     HSA_SIGNAL_CONDITION_EQ,
                                                     0,
                                                     std::chrono::nanoseconds{5s}.count(),
                                                     HSA_WAIT_STATE_BLOCKED) == 0;
        },
        "this queue's prior GPU work");

    return true;
}

}  // namespace hsa
}  // namespace rocprofiler
