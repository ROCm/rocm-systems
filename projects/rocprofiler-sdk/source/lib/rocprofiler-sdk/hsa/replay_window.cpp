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

    for(;;)
    {
        int64_t in_flight = 0;
        queue_controller->iterate_queues([&](const Queue* sibling) {
            if(sibling != nullptr && sibling->get_agent().get_hsa_agent() == agent)
                in_flight += sibling->active_async_packets();
        });

        if(in_flight == 0) return;

        ROCP_FATAL_IF(std::chrono::steady_clock::now() >= deadline) << fmt::format(
            "replay: agent-wide drain stuck ({} async handler(s) still active after ~{}s)",
            in_flight,
            drain_budget_secs);

        std::this_thread::sleep_for(poll_interval);
    }
}

bool
replay_ring_submit(const Queue& queue, const rocprofiler_packet* packets, size_t count)
{
    if(count == 0) return true;

    const auto* hsa_queue = queue.intercept_queue();
    if(hsa_queue == nullptr || hsa_queue->base_address == nullptr) return false;

    const auto& core = queue.core_api();
    // The ring holds hsa_queue->size packets; a slot may only be written once the GPU has consumed
    // the packet `size` slots earlier.
    auto* mutable_queue =
        const_cast<hsa_queue_t*>(hsa_queue);  // NOLINT(cppcoreguidelines-pro-type-const-cast)

    for(size_t i = 0; i < count; ++i)
    {
        const uint64_t write_idx = core.hsa_queue_add_write_index_relaxed_fn(mutable_queue, 1);

        while((write_idx - core.hsa_queue_load_read_index_scacquire_fn(mutable_queue)) >=
              hsa_queue->size)
            std::this_thread::yield();

        const size_t offset = (write_idx % hsa_queue->size) * sizeof(rocprofiler_packet);
        // The ring buffer is a flat array of 64-byte packet slots addressed by index.
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        auto* slot =
            reinterpret_cast<uint32_t*>(reinterpret_cast<size_t>(hsa_queue->base_address) + offset);
        const auto* src = reinterpret_cast<const uint32_t*>(&packets[i]);

        // Header last (release store): the packet processor must not see a slot whose header is
        // valid before its payload is.
        std::memcpy(&slot[1], &src[1], sizeof(rocprofiler_packet) - sizeof(uint32_t));
        reinterpret_cast<std::atomic<uint32_t>*>(slot)->store(src[0], std::memory_order_release);

        core.hsa_signal_store_screlease_fn(hsa_queue->doorbell_signal, write_idx);
    }

    return true;
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
