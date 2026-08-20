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
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"

#include <rocprofiler-sdk/cxx/hash.hpp>
#include <rocprofiler-sdk/cxx/operators.hpp>

#include <fmt/format.h>
#include <hsa/hsa.h>

#include <chrono>
#include <thread>
#include <unordered_map>

namespace rocprofiler
{
namespace hsa
{
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
}  // namespace hsa
}  // namespace rocprofiler
