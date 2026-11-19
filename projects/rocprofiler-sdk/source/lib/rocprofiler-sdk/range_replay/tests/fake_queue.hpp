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

#pragma once

// GPU-free stand-ins for the parts of an HSA queue that the replay drain and fences read: an SDK
// Queue whose in-flight completion-handler count is a plain counter, registered with a
// QueueController like a non-compute queue, so the drain and destroy_queue run their real code
// without a runtime; and a compute queue whose ring is host memory, for the barrier a fence writes.

#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/hsa/signal_pool.hpp"

#include <rocprofiler-sdk/agent.h>

#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <type_traits>
#include <utility>

namespace rocprofiler
{
namespace range_replay
{
namespace test
{
// A signal's handle is the address of its counter.
inline std::atomic<hsa_signal_value_t>&
counter(hsa_signal_t signal)
{
    return *reinterpret_cast<std::atomic<hsa_signal_value_t>*>(signal.handle);
}

inline hsa_status_t
create_counter(hsa_signal_value_t initial, hsa_signal_t* signal)
{
    signal->handle = reinterpret_cast<uint64_t>(new std::atomic<hsa_signal_value_t>{initial});
    return HSA_STATUS_SUCCESS;
}

// Queue::sync and a fence's barrier wait for the counter to reach `expected`.
inline hsa_signal_value_t
wait_for_counter(hsa_signal_t signal,
                 hsa_signal_condition_t,
                 hsa_signal_value_t expected,
                 uint64_t           timeout_ns,
                 hsa_wait_state_t)
{
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::nanoseconds{timeout_ns};
    auto       value    = counter(signal).load();
    while(value != expected && std::chrono::steady_clock::now() < deadline)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
        value = counter(signal).load();
    }
    return value;
}

inline CoreApiTable
counter_signal_table()
{
    auto core = CoreApiTable{};
    core.hsa_signal_create_fn =
        [](hsa_signal_value_t initial, uint32_t, const hsa_agent_t*, hsa_signal_t* signal) {
            return create_counter(initial, signal);
        };
    core.hsa_signal_destroy_fn = [](hsa_signal_t signal) {
        delete &counter(signal);
        return HSA_STATUS_SUCCESS;
    };
    core.hsa_signal_add_relaxed_fn = [](hsa_signal_t signal, hsa_signal_value_t value) {
        counter(signal) += value;
    };
    core.hsa_signal_subtract_relaxed_fn = [](hsa_signal_t signal, hsa_signal_value_t value) {
        counter(signal) -= value;
    };
    core.hsa_signal_store_screlease_fn = [](hsa_signal_t signal, hsa_signal_value_t value) {
        counter(signal).store(value);
    };
    core.hsa_signal_load_scacquire_fn = [](hsa_signal_t signal) { return counter(signal).load(); };
    core.hsa_signal_wait_relaxed_fn   = wait_for_counter;
    core.hsa_signal_wait_scacquire_fn = wait_for_counter;
    return core;
}

// Queue::create_signal and the signal pool use the process-wide HSA tables rather than a queue's.
// Installs counter signals there. The pool keeps every signal it creates for the rest of the
// process.
inline void
install_process_signal_table()
{
    hsa::get_amd_ext_table()->hsa_amd_signal_create_fn = [](hsa_signal_value_t initial,
                                                            uint32_t,
                                                            const hsa_agent_t*,
                                                            uint64_t,
                                                            hsa_signal_t* signal) {
        return create_counter(initial, signal);
    };
    hsa::get_core_table()->hsa_signal_store_screlease_fn =
        [](hsa_signal_t signal, hsa_signal_value_t value) { counter(signal).store(value); };
}

// How many signals the pool can hand out before it has to create more. Requires
// install_process_signal_table().
inline size_t
free_pooled_signals()
{
    // The pool reports this count only in its usage report.
    const auto     report = hsa::get_signal_pool()->get_usage_report();
    constexpr auto key    = std::string_view{"available="};
    const auto     pos    = report.find(key);
    if(pos == std::string::npos) throw std::runtime_error{"no free count in: " + report};
    return std::stoul(report.substr(pos + key.size()));
}

// A queue's ring buffer, read and write indices and doorbell, in host memory. `queue` comes first
// so that ring_table()'s queue functions can get from the hsa_queue_t* they are handed back to
// the ring.
struct fake_ring_t
{
    static constexpr uint32_t size = 4;

    explicit fake_ring_t(uint64_t id)
    {
        queue.id                     = id;
        queue.size                   = size;
        queue.base_address           = packets.data();
        queue.doorbell_signal.handle = reinterpret_cast<uint64_t>(&doorbell);
    }

    // Every slot holds a packet the packet processor has not consumed, and none is consumed until
    // the test advances read_index.
    void fill() { write_index.store(read_index.load() + size); }

    hsa_queue_t                                queue       = {};
    std::atomic<uint64_t>                      write_index = {0};
    std::atomic<uint64_t>                      read_index  = {0};
    std::atomic<hsa_signal_value_t>            doorbell    = {-1};  // last index rung
    std::array<hsa_barrier_and_packet_t, size> packets     = {};
};

inline fake_ring_t&
ring_of(const hsa_queue_t* queue)
{
    static_assert(std::is_standard_layout_v<fake_ring_t>);
    static_assert(sizeof(hsa_barrier_and_packet_t) == 64);
    return *reinterpret_cast<fake_ring_t*>(const_cast<hsa_queue_t*>(queue));
}

// counter_signal_table() plus the queue-index functions a replay write into a fake_ring_t uses.
inline CoreApiTable
ring_table()
{
    auto core                                   = counter_signal_table();
    core.hsa_queue_load_read_index_scacquire_fn = [](const hsa_queue_t* queue) {
        return ring_of(queue).read_index.load();
    };
    core.hsa_queue_load_write_index_relaxed_fn = [](const hsa_queue_t* queue) {
        return ring_of(queue).write_index.load();
    };
    core.hsa_queue_add_write_index_relaxed_fn = [](const hsa_queue_t* queue, uint64_t value) {
        return ring_of(queue).write_index.fetch_add(value);
    };
    core.hsa_queue_cas_write_index_relaxed_fn =
        [](const hsa_queue_t* queue, uint64_t expected, uint64_t value) {
            ring_of(queue).write_index.compare_exchange_strong(expected, value);
            return expected;
        };
    return core;
}

// Consumes the packets rung on `ring`, each `delay` after it is reached, and decrements its
// completion signal: a barrier's signal, once all earlier work on the queue has finished.
class fake_packet_processor_t
{
public:
    fake_packet_processor_t(fake_ring_t& ring, std::chrono::milliseconds delay)
    : m_ring{&ring}
    , m_delay{delay}
    , m_thread{[this]() { run(); }}
    {}

    ~fake_packet_processor_t()
    {
        m_stop.store(true);
        m_thread.join();
    }

    fake_packet_processor_t(const fake_packet_processor_t&) = delete;
    fake_packet_processor_t& operator=(const fake_packet_processor_t&) = delete;

private:
    void run()
    {
        while(!m_stop.load())
        {
            const auto next = m_ring->read_index.load();
            if(m_ring->doorbell.load() < static_cast<hsa_signal_value_t>(next))
            {
                std::this_thread::sleep_for(std::chrono::milliseconds{1});
                continue;
            }
            std::this_thread::sleep_for(m_delay);
            const auto completion = m_ring->packets.at(next % fake_ring_t::size).completion_signal;
            if(completion.handle != 0) counter(completion) -= 1;
            m_ring->read_index.store(next + 1);
        }
    }

    fake_ring_t*              m_ring  = nullptr;
    std::chrono::milliseconds m_delay = {};
    std::atomic<bool>         m_stop  = {false};
    std::thread               m_thread;
};

// One GPU agent's AgentCache, built without a runtime.
struct test_agent_t
{
    explicit test_agent_t(uint64_t handle)
    {
        agent.id.handle = handle;
        agent.name      = "test";
        ext.hsa_amd_agent_iterate_memory_pools_fn =
            [](hsa_agent_t, hsa_status_t (*)(hsa_amd_memory_pool_t, void*), void*) {
                return HSA_STATUS_SUCCESS;
            };
        cache = std::make_unique<hsa::AgentCache>(
            &agent, hsa_agent_t{.handle = handle}, 0, hsa_agent_t{.handle = 0}, ext, core);
    }

    rocprofiler_agent_t              agent = {};
    AmdExtTable                      ext   = {};
    CoreApiTable                     core  = counter_signal_table();
    std::unique_ptr<hsa::AgentCache> cache = {};
};

// Queue's test constructor leaves the runtime queue null, and destroy_queue reads the queue id.
class test_queue_t : public hsa::Queue
{
public:
    test_queue_t(const test_agent_t& agent, uint64_t id)
    : hsa::Queue{*agent.cache, agent.core}
    , m_id{id}
    {}

    test_queue_t(const test_agent_t& agent, fake_ring_t& ring)
    : hsa::Queue{*agent.cache, ring_table(), &ring.queue}
    , m_id{ring.queue.id}
    {}

    rocprofiler_queue_id_t get_id() const override { return {.handle = m_id}; }

private:
    uint64_t m_id = 0;
};

// Registers a test queue with `controller`. The returned reference is valid until the queue is
// destroyed.
inline test_queue_t&
add_test_queue(hsa::QueueController& controller, hsa_queue_t& hsa_queue, const test_agent_t& agent)
{
    auto  queue = std::make_unique<test_queue_t>(agent, hsa_queue.id);
    auto& ref   = *queue;
    controller.add_queue(&hsa_queue, std::move(queue), /*is_compute=*/false);
    return ref;
}

// Registers a compute queue that writes into `ring`. Destroy it with
// controller.destroy_queue(&ring.queue).
inline test_queue_t&
add_ring_queue(hsa::QueueController& controller, fake_ring_t& ring, const test_agent_t& agent)
{
    auto  queue = std::make_unique<test_queue_t>(agent, ring);
    auto& ref   = *queue;
    controller.add_queue(&ring.queue, std::move(queue), /*is_compute=*/true);
    return ref;
}

// Destroys a queue on its own thread while one completion handler is in flight, so destroy_queue
// stops in the queue's sync until finish() retires that handler.
class in_flight_destroy_t
{
public:
    in_flight_destroy_t(hsa::QueueController& controller, hsa_queue_t& hsa_queue, hsa::Queue& queue)
    : m_queue{&queue}
    {
        queue.async_started();
        m_thread =
            std::thread{[&controller, &hsa_queue]() { controller.destroy_queue(&hsa_queue); }};
    }

    ~in_flight_destroy_t() { finish(); }

    in_flight_destroy_t(const in_flight_destroy_t&) = delete;
    in_flight_destroy_t& operator=(const in_flight_destroy_t&) = delete;

    // The queue is freed as soon as destroy_queue sees the handler retire.
    void finish()
    {
        if(m_queue != nullptr) std::exchange(m_queue, nullptr)->async_complete();
        if(m_thread.joinable()) m_thread.join();
    }

private:
    hsa::Queue* m_queue  = nullptr;
    std::thread m_thread = {};
};

// Whether `controller`'s queue map emptied within `timeout`. Shorter than one sync slice, so a
// destroy that is still waiting on a handler has not given up yet.
inline bool
wait_until_unmapped(const hsa::QueueController& controller,
                    std::chrono::milliseconds   timeout = std::chrono::seconds{4})
{
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while(std::chrono::steady_clock::now() < deadline)
    {
        auto mapped = 0;
        controller.iterate_queues([&mapped](const hsa::Queue*) { ++mapped; });
        if(mapped == 0) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    }
    return false;
}
}  // namespace test
}  // namespace range_replay
}  // namespace rocprofiler
