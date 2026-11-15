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

// GPU-free stand-ins for the parts of an HSA queue that the replay drain reads: an SDK Queue whose
// in-flight completion-handler count is a plain counter, registered with a QueueController like a
// non-compute queue, so the drain and destroy_queue run their real code without a runtime.

#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"

#include <rocprofiler-sdk/agent.h>

#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <thread>
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

inline CoreApiTable
counter_signal_table()
{
    auto core = CoreApiTable{};
    core.hsa_signal_create_fn =
        [](hsa_signal_value_t initial, uint32_t, const hsa_agent_t*, hsa_signal_t* signal) {
            signal->handle =
                reinterpret_cast<uint64_t>(new std::atomic<hsa_signal_value_t>{initial});
            return HSA_STATUS_SUCCESS;
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
    core.hsa_signal_load_scacquire_fn = [](hsa_signal_t signal) { return counter(signal).load(); };
    // Queue::sync waits for the counter to reach zero.
    core.hsa_signal_wait_relaxed_fn = [](hsa_signal_t signal,
                                         hsa_signal_condition_t,
                                         hsa_signal_value_t expected,
                                         uint64_t           timeout_ns,
                                         hsa_wait_state_t) {
        const auto deadline =
            std::chrono::steady_clock::now() + std::chrono::nanoseconds{timeout_ns};
        auto value = counter(signal).load();
        while(value != expected && std::chrono::steady_clock::now() < deadline)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds{1});
            value = counter(signal).load();
        }
        return value;
    };
    return core;
}

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
