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
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "lib/rocprofiler-sdk/hsa/hsa_barrier.hpp"
#include "lib/rocprofiler-sdk/agent.hpp"
#include "lib/rocprofiler-sdk/context/context.hpp"
#include "lib/rocprofiler-sdk/counters/tests/hsa_tables.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

#include <gtest/gtest.h>
#include <algorithm>
#include <random>

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/registration.h>

using namespace rocprofiler;
using namespace rocprofiler::hsa;
using namespace rocprofiler::counters::test_constants;

namespace
{
namespace rocprofiler
{
namespace hsa
{
class FakeQueue : public Queue
{
public:
    FakeQueue(const AgentCache& a, rocprofiler_queue_id_t id)
    : Queue(a, get_api_table())
    , _agent(a)
    , _id(id)
    {}
    virtual const AgentCache&      get_agent() const override final { return _agent; };
    virtual rocprofiler_queue_id_t get_id() const override final { return _id; };

    ~FakeQueue() {}

private:
    const AgentCache&      _agent;
    rocprofiler_queue_id_t _id = {};
};

}  // namespace hsa
}  // namespace rocprofiler

QueueController::queue_map_t
create_queue_map(size_t count)
{
    QueueController::queue_map_t ret;

    // ensure test fails if null
    EXPECT_TRUE(hsa::get_queue_controller() != nullptr);

    // prevent segfault
    if(!hsa::get_queue_controller()) return ret;

    auto agents = hsa::get_queue_controller()->get_supported_agents();

    for(size_t i = 0; i < count; i++)
    {
        auto& agent_cache = agents.begin()->second;
        // Create queue
        hsa_queue_t* queue;
        hsa_queue_create(agent_cache.get_hsa_agent(),
                         2048,
                         HSA_QUEUE_TYPE_SINGLE,
                         nullptr,
                         nullptr,
                         0,
                         0,
                         &queue);
        ret[queue] = std::make_unique<rocprofiler::hsa::FakeQueue>(
            agent_cache, rocprofiler_queue_id_t{.handle = i});
    }

    return ret;
}

std::atomic<bool> should_execute_handler{false};
std::atomic<int>  executed_handlers{0};
bool
barrier_signal_handler(hsa_signal_value_t, void* data)
{
    CHECK(data);
    CHECK(should_execute_handler) << "Signal handler called when it should not have been";
    hsa_signal_destroy(*static_cast<hsa_signal_t*>(data));
    delete static_cast<hsa_signal_t*>(data);
    executed_handlers++;
    return false;
}

// Injects a barrier packet into the queue followed by a packet with an async handler
// associated with it. If the barrier is not released, the async handler should not
// be executed (checked with should_execute_handler).
void
inject_barriers(hsa_barrier& barrier, QueueController::queue_map_t& queues)
{
    auto packet_store_release = [](uint32_t* packet, uint16_t header, uint16_t rest) {
        __atomic_store_n(packet, header | (rest << 16), __ATOMIC_RELEASE);
    };

    auto header_pkt = [](hsa_packet_type_t type) {
        uint16_t header = type << HSA_PACKET_HEADER_TYPE;
        header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE;
        header |= HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE;
        return header;
    };

    auto enqueue_pkt = [&](auto& queue, auto& packets, auto& pkt) {
        uint64_t packet_id = hsa_queue_add_write_index_screlease(queue, 1);
        while(packet_id - hsa_queue_load_read_index_scacquire(queue) >= queue->size)
            ;
        hsa_barrier_and_packet_t* packet = packets + packet_id % queue->size;
        (*packet)                        = pkt;
        packet_store_release((uint32_t*) packet, header_pkt(HSA_PACKET_TYPE_BARRIER_AND), 0);
        hsa_signal_store_screlease(queue->doorbell_signal, packet_id);
    };

    for(auto& [hsa_queue, fq] : queues)
    {
        auto complete = barrier.complete();
        auto _pkt     = barrier.enqueue_packet(fq.get());
        // If barrier is complete, no packets should be generated
        ASSERT_NE(complete, _pkt.has_value());

        hsa_barrier_and_packet_t* _packets = (hsa_barrier_and_packet_t*) hsa_queue->base_address;
        if(_pkt.has_value())
        {
            enqueue_pkt(hsa_queue, _packets, _pkt->barrier_and);
        }

        // Construct packet that will trigger async handler after barrier is released
        rocprofiler_packet post_barrier{};
        hsa_signal_t*      completion_signal = new hsa_signal_t;
        hsa_signal_create(1, 0, nullptr, completion_signal);
        post_barrier.barrier_and.header = HSA_PACKET_TYPE_BARRIER_AND << HSA_PACKET_HEADER_TYPE;
        post_barrier.barrier_and.completion_signal = *completion_signal;
        hsa_amd_signal_async_handler(*completion_signal,
                                     HSA_SIGNAL_CONDITION_EQ,
                                     0,
                                     barrier_signal_handler,
                                     static_cast<void*>(completion_signal));
        enqueue_pkt(hsa_queue, _packets, post_barrier.barrier_and);
    }

    // Ensure that the barrier packet is reached on all queues
    usleep(100);
}

void
test_init()
{
    HsaApiTable table;
    table.amd_ext_ = &get_ext_table();
    table.core_    = &get_api_table();
    agent::construct_agent_cache(&table);
    ASSERT_TRUE(hsa::get_queue_controller() != nullptr);
    hsa::get_queue_controller()->init(get_api_table(), get_ext_table());
}
}  // namespace

TEST(hsa_barrier, no_block_single)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);

    bool complete      = false;
    auto finished_func = [&]() { complete = true; };

    auto queues = create_queue_map(1);

    // Immediate return of barrier due to no active async packets
    hsa::hsa_barrier             barrier(finished_func, get_api_table());
    hsa_barrier::queue_map_ptr_t q_map;
    for(const auto& [k, v] : queues)
    {
        q_map[k] = v.get();
    }

    barrier.set_barrier(q_map);
    executed_handlers = 0;
    ASSERT_TRUE(barrier.complete());
    should_execute_handler = true;
    inject_barriers(barrier, queues);
    ASSERT_EQ(complete, true);
    while(executed_handlers != 1)
    {
        usleep(10);
    }

    registration::set_init_status(1);
    registration::finalize();
}

TEST(hsa_barrier, no_block_multi)
{
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);

    bool complete      = false;
    auto finished_func = [&]() { complete = true; };

    auto queues = create_queue_map(10);

    // Immediate return of barrier due to no active async packets
    hsa::hsa_barrier barrier(finished_func, get_api_table());

    hsa_barrier::queue_map_ptr_t q_map;
    for(const auto& [k, v] : queues)
    {
        q_map[k] = v.get();
    }

    barrier.set_barrier(q_map);
    ASSERT_TRUE(barrier.complete());
    should_execute_handler = true;
    executed_handlers      = 0;
    inject_barriers(barrier, queues);
    ASSERT_EQ(complete, true);
    while(executed_handlers != 10)
    {
        usleep(10);
    }

    registration::set_init_status(1);
    registration::finalize();
}

TEST(hsa_barrier, block_single)
{
    std::vector<Queue*> pkt_waiting;
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);

    bool complete      = false;
    auto finished_func = [&]() { complete = true; };

    auto queues = create_queue_map(1);

    hsa::hsa_barrier barrier(finished_func, get_api_table());

    // Simulate waiting on packets already in the queue to complete
    for(auto& [_, queue] : queues)
    {
        pkt_waiting.push_back(queue.get());
        queue->async_started();
    }
    should_execute_handler = false;
    executed_handlers      = 0;

    hsa_barrier::queue_map_ptr_t q_map;
    for(const auto& [k, v] : queues)
    {
        q_map[k] = v.get();
    }

    barrier.set_barrier(q_map);
    ASSERT_FALSE(barrier.complete());

    should_execute_handler = false;
    executed_handlers      = 0;
    inject_barriers(barrier, queues);

    ASSERT_EQ(complete, false);
    should_execute_handler = true;

    for(auto& queue : pkt_waiting)
    {
        queue->async_complete();
        barrier.register_completion(queue);
    }

    ASSERT_EQ(complete, true);
    // Wait for the signal handlers to execute. If we deadlock here,
    // we are not triggering the completion of the signal handler.
    while(executed_handlers != 1)
    {
        usleep(100);
    }

    registration::set_init_status(1);
    registration::finalize();
}

TEST(hsa_barrier, block_multi)
{
    std::vector<Queue*> pkt_waiting;
    ASSERT_EQ(hsa_init(), HSA_STATUS_SUCCESS);
    test_init();

    registration::init_logging();
    registration::set_init_status(-1);
    context::push_client(1);

    bool complete      = false;
    auto finished_func = [&]() { complete = true; };

    auto queues = create_queue_map(10);

    // Immediate return of barrier due to no active async packets
    hsa::hsa_barrier barrier(finished_func, get_api_table());

    // Simulate waiting on packets already in the queue to complete
    for(auto& [_, queue] : queues)
    {
        for(size_t i = 0; i < 30; i++)
        {
            pkt_waiting.push_back(queue.get());
            queue->async_started();
        }
    }
    should_execute_handler = false;
    executed_handlers      = 0;

    hsa_barrier::queue_map_ptr_t q_map;
    for(const auto& [k, v] : queues)
    {
        q_map[k] = v.get();
    }

    barrier.set_barrier(q_map);
    ASSERT_FALSE(barrier.complete());

    should_execute_handler = false;
    executed_handlers      = 0;
    inject_barriers(barrier, queues);

    ASSERT_EQ(complete, false);

    auto rng = std::default_random_engine{};
    std::shuffle(std::begin(pkt_waiting), std::end(pkt_waiting), rng);
    for(size_t i = 0; i < pkt_waiting.size(); i++)
    {
        ASSERT_EQ(complete, false);
        ASSERT_FALSE(barrier.complete());
        if(i == pkt_waiting.size() - 1)
        {
            should_execute_handler = true;
        }
        pkt_waiting[i]->async_complete();
        barrier.register_completion(pkt_waiting[i]);
    }

    ASSERT_EQ(complete, true);
    // Wait for the signal handlers to execute. If we deadlock here,
    // we are not triggering the completion of the signal handler.
    while(executed_handlers != 10)
    {
        usleep(100);
    }

    registration::set_init_status(1);
    registration::finalize();
}
