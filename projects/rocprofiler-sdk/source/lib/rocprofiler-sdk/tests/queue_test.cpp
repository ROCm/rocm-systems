// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk/hsa/queue.hpp"
#include "lib/common/environment.hpp"
#include "lib/rocprofiler-sdk/counters/tests/hsa_tables.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_interposition.hpp"

#include <gtest/gtest.h>

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/fwd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <mutex>
#include <shared_mutex>
#include <thread>
#include <vector>

namespace rocprofiler
{
namespace hsa
{
namespace
{
// Minimal Queue standing in for a real intercept queue: the lifetime lock is taken before
// WriteInterceptor touches the object, so the stub HSA tables never have to do real work.
class FakeQueue : public Queue
{
public:
    FakeQueue(const AgentCache& a, rocprofiler_queue_id_t id)
    : Queue(a, counters::test_constants::get_api_table())
    , _agent(a)
    , _id(id)
    {}

    virtual const AgentCache&      get_agent() const override final { return _agent; }
    virtual rocprofiler_queue_id_t get_id() const override final { return _id; }

private:
    const AgentCache&      _agent;
    rocprofiler_queue_id_t _id = {};
};

// Reports "no memory pools" rather than failing: AgentCache's pool init treats a SUCCESS
// return with no callback invocation as an empty pool set, whereas the real table without
// hsa_init() returns an error and trips ROCP_FATAL_IF.
const AmdExtTable&
stub_ext_table()
{
    static auto _v = []() {
        auto _t = AmdExtTable{};
        _t.hsa_amd_agent_iterate_memory_pools_fn =
            [](hsa_agent_t, hsa_status_t (*)(hsa_amd_memory_pool_t, void*), void*) {
                return HSA_STATUS_SUCCESS;
            };
        return _t;
    }();
    return _v;
}

const AgentCache&
fake_agent_cache()
{
    static auto _agent_info = []() {
        auto _v    = rocprofiler_agent_t{};
        _v.name    = "fake-gfx000";
        _v.node_id = 0;
        _v.type    = ROCPROFILER_AGENT_TYPE_GPU;
        return _v;
    }();

    static auto _cache = AgentCache{&_agent_info,
                                    hsa_agent_t{.handle = 1},
                                    0,
                                    hsa_agent_t{.handle = 2},
                                    stub_ext_table(),
                                    counters::test_constants::get_api_table()};
    return _cache;
}

void
null_writer(const void*, uint64_t)
{}

// Dispatch-storm shape, overridable so the stress can be cranked without a rebuild.
size_t
stress_threads(size_t _default)
{
    return std::max<size_t>(1, common::get_env("ROCPROFILER_TEST_STRESS_THREADS", _default));
}

size_t
stress_dispatches(size_t _default)
{
    return std::max<size_t>(1, common::get_env("ROCPROFILER_TEST_STRESS_DISPATCHES", _default));
}

// WriteInterceptor invokes the writer while still holding the shared lifetime lock
// (queue.cpp:329 under the lock taken at :317), so these counters observe the inside of the
// guarded region. The writer is a raw function pointer and cannot capture, hence file scope.
std::atomic<int>      dispatch_in_flight{0};
std::atomic<bool>     observed_overlap{false};
std::atomic<uint64_t> dispatch_count{0};

void
counting_writer(const void*, uint64_t)
{
    dispatch_in_flight.fetch_add(1, std::memory_order_acq_rel);
    std::this_thread::yield();
    dispatch_count.fetch_add(1, std::memory_order_relaxed);
    dispatch_in_flight.fetch_sub(1, std::memory_order_acq_rel);
}
}  // namespace

// Regression test for the Queue use-after-free on the HSA-intercept path. destroy_queue()
// takes queue_lifetime_mutex() exclusively around the erase that runs ~Queue(), so
// WriteInterceptor must not dereference its user_data Queue until that lock is released.
TEST(queue, write_interceptor_blocks_while_queue_lifetime_lock_is_held)
{
    auto _queue = FakeQueue{fake_agent_cache(), rocprofiler_queue_id_t{.handle = 1}};

    auto packets      = std::array<hsa_kernel_dispatch_packet_t, 1>{};
    packets[0].header = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE);

    auto _destroying = std::unique_lock{queue_lifetime_mutex()};

    std::atomic<bool> done{false};
    std::thread       worker([&]() {
        _queue.invoke_write_interceptor(packets.data(), packets.size(), null_writer);
        done.store(true, std::memory_order_release);
    });

    std::this_thread::sleep_for(std::chrono::milliseconds{200});
    EXPECT_FALSE(done.load(std::memory_order_acquire))
        << "WriteInterceptor dereferenced its user_data Queue while queue_lifetime_mutex() "
           "was held exclusively; a concurrent ~Queue() could free it mid-call";

    _destroying.unlock();

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{3};
    while(!done.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds{1});
    const bool completed = done.load(std::memory_order_acquire);
    worker.join();

    ASSERT_TRUE(completed) << "WriteInterceptor did not resume after the exclusive "
                              "queue_lifetime_mutex() was released";
}

// Stress the same invariant under dispatch-storm conditions: many concurrent interceptor
// calls on one shared Queue while destroy_queue()'s exclusive section repeatedly cuts in.
// The writer runs inside the shared lock, so a writer observed while the exclusive lock is
// held would mean the two sections overlapped and ~Queue() could free the object mid-call.
TEST(queue, write_interceptor_stress_under_concurrent_destroy)
{
    const auto num_dispatch_threads  = stress_threads(8);
    const auto dispatches_per_thread = stress_dispatches(20000);

    auto _queue = FakeQueue{fake_agent_cache(), rocprofiler_queue_id_t{.handle = 1}};

    auto packets      = std::array<hsa_kernel_dispatch_packet_t, 1>{};
    packets[0].header = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE);

    dispatch_in_flight.store(0);
    observed_overlap.store(false);
    dispatch_count.store(0);

    std::atomic<bool> stop_destroyer{false};
    std::atomic<int>  exclusive_acquisitions{0};

    // stands in for QueueController::destroy_queue()'s critical section
    std::thread destroyer([&]() {
        while(!stop_destroyer.load(std::memory_order_acquire))
        {
            auto _lifetime = std::unique_lock{queue_lifetime_mutex()};
            if(dispatch_in_flight.load(std::memory_order_acquire) != 0)
                observed_overlap.store(true, std::memory_order_release);
            std::this_thread::yield();
            if(dispatch_in_flight.load(std::memory_order_acquire) != 0)
                observed_overlap.store(true, std::memory_order_release);
            exclusive_acquisitions.fetch_add(1, std::memory_order_relaxed);
        }
    });

    auto dispatchers = std::vector<std::thread>{};
    dispatchers.reserve(num_dispatch_threads);
    for(size_t i = 0; i < num_dispatch_threads; ++i)
    {
        dispatchers.emplace_back([&]() {
            for(size_t n = 0; n < dispatches_per_thread; ++n)
                _queue.invoke_write_interceptor(packets.data(), packets.size(), counting_writer);
        });
    }

    for(auto& itr : dispatchers)
        itr.join();
    stop_destroyer.store(true, std::memory_order_release);
    destroyer.join();

    EXPECT_FALSE(observed_overlap.load())
        << "a WriteInterceptor call was in flight while queue_lifetime_mutex() was held "
           "exclusively: the shared and exclusive sections overlapped";
    EXPECT_EQ(dispatch_count.load(), num_dispatch_threads * dispatches_per_thread);
    EXPECT_GT(exclusive_acquisitions.load(), 0)
        << "the destroyer never acquired the exclusive lock, so nothing was contended";

    GTEST_LOG_(INFO) << "\n"
                     << "  dispatch threads       = " << num_dispatch_threads << "\n"
                     << "  dispatches/thread      = " << dispatches_per_thread << "\n"
                     << "  dispatches total       = " << dispatch_count.load() << "\n"
                     << "  exclusive_acquisitions = " << exclusive_acquisitions.load();
}

// A real QueueController::destroy_queue() -- which runs the real ~Queue() under the
// exclusive lock -- racing real doorbell traffic, so the destroy path itself is covered and
// not just a stand-in critical section. Liveness and integrity only: with no active tracing
// context write_interceptor() returns before it ever dereferences the queue, so this cannot
// observe a use-after-free even with the lock removed (confirmed under ASan). The blocking
// and overlap tests remain the guards for the lock itself.
TEST(queue, real_destroy_queue_races_doorbell_lookup)
{
    const auto     num_producers        = stress_threads(4);
    const auto     dispatches_per_queue = uint64_t{stress_dispatches(4000)};
    constexpr auto ring_slots           = uint64_t{256};

    auto* qc = get_queue_controller();
    ASSERT_NE(qc, nullptr);

    // one HSA queue id, repeatedly registered and destroyed under the producers' feet
    auto hsa_queue = hsa_queue_t{};
    hsa_queue.id   = 4242;

    auto register_queue = [&]() {
        qc->add_queue(
            &hsa_queue,
            std::make_unique<FakeQueue>(fake_agent_cache(), rocprofiler_queue_id_t{.handle = 4242}),
            /*is_compute=*/false,
            /*is_attach=*/false);
    };
    register_queue();

    struct producer_ring
    {
        std::shared_ptr<queue_interposition::QueueState> state =
            std::make_shared<queue_interposition::QueueState>();
        std::vector<char> ring      = std::vector<char>(64 * ring_slots, 0);
        uint64_t          real_wdid = 0;
        uint64_t          real_rdid = 0;
    };

    auto producers = std::vector<std::unique_ptr<producer_ring>>{};
    producers.reserve(num_producers);
    for(size_t p = 0; p < num_producers; ++p)
    {
        auto _v              = std::make_unique<producer_ring>();
        _v->state->ring_buf  = _v->ring.data();
        _v->state->ring_size = ring_slots;
        _v->state->ring_mask = ring_slots - 1;
        _v->state->real_wdid = &_v->real_wdid;
        _v->state->real_rdid = &_v->real_rdid;
        _v->state->hsa_queue = &hsa_queue;  // every lookup targets the cycled Queue
        producers.emplace_back(std::move(_v));
    }

    std::atomic<bool> stop_destroyer{false};
    std::atomic<int>  destroy_cycles{0};

    std::thread destroyer([&]() {
        while(!stop_destroyer.load(std::memory_order_acquire))
        {
            qc->destroy_queue(&hsa_queue);  // real ~Queue() under the exclusive lock
            register_queue();
            destroy_cycles.fetch_add(1, std::memory_order_relaxed);
        }
    });

    auto workers = std::vector<std::thread>{};
    workers.reserve(num_producers);
    for(size_t p = 0; p < num_producers; ++p)
    {
        workers.emplace_back([&, p]() {
            auto* pr = producers[p].get();
            for(uint64_t n = 0; n < dispatches_per_queue; ++n)
            {
                auto* pkt = &reinterpret_cast<hsa_kernel_dispatch_packet_t*>(
                    pr->ring.data())[n & (ring_slots - 1)];
                pkt->kernel_object = 0xE000 + n;
                __atomic_store_n(&pkt->header,
                                 static_cast<uint16_t>(HSA_PACKET_TYPE_KERNEL_DISPATCH
                                                       << HSA_PACKET_HEADER_TYPE),
                                 __ATOMIC_RELEASE);
                pr->state->virtual_wptr.store(n + 1, std::memory_order_release);

                queue_interposition::process_doorbell_impl(
                    pr->state, n, [](hsa_signal_t, hsa_signal_value_t) {});

                __atomic_store_n(&pr->real_rdid, n + 1, __ATOMIC_RELEASE);
            }
        });
    }

    for(auto& itr : workers)
        itr.join();
    stop_destroyer.store(true, std::memory_order_release);
    destroyer.join();
    qc->destroy_queue(&hsa_queue);

    EXPECT_GT(destroy_cycles.load(), 0)
        << "no queue was ever destroyed, so the lifetime race was never exercised";

    // published either through the interceptor (queue found) or the passthrough (erased);
    // both routes must still publish every packet exactly once
    for(size_t p = 0; p < num_producers; ++p)
    {
        EXPECT_EQ(producers[p]->real_wdid, dispatches_per_queue) << "producer " << p;
        EXPECT_EQ(producers[p]->state->next_submit_pos, dispatches_per_queue) << "producer " << p;
    }

    GTEST_LOG_(INFO) << "\n"
                     << "  producers              = " << num_producers << "\n"
                     << "  dispatches/producer    = " << dispatches_per_queue << "\n"
                     << "  dispatches total       = " << num_producers * dispatches_per_queue
                     << "\n"
                     << "  destroy_cycles         = " << destroy_cycles.load();
}
}  // namespace hsa
}  // namespace rocprofiler
