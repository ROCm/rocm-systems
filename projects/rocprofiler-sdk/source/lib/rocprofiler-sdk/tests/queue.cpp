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
#include "lib/rocprofiler-sdk/counters/tests/hsa_tables.hpp"
#include "lib/rocprofiler-sdk/hsa/agent_cache.hpp"

#include <gtest/gtest.h>

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/fwd.h>

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
    constexpr auto num_dispatch_threads  = size_t{8};
    constexpr auto dispatches_per_thread = size_t{20000};

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

    GTEST_LOG_(INFO) << "dispatches=" << dispatch_count.load()
                     << " exclusive_acquisitions=" << exclusive_acquisitions.load();
}
}  // namespace hsa
}  // namespace rocprofiler
