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

#include "lib/common/container/pool.hpp"
#include "lib/common/container/pool_object.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace
{
namespace container = ::rocprofiler::common::container;

struct payload
{
    uint64_t value = 0;
};

// equal to stable_vector's chunk size so that every batch appends a chunk to the outer
// chunk index, i.e. the allocation that acquire() must not read while another thread grows it
constexpr size_t batch_size    = 32;
constexpr size_t burst_rounds  = 32;
constexpr size_t burst_base    = 8;
constexpr size_t burst_step    = 2;
constexpr size_t min_churn_ops = 2000;
constexpr size_t clear_rounds  = 256;

constexpr uint64_t payload_sentinel = 0xc0ffee;

void
init_payload(payload& obj)
{
    obj.value = payload_sentinel;
}

// records which pool indices are currently checked out. A pool that hands the same index
// to two threads at once shows up here as a duplicate rather than as a silent corruption.
struct checkout_set
{
    void insert(size_t idx)
    {
        auto _lk = std::unique_lock<std::mutex>{m_mtx};
        if(!m_held.emplace(idx).second) m_duplicates++;
    }

    void erase(size_t idx)
    {
        auto _lk = std::unique_lock<std::mutex>{m_mtx};
        if(m_held.erase(idx) != 1) m_unknown++;
    }

    size_t size() const
    {
        auto _lk = std::unique_lock<std::mutex>{m_mtx};
        return m_held.size();
    }

    size_t duplicates() const { return m_duplicates.load(); }
    size_t unknown() const { return m_unknown.load(); }

private:
    mutable std::mutex         m_mtx        = {};
    std::unordered_set<size_t> m_held       = {};
    std::atomic<size_t>        m_duplicates = 0;
    std::atomic<size_t>        m_unknown    = 0;
};

size_t
worker_count()
{
    return std::max<size_t>(4, std::min<size_t>(std::thread::hardware_concurrency(), 8));
}
}  // namespace

// Growth mutates m_pool under m_pool_mtx while acquire() pops and indexes it under that same
// lock, so the two must be driven concurrently rather than in separate phases: the burst
// threads keep raising peak demand to force batch growth while the churn threads keep
// releasing.
TEST(common, pool_concurrent_acquire_release)
{
    container::pool<payload> _pool{std::piecewise_construct, batch_size, init_payload};

    auto _checkout       = checkout_set{};
    auto _failed_release = std::atomic<size_t>{0};
    auto _bursts_done    = std::atomic<bool>{false};

    auto _use_and_record = [&](container::pool_object<payload>& obj) {
        _checkout.insert(obj.index());
        obj.get().value = obj.index();
    };

    auto _drop_and_release = [&](container::pool_object<payload>& obj) {
        // the checkout record must be dropped before the object is handed back, else
        // another thread may legitimately re-acquire the index before this one returns
        _checkout.erase(obj.index());
        if(!obj.release()) _failed_release++;
    };

    auto _burst_worker = [&]() {
        auto _held = std::vector<container::pool_object<payload>*>{};
        for(size_t i = 0; i < burst_rounds; ++i)
        {
            auto _burst = burst_base + (i * burst_step);
            _held.clear();
            _held.reserve(_burst);
            for(size_t j = 0; j < _burst; ++j)
            {
                auto& _obj = _pool.acquire();
                _use_and_record(_obj);
                _held.emplace_back(&_obj);
            }
            for(auto* itr : _held)
                _drop_and_release(*itr);
        }
    };

    auto _churn_worker = [&]() {
        auto _ops = size_t{0};
        while(!_bursts_done.load(std::memory_order_relaxed) || _ops < min_churn_ops)
        {
            auto& _obj = _pool.acquire();
            _use_and_record(_obj);
            _drop_and_release(_obj);
            ++_ops;
        }
    };

    auto _nworker = worker_count();
    auto _nburst  = std::max<size_t>(2, _nworker / 2);
    auto _nchurn  = _nworker - _nburst;

    auto _threads = std::vector<std::thread>{};
    _threads.reserve(_nburst + _nchurn);
    for(size_t i = 0; i < _nchurn; ++i)
        _threads.emplace_back(_churn_worker);
    for(size_t i = 0; i < _nburst; ++i)
        _threads.emplace_back(_burst_worker);

    for(size_t i = 0; i < _nburst; ++i)
        _threads.at(_nchurn + i).join();
    _bursts_done.store(true, std::memory_order_relaxed);
    for(size_t i = 0; i < _nchurn; ++i)
        _threads.at(i).join();

    EXPECT_EQ(_checkout.duplicates(), 0) << "an index was checked out by two threads at once";
    EXPECT_EQ(_checkout.unknown(), 0) << "an index was returned without being checked out";
    EXPECT_EQ(_checkout.size(), 0) << "an acquired object was never released";
    EXPECT_EQ(_failed_release.load(), 0) << "release() rejected an object that was in use";

    EXPECT_GE(_pool.get_usage().batches, 2)
        << "the pool never grew, so the growth path went untested: " << _pool.get_usage_report();
}

// clear() holds m_pool_mtx across its loop and pool<Tp>::release() takes that lock, so clear()
// must not re-enter it: it clears the in-use flag directly rather than calling
// pool_object<Tp>::release(). A regression here deadlocks rather than failing.
TEST(common, pool_clear_with_object_in_use)
{
    container::pool<payload> _pool{std::piecewise_construct, batch_size, init_payload};

    // exceeding the initial batch grows the pool, so clear() has more than one batch to walk
    auto _held = std::vector<container::pool_object<payload>*>{};
    for(size_t i = 0; i < batch_size + 1; ++i)
        _held.emplace_back(&_pool.acquire());

    for(auto* itr : _held)
        EXPECT_TRUE(itr->release());

    // deliberately left checked out so that clear() has an in-use object to clear
    auto& _in_use = _pool.acquire();
    EXPECT_TRUE(_in_use.in_use());

    _pool.clear();
}

// clear() retires the storage instead of freeing it, so a pool_object<Tp>* handed out before
// the call stays addressable after it. Freeing instead makes every read below a use-after-free,
// which a sanitizer build reports; without one, the surviving signal is that the repopulated
// pool must not reuse the retired object's storage or its index.
TEST(common, pool_release_after_clear)
{
    container::pool<payload> _pool{std::piecewise_construct, batch_size, init_payload};

    auto* _held        = &_pool.acquire();
    auto  _idx         = _held->index();
    _held->get().value = payload_sentinel;

    _pool.clear();

    // retired, not freed: the object keeps its address, its index and its payload
    EXPECT_EQ(_held->index(), _idx);
    EXPECT_EQ(_held->get().value, payload_sentinel);

    // clear() cleared the in-use flag on every object it retired, so the late release reaches
    // pool<Tp>::release(), fails its exchange under m_pool_mtx and queues nothing
    EXPECT_FALSE(_held->in_use());
    EXPECT_FALSE(_held->release());
    EXPECT_EQ(_held->get().value, payload_sentinel);

    auto _usage  = _pool.get_usage();
    auto _report = _pool.get_usage_report();
    EXPECT_EQ(_usage.size, 0) << _report;
    EXPECT_EQ(_usage.available, 0) << _report;

    // the repopulated pool starts over at index zero: the same index as the retired object, and
    // necessarily a different object, since the retired storage is still alive
    auto& _fresh = _pool.acquire();
    EXPECT_EQ(_fresh.index(), _idx);
    EXPECT_NE(&_fresh, _held);
    EXPECT_TRUE(_fresh.release());
}

// the only test that drives clear() concurrently with acquire()/release(). acquire() takes
// m_pool_mtx before the free-list pop, so the pop and the stable_vector::at() that consumes the
// popped index are one critical section, and a lock-order inversion here deadlocks.
TEST(common, pool_clear_races_acquire_release)
{
    container::pool<payload> _pool{std::piecewise_construct, batch_size, init_payload};

    auto _clears_done   = std::atomic<bool>{false};
    auto _ops           = std::atomic<size_t>{0};
    auto _acquire_threw = std::atomic<size_t>{0};

    // counted rather than EXPECT_*'d here: the pool is driven from a second thread, and the
    // existing tests keep their assertions on the main thread for the same reason
    auto _churn_worker = [&]() {
        do
        {
            try
            {
                // a clear() landing here releases the object acquire() just handed out, so
                // neither in_use() nor the release() result below is assertable from this
                // thread; acquire()'s own exchange already succeeded before it returned
                auto& _obj       = _pool.acquire();
                _obj.get().value = _obj.index();
                _obj.release();
            } catch(const std::exception&)
            {
                _acquire_threw++;
            }
            _ops++;
        } while(!_clears_done.load(std::memory_order_relaxed));
    };

    auto _churn = std::thread{_churn_worker};

    for(size_t i = 0; i < clear_rounds; ++i)
    {
        _pool.clear();
        std::this_thread::yield();
    }

    _clears_done.store(true, std::memory_order_relaxed);
    _churn.join();

    EXPECT_EQ(_acquire_threw.load(), 0) << "acquire() indexed the pool with a stranded index";
    // the do/while guarantees one iteration, so a bound of zero would be vacuous
    EXPECT_GT(_ops.load(), clear_rounds) << "the churn thread barely ran";
}

// pool<Tp>::acquire(FuncT&&, Args&&...) runs the callable on every acquire, reused objects
// included, not only on the ones it had to create. An "optimization" to fire it only on newly
// created objects would silently break hsa::construct_hsa_signal's lazy-creation path.
TEST(common, pool_acquire_runs_ctor_on_reused_object)
{
    container::pool<payload> _pool{std::piecewise_construct, batch_size, init_payload};

    auto _ctor = [](payload& obj) { obj.value = payload_sentinel; };

    // drains the initial batch without growing it, so every object here is a first use
    auto _held = std::vector<container::pool_object<payload>*>{};
    for(size_t i = 0; i < batch_size; ++i)
        _held.emplace_back(&_pool.acquire(_ctor));

    for(auto* itr : _held)
        EXPECT_TRUE(itr->release());

    // zeroed so that the sentinel below can only have come from the callable running again
    for(auto* itr : _held)
        itr->get().value = 0;

    // and now every object is a reuse
    for(size_t i = 0; i < batch_size; ++i)
    {
        auto& _obj = _pool.acquire(_ctor);
        EXPECT_EQ(_obj.get().value, payload_sentinel) << "the callable did not run on a reuse";
        EXPECT_TRUE(_obj.release());
    }
}
