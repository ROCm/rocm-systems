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
#include <mutex>
#include <string>
#include <string_view>
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
// chunk index, i.e. the allocation that release() must not read
constexpr size_t batch_size    = 32;
constexpr size_t burst_rounds  = 32;
constexpr size_t burst_base    = 8;
constexpr size_t burst_step    = 2;
constexpr size_t min_churn_ops = 2000;

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
usage_field(const std::string& report, std::string_view key)
{
    auto _needle = " " + std::string{key} + "=";
    auto _pos    = report.find(_needle);
    EXPECT_NE(_pos, std::string::npos) << "'" << key << "' missing from: " << report;
    if(_pos == std::string::npos) return 0;
    return std::stoull(report.substr(_pos + _needle.size()));
}

size_t
worker_count()
{
    return std::max<size_t>(4, std::min<size_t>(std::thread::hardware_concurrency(), 8));
}
}  // namespace

// Growth mutates m_pool under m_pool_mtx while release() runs unsynchronized with it, so
// the two must be driven concurrently rather than in separate phases: the burst threads
// keep raising peak demand to force batch growth while the churn threads keep releasing.
TEST(common, pool_concurrent_acquire_release)
{
    container::pool<payload> _pool{std::piecewise_construct, batch_size, init_payload};

    auto _checkout       = checkout_set{};
    auto _total_ops      = std::atomic<size_t>{0};
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
        auto _ops  = size_t{0};
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
            _ops += _burst;
        }
        _total_ops += _ops;
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
        _total_ops += _ops;
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

    auto _report    = _pool.get_usage_report();
    auto _size      = usage_field(_report, "size");
    auto _available = usage_field(_report, "available");
    auto _released  = usage_field(_report, "released");
    auto _batches   = usage_field(_report, "batches");

    EXPECT_GE(_batches, 2) << "the pool never grew, so the growth path went untested: " << _report;
    EXPECT_EQ(_size, batch_size * (_batches + 1)) << _report;
    EXPECT_EQ(_available, _size) << "every object must be back in the free list: " << _report;
    EXPECT_EQ(_released, _total_ops.load()) << _report;
}

// clear() releases every object while holding m_pool_mtx, so release() must reach the free
// list without taking that lock. A regression here deadlocks rather than failing.
TEST(common, pool_clear_with_object_in_use)
{
    container::pool<payload> _pool{std::piecewise_construct, batch_size, init_payload};

    auto _held = std::vector<container::pool_object<payload>*>{};
    for(size_t i = 0; i < batch_size + 1; ++i)
        _held.emplace_back(&_pool.acquire());

    // exceeding the initial batch grows the pool exactly once, and the per-object ctor
    // callback must have run for objects from both the initial batch and the new one
    EXPECT_EQ(usage_field(_pool.get_usage_report(), "batches"), 1);
    EXPECT_EQ(_held.front()->get().value, payload_sentinel);
    EXPECT_EQ(_held.back()->get().value, payload_sentinel);

    for(auto* itr : _held)
        EXPECT_TRUE(itr->release());

    // deliberately left checked out so that clear() calls back into release()
    auto& _in_use = _pool.acquire();
    EXPECT_TRUE(_in_use.in_use());

    _pool.clear();

    auto _report = _pool.get_usage_report();
    EXPECT_EQ(usage_field(_report, "size"), 0) << _report;
    EXPECT_EQ(usage_field(_report, "available"), 0) << _report;
    EXPECT_EQ(usage_field(_report, "released"), 0) << _report;
    EXPECT_EQ(usage_field(_report, "batches"), 0) << _report;

    // a cleared pool must repopulate on the next acquire
    auto& _obj = _pool.acquire();
    EXPECT_LT(_obj.index(), batch_size);
    EXPECT_TRUE(_obj.release());
}
