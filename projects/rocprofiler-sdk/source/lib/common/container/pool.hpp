// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
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

#pragma once

#include "lib/common/container/pool_object.hpp"
#include "lib/common/container/stable_vector.hpp"
#include "lib/common/defines.hpp"
#include "lib/common/demangle.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/mpl.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace rocprofiler
{
namespace common
{
namespace container
{
template <typename Tp>
struct pool
{
    using size_type          = size_t;
    using pool_array_type    = stable_vector<pool_object<Tp>, 32>;
    using retired_array_type = std::vector<pool_array_type>;

    template <typename FuncT, typename... Args>
    explicit pool(std::piecewise_construct_t, size_type count, FuncT&& ctor, Args&&... args);

    ~pool()               = default;
    pool(const pool&)     = delete;
    pool(pool&&) noexcept = delete;
    pool& operator=(const pool&) = delete;
    pool& operator=(pool&&) noexcept = delete;

    // get an object from the pool. if all objects are in use, a new one will be created and added
    // to the pool
    pool_object<Tp>& acquire();
    bool             release(pool_object<Tp>& obj);

    template <typename FuncT, typename... Args>
    pool_object<Tp>& acquire(FuncT&& ctor, Args&&... args);

    // retire every object in the pool, running func on each one before it is retired.
    //
    // func runs with the pool lock held -- the same lock release() takes -- so it must not call
    // acquire(), release() or pool_object<Tp>::release(), not even on the object it was handed.
    // Any of the three self-deadlocks. Read the object and destroy what it owns; that is all
    // this callback is for. clear() itself clears the in-use flag directly rather than calling
    // pool_object<Tp>::release() for exactly this reason.
    //
    // A violation hangs rather than fails, so it stalls CI on a timeout instead of reporting
    // anything, which is why the rule is stated here rather than left to be discovered.
    template <typename FuncT = void (*)(pool_object<Tp>&)>
    void clear(FuncT&& func = [](pool_object<Tp>&) {});

    // the pool's counters, snapshotted under both locks. get_usage_report() is this formatted
    // for a log, so callers that want the numbers should read them here rather than parse the
    // string: the wording of the report is not an interface.
    struct usage
    {
        size_type size      = 0;
        size_type available = 0;
        size_type reused    = 0;
        size_type released  = 0;
        size_type batches   = 0;
    };

    usage       get_usage() const;
    std::string get_usage_report() const;

private:
    size_type              m_count         = 256;
    std::function<void()>  m_function      = nullptr;
    mutable std::mutex     m_pool_mtx      = {};
    pool_array_type        m_pool          = {};
    retired_array_type     m_retired       = {};
    mutable std::mutex     m_available_mtx = {};
    std::queue<size_type>  m_available     = {};
    std::atomic<size_type> m_released      = 0;
    std::atomic<size_type> m_reused        = 0;
    std::atomic<size_type> m_new_batch     = 0;
};

template <typename Tp>
template <typename FuncT, typename... Args>
pool<Tp>::pool(std::piecewise_construct_t, size_type count, FuncT&& ctor, Args&&... args)
: m_count{count}
, m_function{[this,
              _ctor       = std::forward<FuncT>(ctor),
              _args_tuple = std::make_tuple(std::forward<Args>(args)...)]() {
    for(size_type i = 0; i < m_count; ++i)
    {
        auto idx = m_pool.size();
        m_pool.emplace_back(idx, false, this);
        std::apply(
            [&](auto&&... unpacked_args) {
                _ctor(m_pool[idx].get(), std::forward<decltype(unpacked_args)>(unpacked_args)...);
            },
            _args_tuple);
        m_available.push(idx);
    }
}}
{
    m_function();
}

template <typename Tp>
pool_object<Tp>&
pool<Tp>::acquire()
{
    // m_pool_mtx is taken before the free-list pop rather than after it, so the pop and the
    // lookup that consumes the popped index are one critical section. clear() retires m_pool
    // under this lock, so it can no longer land between the two and strand the popped index:
    // a stale index is never produced, and there is nothing here left to validate. A bounds
    // check could not have done that job -- it cannot tell an index stranded by a clear()
    // from an in-range index belonging to a later generation -- and this is why no
    // epoch/generation counter is needed to tell those apart.
    auto _pool_lk  = std::unique_lock<std::mutex>{m_pool_mtx};
    auto _avail_lk = std::unique_lock<std::mutex>{m_available_mtx};

    // m_function() appends a batch to m_pool and pushes its indices onto the free list, so it
    // needs both locks, exactly as it already did on the growth path
    while(m_available.empty())
    {
        ROCP_INFO << fmt::format(
            "Pool of type {} exhausted. Creating new batch of {} objects. New pool size: {}",
            cxx_demangle(typeid(Tp).name()),
            m_count,
            m_pool.size() + m_count);
        m_new_batch++;
        m_function();
    }

    auto _idx = m_available.front();
    m_available.pop();
    if(m_released > 0)
    {
        m_reused++;
    }

    auto& _obj = m_pool.at(_idx);
    ROCP_FATAL_IF(!_obj.acquire())
        << fmt::format("Pool object at index {} was expected to be available but was not", _idx);
    return _obj;
}

template <typename Tp>
bool
pool<Tp>::release(pool_object<Tp>& obj)
{
    // the in-use exchange happens here, under m_pool_mtx, and not in the caller before it
    // arrives. clear() clears that flag on every object it retires while holding this same
    // lock, so a successful exchange is itself proof that no clear() has intervened since the
    // object was acquired: the flag does the work a generation counter would, and a retired
    // object can never queue its index for a later acquire() to pop. An object is taken by
    // reference rather than by index for the same reason -- an identity can be validated this
    // way, a bare index cannot.
    auto _pool_lk = std::unique_lock<std::mutex>{m_pool_mtx};
    if(!obj.clear_in_use()) return false;

    auto _write_avail_lk = std::unique_lock<std::mutex>{m_available_mtx};
    m_available.push(obj.index());
    m_released++;
    return true;
}

// get an object from the pool. if all objects are in use, a new one will be created and added to
// the pool
template <typename Tp>
template <typename FuncT, typename... Args>
pool_object<Tp>&
pool<Tp>::acquire(FuncT&& ctor, Args&&... args)
{
    auto& _ref = acquire();
    ctor(_ref.get(), std::forward<Args>(args)...);
    return _ref;
}

template <typename Tp>
template <typename FuncT>
void
pool<Tp>::clear(FuncT&& func)
{
    auto _write_pool_lk = std::unique_lock<std::mutex>{m_pool_mtx};

    for(auto& itr : m_pool)
    {
        ROCP_WARNING_IF(itr.in_use()) << fmt::format(
            "Pool object at index {} is still in use during pool clear", itr.index());
        // not itr.release(): that re-enters pool<Tp>::release(), which takes m_pool_mtx. The
        // free list is drained below, so the bookkeeping it would do is of no use here.
        itr.clear_in_use();
        // run cleanup lambda
        if constexpr(std::is_invocable_v<FuncT, pool_object<Tp>&>)
        {
            func(itr);
        }
        else if constexpr(std::is_invocable_v<FuncT, Tp&>)
        {
            func(itr.get());
        }
        else
        {
            static_assert(mpl::assert_false<FuncT>::value,
                          "Invalid function type for pool<Tp>::clear");
        }
    }

    auto _write_avail_lk = std::unique_lock<std::mutex>{m_available_mtx};

    while(!m_available.empty())
        m_available.pop();

    // The storage is retired, not freed. A pool_object<Tp>* handed out before this call can
    // outlive it -- the HSA async signal handler holds one per dispatch, and finalization
    // reaches here without waiting for those handlers -- and a late release() on such an
    // object reaches pool<Tp>::release(), which exchanges m_in_use on it, so freeing here
    // would be a use-after-free on the object. The loop above already cleared m_in_use on
    // every object, so that exchange fails and no retired index re-enters the free list.
    // Moving a stable_vector moves only its chunk index, so the objects keep their addresses.
    // Bounded: clear() is a teardown operation, and the storage is freed with the pool.
    //
    // Narrowed, not eliminated. The retired chunks go away when the pool itself is destroyed by
    // destroy_static_objects(), which registration.cpp runs from the same atexit handler as
    // finalize(); a signal handler already past its get_fini_status() > 0 early return can still
    // be mid-body by then. The window shrinks from "after clear()" to "after set_fini_status(1)".
    m_retired.emplace_back(std::move(m_pool));
    m_pool = pool_array_type{};
    m_released.store(0);
    m_reused.store(0);
    m_new_batch.store(0);
}

template <typename Tp>
typename pool<Tp>::usage
pool<Tp>::get_usage() const
{
    auto _pool_lk  = std::unique_lock<std::mutex>{m_pool_mtx};
    auto _avail_lk = std::unique_lock<std::mutex>{m_available_mtx};

    auto _usage      = usage{};
    _usage.size      = m_pool.size();
    _usage.available = m_available.size();
    _usage.reused    = m_reused.load();
    _usage.released  = m_released.load();
    _usage.batches   = m_new_batch.load();
    return _usage;
}

template <typename Tp>
std::string
pool<Tp>::get_usage_report() const
{
    const auto _usage = get_usage();
    return fmt::format("Usage report for pool (type='{}') :: size={}, available={}, reused={}, "
                       "released={}, batches={}",
                       cxx_demangle(typeid(Tp).name()),
                       _usage.size,
                       _usage.available,
                       _usage.reused,
                       _usage.released,
                       _usage.batches);
}
}  // namespace container
}  // namespace common
}  // namespace rocprofiler
