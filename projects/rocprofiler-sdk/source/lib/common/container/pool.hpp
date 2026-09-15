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
#include <optional>
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
    void             release(size_type idx);

    template <typename FuncT, typename... Args>
    pool_object<Tp>& acquire(FuncT&& ctor, Args&&... args);

    template <typename FuncT = void (*)(pool_object<Tp>&)>
    void clear(FuncT&& func = [](pool_object<Tp>&) {});

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
    auto _idx = std::optional<size_type>{};
    {
        auto _avail_lk = std::unique_lock<std::mutex>{m_available_mtx};
        if(!m_available.empty())
        {
            _idx = m_available.front();
            m_available.pop();
            if(m_released > 0)
            {
                m_reused++;
            }
        }
    }

    if(_idx.has_value())
    {
        auto _read_lk = std::unique_lock<std::mutex>{m_pool_mtx};
        // the free list is unlocked between the pop above and this lock, so clear() can retire
        // m_pool in between and strand the popped index. Drop it and take the growth path
        // rather than indexing past the end of the repopulated pool.
        //
        // TODO(ihhuang): the check below narrows that window, it does not close it, and what
        // is left is a known gap rather than a caveat: a bounds check cannot tell a
        // stale generation from a new one. An index stranded by a clear() that another thread
        // has since regrown past is back in range here, so it passes the check and aliases
        // the new generation's object while still sitting on the free list. release() carries
        // the same residual. Closing it needs an epoch/generation counter stamped on the
        // index and compared here, which is deliberately out of scope for this change.
        if(_idx.value() < m_pool.size())
        {
            auto& _obj = m_pool.at(_idx.value());
            ROCP_FATAL_IF(!_obj.acquire()) << fmt::format(
                "Pool object at index {} was expected to be available but was not", _idx.value());
            return _obj;
        }
    }

    // add a new batch
    {
        auto _write_pool_lk  = std::unique_lock<std::mutex>{m_pool_mtx};
        auto _write_avail_lk = std::unique_lock<std::mutex>{m_available_mtx};
        if(m_available.empty())
        {
            ROCP_INFO << fmt::format(
                "Pool of type {} exhausted. Creating new batch of {} objects. New pool size: {}",
                cxx_demangle(typeid(Tp).name()),
                m_count,
                m_pool.size() + m_count);
            m_new_batch++;
            m_function();
        }
    }

    return acquire();
}

template <typename Tp>
void
pool<Tp>::release(size_type idx)
{
    // m_pool_mtx is held across both the bounds check and the push: clear() holds it while it
    // retires m_pool and drains the free list, so neither can interleave here and leave a
    // retired index queued for a later acquire() to pop. Same residual as acquire(), see the
    // TODO there.
    auto _read_pool_lk = std::unique_lock<std::mutex>{m_pool_mtx};
    if(idx >= m_pool.size()) return;

    auto _write_avail_lk = std::unique_lock<std::mutex>{m_available_mtx};
    m_available.push(idx);
    m_released++;
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
    // reaches here without waiting for those handlers -- and pool_object::release() exchanges
    // m_in_use on itself before it ever reaches this pool, so freeing here is a use-after-free
    // on the object that no bounds check in release() can prevent. The loop above already
    // cleared m_in_use on every object, so that exchange fails and no retired index re-enters
    // the free list.
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
std::string
pool<Tp>::get_usage_report() const
{
    auto _pool_lk  = std::unique_lock<std::mutex>{m_pool_mtx};
    auto _avail_lk = std::unique_lock<std::mutex>{m_available_mtx};
    return fmt::format("Usage report for pool (type='{}') :: size={}, available={}, reused={}, "
                       "released={}, batches={}",
                       cxx_demangle(typeid(Tp).name()),
                       m_pool.size(),
                       m_available.size(),
                       m_reused.load(),
                       m_released.load(),
                       m_new_batch.load());
}
}  // namespace container
}  // namespace common
}  // namespace rocprofiler
