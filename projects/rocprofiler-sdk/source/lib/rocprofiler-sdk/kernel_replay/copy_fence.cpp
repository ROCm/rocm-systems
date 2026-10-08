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

#include "lib/rocprofiler-sdk/kernel_replay/copy_fence.hpp"

#include "lib/common/logging.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/synchronized.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"
#include "lib/rocprofiler-sdk/kernel_replay/memory_tracker.hpp"
#include "lib/rocprofiler-sdk/registration.hpp"

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace rocprofiler
{
namespace kernel_replay
{
namespace copy_fence
{
namespace
{
struct fence_t
{
    std::mutex              mtx       = {};
    std::condition_variable cv        = {};
    int64_t                 in_flight = 0;
    int64_t                 waiting   = 0;
    bool                    window    = false;
};

using fence_map_t = std::unordered_map<uint64_t, std::unique_ptr<fence_t>>;

// Entries are never erased, so a reference stays valid after the map lock is released.
fence_t&
fence_of(uint64_t agent_handle)
{
    static auto*& _map = common::static_object<common::Synchronized<fence_map_t>>::construct();
    return _map->wlock(
        [](fence_map_t& _m, uint64_t _h) -> fence_t& {
            auto& _f = _m[_h];
            if(!_f) _f = std::make_unique<fence_t>();
            return *_f;
        },
        agent_handle);
}

// The agents one copy touches, deduplicated and sorted so concurrent copies enter fences in the
// same order.
struct agents_t
{
    uint64_t handles[2] = {0, 0};
    int      count      = 0;

    agents_t(hsa_agent_t a, hsa_agent_t b)
    {
        if(a.handle != 0) handles[count++] = a.handle;
        if(b.handle != 0 && b.handle != a.handle) handles[count++] = b.handle;
        if(count == 2 && handles[0] > handles[1]) std::swap(handles[0], handles[1]);
    }
};

void
enter(const agents_t& agents)
{
    for(int i = 0; i < agents.count; ++i)
    {
        auto& _f  = fence_of(agents.handles[i]);
        auto  _lk = std::unique_lock<std::mutex>{_f.mtx};
        if(_f.window)
        {
            ++_f.waiting;
            _f.cv.wait(_lk, [&_f]() { return !_f.window; });
            --_f.waiting;
            _f.cv.notify_all();
        }
        ++_f.in_flight;
    }
}

void
leave(const agents_t& agents)
{
    for(int i = 0; i < agents.count; ++i)
    {
        auto& _f = fence_of(agents.handles[i]);
        {
            auto _lk = std::lock_guard<std::mutex>{_f.mtx};
            --_f.in_flight;
        }
        _f.cv.notify_all();
    }
}

// Proxy completion signals are reused: creating one costs a KFD event allocation.
using signal_pool_t = std::vector<hsa_signal_t>;

common::Synchronized<signal_pool_t>&
proxy_pool()
{
    static auto*& _pool = common::static_object<common::Synchronized<signal_pool_t>>::construct();
    return *_pool;
}

hsa_signal_t
acquire_proxy()
{
    auto _sig = proxy_pool().wlock([](signal_pool_t& _p) {
        auto _s = hsa_signal_t{.handle = 0};
        if(!_p.empty())
        {
            _s = _p.back();
            _p.pop_back();
        }
        return _s;
    });
    if(_sig.handle == 0 && hsa::get_amd_ext_table()->hsa_amd_signal_create_fn(
                               1, 0, nullptr, 0, &_sig) != HSA_STATUS_SUCCESS)
        return hsa_signal_t{.handle = 0};
    hsa::get_core_table()->hsa_signal_store_screlease_fn(_sig, 1);
    return _sig;
}

void
release_proxy(hsa_signal_t sig)
{
    proxy_pool().wlock([sig](signal_pool_t& _p) { _p.emplace_back(sig); });
}

struct pending_copy_t
{
    agents_t          agents;
    hsa_signal_t      app_signal = {};
    hsa_signal_t      proxy      = {};
    std::atomic<bool> submitted  = true;

    pending_copy_t(agents_t a, hsa_signal_t app, hsa_signal_t px)
    : agents{a}
    , app_signal{app}
    , proxy{px}
    {}
};

bool
on_copy_done(hsa_signal_value_t, void* arg)
{
    auto* _p = static_cast<pending_copy_t*>(arg);
    // The copy engine decrements the completion signal by one; forward exactly that.
    if(_p->submitted.load() && _p->app_signal.handle != 0)
        hsa::get_core_table()->hsa_signal_subtract_screlease_fn(_p->app_signal, 1);
    leave(_p->agents);
    release_proxy(_p->proxy);
    delete _p;
    return false;
}

template <typename SubmitFn>
hsa_status_t
fenced(agents_t agents, hsa_signal_t app_signal, SubmitFn&& submit)
{
    if(!memory_tracker::tracking_enabled() || registration::get_fini_status() > 0)
        return submit(app_signal);

    auto _proxy = acquire_proxy();
    if(_proxy.handle == 0)
    {
        LOG_FIRST_N(WARNING, 1) << "kernel replay: could not create a completion signal to fence "
                                   "an async copy against replay windows; copying unfenced";
        return submit(app_signal);
    }

    auto* _p      = new pending_copy_t{agents, app_signal, _proxy};
    auto  _status = hsa::get_amd_ext_table()->hsa_amd_signal_async_handler_fn(
        _proxy, HSA_SIGNAL_CONDITION_LT, 1, on_copy_done, _p);
    if(_status != HSA_STATUS_SUCCESS)
    {
        LOG_FIRST_N(WARNING, 1) << "kernel replay: hsa_amd_signal_async_handler failed (" << _status
                                << "); copying unfenced";
        release_proxy(_proxy);
        delete _p;
        return submit(app_signal);
    }

    enter(agents);
    _status = submit(_proxy);
    if(_status != HSA_STATUS_SUCCESS)
    {
        // Nothing will decrement the proxy: fire the handler ourselves, without forwarding.
        _p->submitted.store(false);
        hsa::get_core_table()->hsa_signal_store_screlease_fn(_proxy, 0);
    }
    return _status;
}

decltype(hsa::hsa_amd_ext_table_t{}.hsa_amd_memory_async_copy_fn)           next_copy = nullptr;
decltype(hsa::hsa_amd_ext_table_t{}.hsa_amd_memory_async_copy_on_engine_fn) next_copy_on_engine =
    nullptr;
decltype(hsa::hsa_amd_ext_table_t{}.hsa_amd_memory_async_copy_rect_fn) next_copy_rect = nullptr;

hsa_status_t
fenced_copy(void*               dst,
            hsa_agent_t         dst_agent,
            const void*         src,
            hsa_agent_t         src_agent,
            size_t              size,
            uint32_t            num_dep_signals,
            const hsa_signal_t* dep_signals,
            hsa_signal_t        completion_signal)
{
    return fenced(agents_t{dst_agent, src_agent}, completion_signal, [&](hsa_signal_t sig) {
        return next_copy(dst, dst_agent, src, src_agent, size, num_dep_signals, dep_signals, sig);
    });
}

hsa_status_t
fenced_copy_on_engine(void*                    dst,
                      hsa_agent_t              dst_agent,
                      const void*              src,
                      hsa_agent_t              src_agent,
                      size_t                   size,
                      uint32_t                 num_dep_signals,
                      const hsa_signal_t*      dep_signals,
                      hsa_signal_t             completion_signal,
                      hsa_amd_sdma_engine_id_t engine_id,
                      bool                     force_copy_on_sdma)
{
    return fenced(agents_t{dst_agent, src_agent}, completion_signal, [&](hsa_signal_t sig) {
        return next_copy_on_engine(dst,
                                   dst_agent,
                                   src,
                                   src_agent,
                                   size,
                                   num_dep_signals,
                                   dep_signals,
                                   sig,
                                   engine_id,
                                   force_copy_on_sdma);
    });
}

hsa_status_t
fenced_copy_rect(const hsa_pitched_ptr_t* dst,
                 const hsa_dim3_t*        dst_offset,
                 const hsa_pitched_ptr_t* src,
                 const hsa_dim3_t*        src_offset,
                 const hsa_dim3_t*        range,
                 hsa_agent_t              copy_agent,
                 hsa_amd_copy_direction_t dir,
                 uint32_t                 num_dep_signals,
                 const hsa_signal_t*      dep_signals,
                 hsa_signal_t             completion_signal)
{
    return fenced(
        agents_t{copy_agent, hsa_agent_t{.handle = 0}}, completion_signal, [&](hsa_signal_t sig) {
            return next_copy_rect(dst,
                                  dst_offset,
                                  src,
                                  src_offset,
                                  range,
                                  copy_agent,
                                  dir,
                                  num_dep_signals,
                                  dep_signals,
                                  sig);
        });
}
}  // namespace

bool
open_window(hsa_agent_t agent, std::chrono::milliseconds timeout)
{
    auto&      _f        = fence_of(agent.handle);
    auto       _lk       = std::unique_lock<std::mutex>{_f.mtx};
    const auto _deadline = std::chrono::steady_clock::now() + timeout;
    // Copies that the previous window held off start first. A thread that replays dispatches back
    // to back would otherwise reopen the window before they are scheduled, and starve them.
    _f.cv.wait_until(_lk, _deadline, [&_f]() { return _f.waiting == 0; });
    _f.window = true;
    if(_f.cv.wait_until(_lk, _deadline, [&_f]() { return _f.in_flight == 0; })) return true;
    _f.window = false;
    _lk.unlock();
    _f.cv.notify_all();
    return false;
}

void
close_window(hsa_agent_t agent)
{
    auto& _f = fence_of(agent.handle);
    {
        auto _lk  = std::lock_guard<std::mutex>{_f.mtx};
        _f.window = false;
    }
    _f.cv.notify_all();
}

int64_t
copies_in_flight(hsa_agent_t agent)
{
    auto& _f  = fence_of(agent.handle);
    auto  _lk = std::lock_guard<std::mutex>{_f.mtx};
    return _f.in_flight;
}
}  // namespace copy_fence

void
copy_fence_init(hsa::hsa_amd_ext_table_t* table, uint64_t lib_instance)
{
    if(!table || lib_instance > 0) return;
    // Idempotent: restore_table may hand back a table that already carries our wrappers.
    if(table->hsa_amd_memory_async_copy_fn == copy_fence::fenced_copy) return;

    copy_fence::next_copy                         = table->hsa_amd_memory_async_copy_fn;
    copy_fence::next_copy_on_engine               = table->hsa_amd_memory_async_copy_on_engine_fn;
    copy_fence::next_copy_rect                    = table->hsa_amd_memory_async_copy_rect_fn;
    table->hsa_amd_memory_async_copy_fn           = copy_fence::fenced_copy;
    table->hsa_amd_memory_async_copy_on_engine_fn = copy_fence::fenced_copy_on_engine;
    table->hsa_amd_memory_async_copy_rect_fn      = copy_fence::fenced_copy_rect;
}
}  // namespace kernel_replay
}  // namespace rocprofiler
