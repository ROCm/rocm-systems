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

#include "lib/rocprofiler-sdk/hsa/async_wait_manager.hpp"
#include "lib/common/logging.hpp"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"

#include <hsa/hsa_ext_amd.h>

#include <thread>

namespace rocprofiler
{
namespace hsa
{
namespace
{
std::atomic<bool>     s_initialized{false};
std::function<void()> s_shutdown_callback = nullptr;
std::mutex            s_callback_mutex;

class async_wait_manager
{
public:
    static async_wait_manager& instance()
    {
        static async_wait_manager mgr;
        return mgr;
    }

    bool is_shutdown() const { return _shutdown.load(std::memory_order_acquire); }
    void notify_shutdown() { _shutdown.store(true, std::memory_order_release); }
    void reset() { _shutdown.store(false, std::memory_order_release); }

private:
    std::atomic<bool> _shutdown{false};
};

#ifdef HSA_AMD_SYSTEM_ASYNC_HANDLER_DESTROY_EVENT
hsa_status_t
system_event_handler(const hsa_amd_event_t* event, void* /*data*/)
{
    if(event && event->event_type == HSA_AMD_SYSTEM_ASYNC_HANDLER_DESTROY_EVENT)
    {
        async_wait_manager::instance().notify_shutdown();

        // Invoke deferred finalization callback if registered
        std::function<void()> cb;
        {
            std::lock_guard<std::mutex> lock(s_callback_mutex);
            cb                  = std::move(s_shutdown_callback);
            s_shutdown_callback = nullptr;
        }
        if(cb) cb();
    }
    return HSA_STATUS_SUCCESS;
}
#endif

bool
signal_condition_satisfied(hsa_signal_condition_t cond,
                           hsa_signal_value_t     result,
                           hsa_signal_value_t     value)
{
    switch(cond)
    {
        case HSA_SIGNAL_CONDITION_EQ: return (result == value);
        case HSA_SIGNAL_CONDITION_NE: return (result != value);
        case HSA_SIGNAL_CONDITION_LT: return (result < value);
        case HSA_SIGNAL_CONDITION_GTE: return (result >= value);
    }
    return false;
}
}  // namespace

void
async_wait_manager_init()
{
    async_wait_manager::instance().reset();
    s_initialized.store(true, std::memory_order_release);

#ifdef HSA_AMD_SYSTEM_ASYNC_HANDLER_DESTROY_EVENT
    auto* ext_table = get_amd_ext_table();
    if(ext_table && ext_table->hsa_amd_register_system_event_handler_fn)
    {
        ext_table->hsa_amd_register_system_event_handler_fn(system_event_handler, nullptr);
    }
#endif
}

void
notify_async_shutdown()
{
    async_wait_manager::instance().notify_shutdown();
}

void
reset_async_shutdown()
{
    async_wait_manager::instance().reset();
}

bool
is_async_shutdown()
{
    return async_wait_manager::instance().is_shutdown();
}

bool
is_async_wait_initialized()
{
    return s_initialized.load(std::memory_order_acquire);
}

void
register_shutdown_callback(shutdown_callback_t callback)
{
    std::lock_guard<std::mutex> lock(s_callback_mutex);
    s_shutdown_callback = std::move(callback);
}

wait_result
wait_or_shutdown(hsa_signal_t           signal,
                 hsa_signal_condition_t cond,
                 hsa_signal_value_t     value,
                 std::string_view       callsite,
                 uint64_t               timeout_ns,
                 uint64_t               poll_interval_ns,
                 const CoreApiTable*    core_api)
{
    auto&       mgr          = async_wait_manager::instance();
    const auto* resolved_api = (core_api != nullptr) ? core_api : get_core_table();
    auto        start        = std::chrono::steady_clock::now();

    while(!mgr.is_shutdown())
    {
        auto remaining_ns = timeout_ns;
        if(timeout_ns != UINT64_MAX)
        {
            auto now        = std::chrono::steady_clock::now();
            auto elapsed_ns = static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(now - start).count());
            if(elapsed_ns >= timeout_ns)
            {
                ROCP_ERROR << "signal wait timed out at " << callsite;
                return wait_result::timeout;
            }
            remaining_ns = timeout_ns - elapsed_ns;
        }

        auto interval = std::min(poll_interval_ns, remaining_ns);
        auto result   = resolved_api->hsa_signal_wait_relaxed_fn(
            signal, cond, value, interval, HSA_WAIT_STATE_BLOCKED);

        if(signal_condition_satisfied(cond, result, value))
        {
            return wait_result::completed;
        }
    }

    ROCP_WARNING << "signal wait interrupted by HSA async handler shutdown at " << callsite;
    return wait_result::shutdown;
}

wait_result
wait_or_shutdown(std::condition_variable&      cv,
                 std::unique_lock<std::mutex>& lock,
                 const std::function<bool()>&  predicate,
                 std::string_view              callsite,
                 uint64_t                      timeout_ns,
                 std::chrono::milliseconds     interval)
{
    auto& mgr   = async_wait_manager::instance();
    auto  start = std::chrono::steady_clock::now();

    while(!predicate())
    {
        if(mgr.is_shutdown())
        {
            ROCP_WARNING << "cv wait interrupted by HSA async handler shutdown at " << callsite;
            return wait_result::shutdown;
        }
        auto now     = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(now - start).count();
        if(timeout_ns != UINT64_MAX && static_cast<uint64_t>(elapsed) >= timeout_ns)
        {
            ROCP_ERROR << "cv wait timed out at " << callsite;
            return wait_result::timeout;
        }
        cv.wait_for(lock, interval);
    }
    return wait_result::completed;
}

template <typename T>
wait_result
wait_or_shutdown(std::atomic<T>&  atomic_val,
                 T                expected,
                 std::string_view callsite,
                 uint64_t         timeout_ns)
{
    auto& mgr   = async_wait_manager::instance();
    auto  start = std::chrono::steady_clock::now();

    while(atomic_val.load(std::memory_order_acquire) != expected)
    {
        if(mgr.is_shutdown())
        {
            ROCP_WARNING << "atomic wait interrupted by HSA async handler shutdown at " << callsite;
            return wait_result::shutdown;
        }
        auto now     = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(now - start).count();
        if(timeout_ns != UINT64_MAX && static_cast<uint64_t>(elapsed) >= timeout_ns)
        {
            ROCP_ERROR << "atomic wait timed out at " << callsite;
            return wait_result::timeout;
        }
        std::this_thread::yield();
    }
    return wait_result::completed;
}

template wait_result
wait_or_shutdown(std::atomic<int>&, int, std::string_view, uint64_t);

}  // namespace hsa
}  // namespace rocprofiler
