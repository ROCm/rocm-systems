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

#include "lib/rocprofiler-sdk/hsa/queue_signal_subscription.hpp"

#include "lib/common/environment.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/hsa/queue.hpp"

#include <fmt/format.h>

#include <memory>
#include <mutex>
#include <utility>

namespace rocprofiler::hsa
{
namespace
{
bool
queue_signal_diag_enabled()
{
    static auto _v = common::get_env("ROCPROF_QUEUE_SIGNAL_DIAG", false);
    return _v;
}

std::mutex&
get_queue_signal_monitor_mutex()
{
    static auto _v = std::mutex{};
    return _v;
}

std::shared_ptr<SignalMonitor>&
get_queue_signal_monitor()
{
    static auto monitor = [] {
        SignalMonitorConfig cfg{};
        cfg.allow_poll_fallback = false;
        auto _monitor           = create_signal_monitor(SignalMonitorBackend::ioctl, cfg);
        ROCP_ERROR_IF(queue_signal_diag_enabled())
            << fmt::format("DEBUG: queue signal monitor create backend=ioctl monitor={} "
                           "allow_poll_fallback={} poll_interval_us={} active_spin_window_us={} "
                           "callback_threads={}",
                           static_cast<void*>(_monitor.get()),
                           cfg.allow_poll_fallback,
                           cfg.poll_interval_us,
                           cfg.active_spin_window_us,
                           cfg.callback_threads);
        return _monitor;
    }();
    return monitor;
}
}  // namespace

bool
QueueSignalSubscription::arm(Queue&                    queue,
                             hsa_signal_t              signal,
                             signal_monitor_callback_t callback,
                             void*                     callback_data)
{
    auto& monitor = get_queue_signal_monitor();
    ROCP_ERROR_IF(queue_signal_diag_enabled())
        << fmt::format("DEBUG: QueueSignalSubscription::arm queue_id={} signal_handle={} "
                       "callback_data={} monitor={} tid={}",
                       queue.get_id(),
                       signal.handle,
                       callback_data,
                       static_cast<void*>(monitor.get()),
                       common::get_tid());
    if(!monitor)
    {
        ROCP_ERROR_IF(queue_signal_diag_enabled())
            << fmt::format("DEBUG: QueueSignalSubscription::arm failed: monitor missing "
                           "queue_id={} signal_handle={}",
                           queue.get_id(),
                           signal.handle);
        return false;
    }
    auto id = monitor->subscribe(signal, HSA_SIGNAL_CONDITION_EQ, -1, std::move(callback));
    ROCP_ERROR_IF(queue_signal_diag_enabled()) << fmt::format(
        "DEBUG: QueueSignalSubscription::arm subscribed queue_id={} signal_handle={} "
        "subscription_id={} success={}",
        queue.get_id(),
        signal.handle,
        id,
        (id != 0));
    return (id != 0);
}

bool
QueueSignalSubscription::test_invoke_for_unit(signal_monitor_callback_t callback,
                                              hsa_signal_value_t        value)
{
    if(!callback) return false;
    return callback(value) == false;
}

void
QueueSignalSubscription::initialize()
{
    std::lock_guard<std::mutex> lock(get_queue_signal_monitor_mutex());
    auto&                       monitor = get_queue_signal_monitor();
    ROCP_ERROR_IF(queue_signal_diag_enabled()) << fmt::format(
        "DEBUG: QueueSignalSubscription::initialize monitor={}", static_cast<void*>(monitor.get()));
    if(monitor) monitor->start();
}

void
QueueSignalSubscription::shutdown()
{
    std::lock_guard<std::mutex> lock(get_queue_signal_monitor_mutex());
    auto&                       monitor = get_queue_signal_monitor();
    ROCP_ERROR_IF(queue_signal_diag_enabled()) << fmt::format(
        "DEBUG: QueueSignalSubscription::shutdown monitor={}", static_cast<void*>(monitor.get()));
    if(monitor) monitor->stop();
}
}  // namespace rocprofiler::hsa
