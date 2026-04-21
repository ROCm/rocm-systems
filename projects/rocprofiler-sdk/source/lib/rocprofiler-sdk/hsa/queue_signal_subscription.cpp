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

#include "lib/rocprofiler-sdk/hsa/queue.hpp"

#include <memory>
#include <mutex>
#include <utility>

namespace rocprofiler::hsa
{
namespace
{
std::mutex&
get_queue_signal_monitor_mutex()
{
    static auto _v = std::mutex{};
    return _v;
}

std::shared_ptr<SignalMonitor>&
get_queue_signal_monitor()
{
    static auto monitor =
        create_signal_monitor(parse_signal_monitor_backend_env(), SignalMonitorConfig{});
    return monitor;
}
}  // namespace

bool
QueueSignalSubscription::arm(Queue&               queue,
                             hsa_signal_t         signal,
                             signal_monitor_callback_t callback,
                             void*                callback_data)
{
    (void) queue;
    (void) callback_data;

    auto& monitor = get_queue_signal_monitor();
    if(!monitor) return false;
    auto id = monitor->subscribe(signal, HSA_SIGNAL_CONDITION_EQ, -1, std::move(callback));
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
    if(monitor) monitor->start();
}

void
QueueSignalSubscription::shutdown()
{
    std::lock_guard<std::mutex> lock(get_queue_signal_monitor_mutex());
    auto&                       monitor = get_queue_signal_monitor();
    if(monitor) monitor->stop();
}
}  // namespace rocprofiler::hsa
