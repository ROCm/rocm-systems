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

#include "lib/rocprofiler-sdk/hsa/signal_monitor.hpp"

#include "lib/rocprofiler-sdk/details/kfd_ioctl.h"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"

#include <hsa/hsa.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <fcntl.h>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <sys/ioctl.h>
#include <unistd.h>

namespace rocprofiler::hsa
{
namespace
{
struct CallbackItem
{
    signal_subscription_id_t id = 0;
    hsa_signal_value_t        current_value = 0;
};

struct Subscription
{
    hsa_signal_t               signal = {};
    hsa_signal_condition_t     condition = HSA_SIGNAL_CONDITION_EQ;
    hsa_signal_value_t         compare_value = 0;
    signal_monitor_callback_t  callback = {};
    bool                       pending = false;
    uint32_t                   event_id = 0;
    bool                       has_event_id = false;
};

hsa_signal_value_t
default_load_signal(hsa_signal_t signal)
{
    return hsa_signal_load_relaxed(signal);
}

int
get_kfd_wait_fd()
{
    static std::atomic<int> cached_fd{-2};  // -2 means uninitialized
    auto                    fd = cached_fd.load(std::memory_order_acquire);
    if(fd != -2) return fd;

    auto opened = open("/dev/kfd", O_RDWR | O_CLOEXEC);
    auto expected = -2;
    if(cached_fd.compare_exchange_strong(expected, opened, std::memory_order_release))
    {
        return opened;
    }

    if(opened >= 0) close(opened);
    return cached_fd.load(std::memory_order_acquire);
}

bool
default_get_event_id(hsa_signal_t signal, uint32_t& event_id)
{
    auto* ext_api = get_amd_ext_table();
    if(!ext_api || !ext_api->hsa_amd_signal_get_event_id_fn) return false;
    return (ext_api->hsa_amd_signal_get_event_id_fn(signal, &event_id) == HSA_STATUS_SUCCESS);
}

bool
default_wait_events(const std::vector<uint32_t>& event_ids, uint32_t timeout_ms)
{
    if(event_ids.empty())
    {
        if(timeout_ms > 0)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds{timeout_ms});
        }
        else
        {
            std::this_thread::yield();
        }
        return true;
    }

    auto fd = get_kfd_wait_fd();
    if(fd < 0) return false;

    std::vector<kfd_event_data> wait_data(event_ids.size());
    for(size_t i = 0; i < event_ids.size(); ++i)
    {
        wait_data[i].event_id = event_ids.at(i);
    }

    kfd_ioctl_wait_events_args args = {};
    args.events_ptr                 = reinterpret_cast<uint64_t>(wait_data.data());
    args.num_events                 = static_cast<uint32_t>(wait_data.size());
    args.wait_for_all               = 0;
    args.timeout                    = timeout_ms;

    if(ioctl(fd, AMDKFD_IOC_WAIT_EVENTS, &args) != 0) return false;
    return (args.wait_result != KFD_IOC_WAIT_RESULT_FAIL);
}

bool
has_default_ioctl_support()
{
    auto* ext_api = get_amd_ext_table();
    if(!ext_api || !ext_api->hsa_amd_signal_get_event_id_fn) return false;
    return (get_kfd_wait_fd() >= 0);
}

class IoctlSignalMonitor final : public SignalMonitor
{
public:
    IoctlSignalMonitor(const SignalMonitorConfig& cfg, SignalMonitorOps ops)
    : cfg_{cfg}
    , ops_{std::move(ops)}
    {
        if(!ops_.load) ops_.load = default_load_signal;
    }

    ~IoctlSignalMonitor() override { stop(); }

    void start() override
    {
        std::lock_guard<std::mutex> lock(lifecycle_mutex_);
        if(running_.exchange(true)) return;

        const auto thread_count = (cfg_.callback_threads > 0) ? cfg_.callback_threads : 1u;
        worker_threads_.reserve(thread_count);
        for(uint32_t i = 0; i < thread_count; ++i)
        {
            worker_threads_.emplace_back(&IoctlSignalMonitor::callback_loop, this);
        }

        detector_thread_ = std::thread(&IoctlSignalMonitor::detector_loop, this);
    }

    void stop() override
    {
        {
            std::lock_guard<std::mutex> lock(lifecycle_mutex_);
            if(!running_.exchange(false)) return;
        }

        {
            std::lock_guard<std::mutex> lock(subscriptions_mutex_);
            for(auto& [id, subscription] : subscriptions_)
            {
                (void) id;
                subscription.pending = false;
            }
        }

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            callback_queue_.clear();
        }
        queue_cv_.notify_all();

        if(detector_thread_.joinable()) detector_thread_.join();
        for(auto& worker : worker_threads_)
        {
            if(worker.joinable()) worker.join();
        }
        worker_threads_.clear();
    }

    signal_subscription_id_t subscribe(hsa_signal_t              signal,
                                       hsa_signal_condition_t    condition,
                                       hsa_signal_value_t        compare_value,
                                       signal_monitor_callback_t cb) override
    {
        const auto id = next_id_.fetch_add(1);

        Subscription subscription{};
        subscription.signal = signal;
        subscription.condition = condition;
        subscription.compare_value = compare_value;
        subscription.callback = std::move(cb);
        if(ops_.get_event_id)
        {
            subscription.has_event_id = ops_.get_event_id(signal, subscription.event_id);
        }

        std::lock_guard<std::mutex> lock(subscriptions_mutex_);
        subscriptions_.emplace(id, std::move(subscription));
        return id;
    }

    bool unsubscribe(signal_subscription_id_t id) override
    {
        bool erased = false;
        {
            std::lock_guard<std::mutex> lock(subscriptions_mutex_);
            erased = (subscriptions_.erase(id) > 0);
        }

        if(!erased) return false;

        std::lock_guard<std::mutex> lock(queue_mutex_);
        if(callback_queue_.empty()) return true;

        std::deque<CallbackItem> remaining;
        remaining.swap(callback_queue_);
        for(const auto& item : remaining)
        {
            if(item.id != id) callback_queue_.push_back(item);
        }
        return true;
    }

private:
    void collect_ready(std::vector<CallbackItem>& ready, std::vector<uint32_t>* event_ids)
    {
        std::lock_guard<std::mutex> lock(subscriptions_mutex_);
        for(auto& [id, subscription] : subscriptions_)
        {
            if(subscription.pending) continue;
            if(event_ids && subscription.has_event_id) event_ids->push_back(subscription.event_id);

            const auto current_value = ops_.load(subscription.signal);
            if(!evaluate_signal_condition(
                   subscription.condition, current_value, subscription.compare_value))
            {
                continue;
            }

            subscription.pending = true;
            ready.push_back(CallbackItem{id, current_value});
        }
    }

    void enqueue_ready(const std::vector<CallbackItem>& ready)
    {
        if(ready.empty()) return;

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            for(const auto& item : ready)
                callback_queue_.push_back(item);
        }
        queue_cv_.notify_all();
    }

    void detector_loop()
    {
        const auto poll_interval = std::chrono::microseconds{cfg_.poll_interval_us};
        const auto spin_window   = std::chrono::microseconds{cfg_.active_spin_window_us};
        const auto wait_timeout_ms = std::max<uint32_t>((cfg_.poll_interval_us + 999u) / 1000u, 1u);

        while(running_.load())
        {
            std::vector<CallbackItem> ready = {};
            collect_ready(ready, nullptr);
            if(!ready.empty())
            {
                enqueue_ready(ready);
                continue;
            }

            if(spin_window.count() > 0)
            {
                auto spin_deadline = std::chrono::steady_clock::now() + spin_window;
                while(running_.load() && std::chrono::steady_clock::now() < spin_deadline)
                {
                    ready.clear();
                    collect_ready(ready, nullptr);
                    if(!ready.empty()) break;
                    std::this_thread::yield();
                }

                if(!ready.empty())
                {
                    enqueue_ready(ready);
                    continue;
                }
            }

            std::vector<uint32_t> event_ids = {};
            ready.clear();
            collect_ready(ready, &event_ids);
            if(!ready.empty())
            {
                enqueue_ready(ready);
                continue;
            }

            if(!running_.load()) break;

            if(!event_ids.empty() && ops_.wait_events)
            {
                (void) ops_.wait_events(event_ids, wait_timeout_ms);
            }
            else if(poll_interval.count() > 0)
            {
                std::this_thread::sleep_for(poll_interval);
            }
            else
            {
                std::this_thread::yield();
            }
        }
    }

    void callback_loop()
    {
        while(true)
        {
            CallbackItem item{};
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_cv_.wait(lock, [&]() { return !running_.load() || !callback_queue_.empty(); });
                if(!running_.load() && callback_queue_.empty()) return;

                item = callback_queue_.front();
                callback_queue_.pop_front();
            }

            signal_monitor_callback_t callback;
            {
                std::lock_guard<std::mutex> lock(subscriptions_mutex_);
                auto it = subscriptions_.find(item.id);
                if(it == subscriptions_.end()) continue;
                callback = it->second.callback;
            }

            bool keep = false;
            if(callback)
            {
                try
                {
                    keep = callback(item.current_value);
                }
                catch(...)
                {
                    keep = false;
                }
            }

            std::lock_guard<std::mutex> lock(subscriptions_mutex_);
            auto                        it = subscriptions_.find(item.id);
            if(it == subscriptions_.end()) continue;

            if(!keep)
            {
                subscriptions_.erase(it);
            }
            else
            {
                it->second.pending = false;
            }
        }
    }

    SignalMonitorConfig cfg_ = {};
    SignalMonitorOps    ops_ = {};

    std::atomic<bool>                    running_{false};
    std::atomic<signal_subscription_id_t> next_id_{1};

    std::mutex               lifecycle_mutex_;
    std::thread              detector_thread_;
    std::vector<std::thread> worker_threads_;

    std::mutex                                                 subscriptions_mutex_;
    std::unordered_map<signal_subscription_id_t, Subscription> subscriptions_;

    std::mutex                queue_mutex_;
    std::condition_variable   queue_cv_;
    std::deque<CallbackItem> callback_queue_;
};
}  // namespace

namespace detail
{
std::shared_ptr<SignalMonitor>
create_ioctl_signal_monitor(const SignalMonitorConfig& cfg, SignalMonitorOps ops)
{
    const bool has_custom_ops =
        static_cast<bool>(ops.load) || static_cast<bool>(ops.get_event_id) ||
        static_cast<bool>(ops.wait_events);

    if(!ops.load) ops.load = default_load_signal;

    if(!has_custom_ops)
    {
        if(!has_default_ioctl_support()) return {};
        ops.get_event_id = default_get_event_id;
        ops.wait_events  = default_wait_events;
    }

    if(!ops.get_event_id || !ops.wait_events) return {};
    return std::make_shared<IoctlSignalMonitor>(cfg, std::move(ops));
}
}  // namespace detail
}  // namespace rocprofiler::hsa
