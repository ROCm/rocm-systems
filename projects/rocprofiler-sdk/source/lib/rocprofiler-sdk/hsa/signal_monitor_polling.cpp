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

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

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
};

class PollingSignalMonitor final : public SignalMonitor
{
public:
    PollingSignalMonitor(const SignalMonitorConfig& cfg, SignalMonitorOps ops)
    : cfg_{cfg}
    , ops_{std::move(ops)}
    {
        if(!ops_.load)
        {
            ops_.load = [](hsa_signal_t signal) { return hsa_signal_load_relaxed(signal); };
        }
    }

    ~PollingSignalMonitor() override { stop(); }

    void start() override
    {
        std::lock_guard<std::mutex> lock(lifecycle_mutex_);
        if(running_.exchange(true)) return;

        const auto thread_count = (cfg_.callback_threads > 0) ? cfg_.callback_threads : 1u;
        worker_threads_.reserve(thread_count);
        for(uint32_t i = 0; i < thread_count; ++i)
        {
            worker_threads_.emplace_back(&PollingSignalMonitor::callback_loop, this);
        }

        detector_thread_ = std::thread(&PollingSignalMonitor::detector_loop, this);
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
    void detector_loop()
    {
        const auto interval = std::chrono::microseconds{cfg_.poll_interval_us};
        while(running_.load())
        {
            std::vector<CallbackItem> ready;
            {
                std::lock_guard<std::mutex> lock(subscriptions_mutex_);
                for(auto& [id, subscription] : subscriptions_)
                {
                    if(subscription.pending) continue;
                    const auto current_value = ops_.load(subscription.signal);
                    if(!evaluate_signal_condition(subscription.condition,
                                                  current_value,
                                                  subscription.compare_value))
                    {
                        continue;
                    }

                    subscription.pending = true;
                    ready.push_back(CallbackItem{id, current_value});
                }
            }

            if(!ready.empty())
            {
                {
                    std::lock_guard<std::mutex> lock(queue_mutex_);
                    for(const auto& item : ready)
                        callback_queue_.push_back(item);
                }
                queue_cv_.notify_all();
            }

            if(interval.count() > 0)
            {
                std::this_thread::sleep_for(interval);
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
            auto it = subscriptions_.find(item.id);
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

    std::atomic<bool> running_{false};
    std::atomic<signal_subscription_id_t> next_id_{1};

    std::mutex              lifecycle_mutex_;
    std::thread             detector_thread_;
    std::vector<std::thread> worker_threads_;

    std::mutex                                                   subscriptions_mutex_;
    std::unordered_map<signal_subscription_id_t, Subscription> subscriptions_;

    std::mutex                queue_mutex_;
    std::condition_variable   queue_cv_;
    std::deque<CallbackItem> callback_queue_;
};
}  // namespace

namespace detail
{
std::shared_ptr<SignalMonitor>
create_polling_signal_monitor(const SignalMonitorConfig& cfg, SignalMonitorOps ops)
{
    return std::make_shared<PollingSignalMonitor>(cfg, std::move(ops));
}
}  // namespace detail
}  // namespace rocprofiler::hsa
