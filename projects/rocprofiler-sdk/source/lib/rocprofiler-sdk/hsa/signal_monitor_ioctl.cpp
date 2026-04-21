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

#include "lib/common/environment.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/utility.hpp"
#include "lib/rocprofiler-sdk/details/kfd_ioctl.h"
#include "lib/rocprofiler-sdk/hsa/hsa.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <hsa/hsa.h>

#include <fcntl.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <thread>
#include <type_traits>
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
    signal_subscription_id_t id            = 0;
    hsa_signal_value_t       current_value = 0;
};

struct Subscription
{
    hsa_signal_t              signal        = {};
    hsa_signal_condition_t    condition     = HSA_SIGNAL_CONDITION_EQ;
    hsa_signal_value_t        compare_value = 0;
    signal_monitor_callback_t callback      = {};
    bool                      pending       = false;
    uint32_t                  event_id      = 0;
    bool                      has_event_id  = false;
};

bool
ioctl_monitor_diag_enabled()
{
    static auto _v = common::get_env("ROCPROF_SIGNAL_MONITOR_DIAG", false);
    return _v;
}

const char*
to_string(hsa_signal_condition_t condition)
{
    switch(condition)
    {
        case HSA_SIGNAL_CONDITION_EQ: return "EQ";
        case HSA_SIGNAL_CONDITION_NE: return "NE";
        case HSA_SIGNAL_CONDITION_LT: return "LT";
        case HSA_SIGNAL_CONDITION_GTE: return "GTE";
        default: return "UNKNOWN";
    }
}

hsa_signal_value_t
default_load_signal(hsa_signal_t signal)
{
    auto* core_api = get_core_table();
    if(core_api && core_api->hsa_signal_load_relaxed_fn)
    {
        return core_api->hsa_signal_load_relaxed_fn(signal);
    }
    return hsa_signal_value_t{0};
}

template <typename TableT, typename = void>
struct has_signal_get_event_id_fn : std::false_type
{};

template <typename TableT>
struct has_signal_get_event_id_fn<
    TableT,
    std::void_t<decltype(std::declval<TableT*>()->hsa_amd_signal_get_event_id_fn)>> : std::true_type
{};

template <typename TableT>
bool
get_event_id_via_table(TableT* ext_api, hsa_signal_t signal, uint32_t& event_id)
{
    if constexpr(has_signal_get_event_id_fn<TableT>::value)
    {
        if(!ext_api->hsa_amd_signal_get_event_id_fn) return false;
        return (ext_api->hsa_amd_signal_get_event_id_fn(signal, &event_id) == HSA_STATUS_SUCCESS);
    }
    else
    {
        (void) ext_api;
        (void) signal;
        (void) event_id;
        return false;
    }
}

template <typename TableT>
bool
has_event_id_support(TableT* ext_api)
{
    if constexpr(has_signal_get_event_id_fn<TableT>::value)
    {
        return (ext_api != nullptr && ext_api->hsa_amd_signal_get_event_id_fn != nullptr);
    }
    else
    {
        (void) ext_api;
        return false;
    }
}

int
get_kfd_wait_fd()
{
    static std::atomic<int> cached_fd{-2};  // -2 means uninitialized
    auto                    fd = cached_fd.load(std::memory_order_acquire);
    if(fd != -2) return fd;

    auto opened   = open("/dev/kfd", O_RDWR | O_CLOEXEC);
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
    if(!ext_api) return false;
    return get_event_id_via_table(ext_api, signal, event_id);
}

bool
default_wait_events(const std::vector<uint32_t>& event_ids, uint32_t timeout_ms)
{
    ROCP_ERROR_IF(ioctl_monitor_diag_enabled())
        << fmt::format("DEBUG: ioctl monitor wait_events enter event_count={} timeout_ms={} tid={}",
                       event_ids.size(),
                       timeout_ms,
                       common::get_tid());

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
    if(fd < 0)
    {
        ROCP_ERROR_IF(ioctl_monitor_diag_enabled())
            << "DEBUG: ioctl monitor wait_events failed: /dev/kfd unavailable";
        return false;
    }

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

    if(ioctl(fd, AMDKFD_IOC_WAIT_EVENTS, &args) != 0)
    {
        ROCP_ERROR_IF(ioctl_monitor_diag_enabled())
            << fmt::format("DEBUG: ioctl monitor wait_events ioctl failure fd={} event_count={} "
                           "timeout_ms={}",
                           fd,
                           event_ids.size(),
                           timeout_ms);
        return false;
    }

    const auto success = (args.wait_result != KFD_IOC_WAIT_RESULT_FAIL);
    ROCP_ERROR_IF(ioctl_monitor_diag_enabled())
        << fmt::format("DEBUG: ioctl monitor wait_events exit wait_result={} success={}",
                       args.wait_result,
                       success);
    return success;
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

        ROCP_ERROR_IF(ioctl_monitor_diag_enabled())
            << fmt::format("DEBUG: ioctl monitor start monitor={} callback_threads={} "
                           "poll_interval_us={} active_spin_window_us={} allow_poll_fallback={} "
                           "ops.get_event_id={} ops.wait_events={} tid={}",
                           static_cast<void*>(this),
                           thread_count,
                           cfg_.poll_interval_us,
                           cfg_.active_spin_window_us,
                           cfg_.allow_poll_fallback,
                           static_cast<bool>(ops_.get_event_id),
                           static_cast<bool>(ops_.wait_events),
                           common::get_tid());
    }

    void stop() override
    {
        ROCP_ERROR_IF(ioctl_monitor_diag_enabled())
            << fmt::format("DEBUG: ioctl monitor stop requested monitor={} tid={}",
                           static_cast<void*>(this),
                           common::get_tid());

        {
            std::lock_guard<std::mutex> lock(lifecycle_mutex_);
            if(!running_.exchange(false)) return;
        }

        size_t remaining_subscriptions = 0;
        {
            std::lock_guard<std::mutex> lock(subscriptions_mutex_);
            remaining_subscriptions = subscriptions_.size();
        }

        size_t queued_callbacks = 0;
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            queued_callbacks = callback_queue_.size();
        }
        queue_cv_.notify_all();

        if(detector_thread_.joinable()) detector_thread_.join();
        for(auto& worker : worker_threads_)
        {
            if(worker.joinable()) worker.join();
        }
        worker_threads_.clear();

        size_t cleared_subscriptions = 0;
        {
            std::lock_guard<std::mutex> lock(subscriptions_mutex_);
            cleared_subscriptions = subscriptions_.size();
            subscriptions_.clear();
        }

        size_t cleared_callbacks = 0;
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            cleared_callbacks = callback_queue_.size();
            callback_queue_.clear();
        }

        ROCP_ERROR_IF(ioctl_monitor_diag_enabled())
            << fmt::format("DEBUG: ioctl monitor stop complete monitor={} "
                           "remaining_subscriptions={} queued_callbacks={} "
                           "cleared_subscriptions={} cleared_callbacks={} tid={}",
                           static_cast<void*>(this),
                           remaining_subscriptions,
                           queued_callbacks,
                           cleared_subscriptions,
                           cleared_callbacks,
                           common::get_tid());
    }

    signal_subscription_id_t subscribe(hsa_signal_t              signal,
                                       hsa_signal_condition_t    condition,
                                       hsa_signal_value_t        compare_value,
                                       signal_monitor_callback_t cb) override
    {
        const auto id = next_id_.fetch_add(1);

        Subscription subscription{};
        subscription.signal        = signal;
        subscription.condition     = condition;
        subscription.compare_value = compare_value;
        subscription.callback      = std::move(cb);
        if(ops_.get_event_id)
        {
            subscription.has_event_id = ops_.get_event_id(signal, subscription.event_id);
        }

        std::lock_guard<std::mutex> lock(subscriptions_mutex_);
        subscriptions_.emplace(id, std::move(subscription));

        auto it = subscriptions_.find(id);
        ROCP_ERROR_IF(ioctl_monitor_diag_enabled())
            << fmt::format("DEBUG: ioctl monitor subscribe monitor={} id={} signal_handle={} "
                           "condition={} compare_value={} has_event_id={} event_id={} "
                           "subscription_count={} tid={}",
                           static_cast<void*>(this),
                           id,
                           signal.handle,
                           to_string(condition),
                           compare_value,
                           (it != subscriptions_.end()) ? it->second.has_event_id : false,
                           (it != subscriptions_.end()) ? it->second.event_id : 0,
                           subscriptions_.size(),
                           common::get_tid());
        return id;
    }

    bool unsubscribe(signal_subscription_id_t id) override
    {
        bool   erased             = false;
        size_t subscription_count = 0;
        {
            std::lock_guard<std::mutex> lock(subscriptions_mutex_);
            erased             = (subscriptions_.erase(id) > 0);
            subscription_count = subscriptions_.size();
        }

        if(!erased)
        {
            ROCP_ERROR_IF(ioctl_monitor_diag_enabled())
                << fmt::format("DEBUG: ioctl monitor unsubscribe miss monitor={} id={} "
                               "subscription_count={} tid={}",
                               static_cast<void*>(this),
                               id,
                               subscription_count,
                               common::get_tid());
            return false;
        }

        std::lock_guard<std::mutex> lock(queue_mutex_);
        if(callback_queue_.empty())
        {
            ROCP_ERROR_IF(ioctl_monitor_diag_enabled())
                << fmt::format("DEBUG: ioctl monitor unsubscribe monitor={} id={} "
                               "subscription_count={} queue_size=0 tid={}",
                               static_cast<void*>(this),
                               id,
                               subscription_count,
                               common::get_tid());
            return true;
        }

        std::deque<CallbackItem> remaining;
        remaining.swap(callback_queue_);
        for(const auto& item : remaining)
        {
            if(item.id != id) callback_queue_.push_back(item);
        }
        ROCP_ERROR_IF(ioctl_monitor_diag_enabled())
            << fmt::format("DEBUG: ioctl monitor unsubscribe monitor={} id={} "
                           "subscription_count={} queue_size={} tid={}",
                           static_cast<void*>(this),
                           id,
                           subscription_count,
                           callback_queue_.size(),
                           common::get_tid());
        return true;
    }

private:
    void log_subscription_snapshot(const char* reason, size_t max_entries = 8)
    {
        if(!ioctl_monitor_diag_enabled()) return;

        std::lock_guard<std::mutex> lock(subscriptions_mutex_);

        std::vector<std::string> entries = {};
        entries.reserve(std::min(max_entries, subscriptions_.size()));
        size_t count = 0;
        for(const auto& [id, subscription] : subscriptions_)
        {
            auto current_value = ops_.load(subscription.signal);
            entries.emplace_back(fmt::format("{{id={}, signal={}, pending={}, value={}, "
                                             "condition={}, compare={}, has_event_id={}, "
                                             "event_id={}}}",
                                             id,
                                             subscription.signal.handle,
                                             subscription.pending,
                                             current_value,
                                             to_string(subscription.condition),
                                             subscription.compare_value,
                                             subscription.has_event_id,
                                             subscription.event_id));
            ++count;
            if(count >= max_entries) break;
        }

        ROCP_ERROR << fmt::format(
            "DEBUG: ioctl monitor snapshot reason={} subscription_count={} entries={}",
            reason,
            subscriptions_.size(),
            fmt::join(entries, ", "));
    }

    void collect_ready(std::vector<CallbackItem>& ready, std::vector<uint32_t>* event_ids)
    {
        if(!running_.load()) return;

        std::lock_guard<std::mutex> lock(subscriptions_mutex_);
        if(!running_.load()) return;

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
            ROCP_ERROR_IF(ioctl_monitor_diag_enabled())
                << fmt::format("DEBUG: ioctl monitor ready monitor={} id={} signal_handle={} "
                               "current_value={} condition={} compare_value={} has_event_id={} "
                               "event_id={} pending=true tid={}",
                               static_cast<void*>(this),
                               id,
                               subscription.signal.handle,
                               current_value,
                               to_string(subscription.condition),
                               subscription.compare_value,
                               subscription.has_event_id,
                               subscription.event_id,
                               common::get_tid());
        }
    }

    void enqueue_ready(const std::vector<CallbackItem>& ready)
    {
        if(ready.empty() || !running_.load()) return;

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            for(const auto& item : ready)
                callback_queue_.push_back(item);
            ROCP_ERROR_IF(ioctl_monitor_diag_enabled())
                << fmt::format("DEBUG: ioctl monitor enqueue monitor={} ready_count={} "
                               "queue_size={} tid={}",
                               static_cast<void*>(this),
                               ready.size(),
                               callback_queue_.size(),
                               common::get_tid());
        }
        queue_cv_.notify_all();
    }

    void detector_loop()
    {
        const auto poll_interval   = std::chrono::microseconds{cfg_.poll_interval_us};
        const auto spin_window     = std::chrono::microseconds{cfg_.active_spin_window_us};
        const auto wait_timeout_ms = std::max<uint32_t>((cfg_.poll_interval_us + 999u) / 1000u, 1u);
        uint64_t   idle_sleep_cycles = 0;
        ROCP_ERROR_IF(ioctl_monitor_diag_enabled()) << fmt::format(
            "DEBUG: ioctl monitor detector loop start monitor={} poll_interval_us={} "
            "active_spin_window_us={} wait_timeout_ms={} tid={}",
            static_cast<void*>(this),
            cfg_.poll_interval_us,
            cfg_.active_spin_window_us,
            wait_timeout_ms,
            common::get_tid());

        while(running_.load())
        {
            std::vector<CallbackItem> ready = {};
            collect_ready(ready, nullptr);
            if(!ready.empty())
            {
                idle_sleep_cycles = 0;
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
                    idle_sleep_cycles = 0;
                    enqueue_ready(ready);
                    continue;
                }
            }

            std::vector<uint32_t> event_ids = {};
            ready.clear();
            collect_ready(ready, &event_ids);
            if(!ready.empty())
            {
                idle_sleep_cycles = 0;
                enqueue_ready(ready);
                continue;
            }

            if(!running_.load()) break;

            if(!event_ids.empty() && ops_.wait_events)
            {
                idle_sleep_cycles = 0;
                auto _wait_result = ops_.wait_events(event_ids, wait_timeout_ms);
                ROCP_ERROR_IF(ioctl_monitor_diag_enabled())
                    << fmt::format("DEBUG: ioctl monitor detector waited via ioctl monitor={} "
                                   "event_count={} result={} tid={}",
                                   static_cast<void*>(this),
                                   event_ids.size(),
                                   _wait_result,
                                   common::get_tid());
            }
            else if(poll_interval.count() > 0)
            {
                ++idle_sleep_cycles;
                ROCP_ERROR_IF(ioctl_monitor_diag_enabled() &&
                              (idle_sleep_cycles == 1 || (idle_sleep_cycles % 10000) == 0))
                    << fmt::format("DEBUG: ioctl monitor detector timed sleep monitor={} "
                                   "poll_interval_us={} event_count={} wait_events_supported={} "
                                   "idle_sleep_cycles={} tid={}",
                                   static_cast<void*>(this),
                                   cfg_.poll_interval_us,
                                   event_ids.size(),
                                   static_cast<bool>(ops_.wait_events),
                                   idle_sleep_cycles,
                                   common::get_tid());

                if(ioctl_monitor_diag_enabled() &&
                   (idle_sleep_cycles == 1 || (idle_sleep_cycles % 10000) == 0))
                {
                    log_subscription_snapshot("timed-sleep");
                }
                std::this_thread::sleep_for(poll_interval);
            }
            else
            {
                std::this_thread::yield();
            }
        }

        ROCP_ERROR_IF(ioctl_monitor_diag_enabled())
            << fmt::format("DEBUG: ioctl monitor detector loop exit monitor={} tid={}",
                           static_cast<void*>(this),
                           common::get_tid());
    }

    void callback_loop()
    {
        ROCP_ERROR_IF(ioctl_monitor_diag_enabled())
            << fmt::format("DEBUG: ioctl monitor callback loop start monitor={} tid={}",
                           static_cast<void*>(this),
                           common::get_tid());

        while(true)
        {
            CallbackItem item{};
            size_t       queue_size_after_pop = 0;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_cv_.wait(lock,
                               [&]() { return !running_.load() || !callback_queue_.empty(); });
                if(!running_.load() && callback_queue_.empty()) return;

                item = callback_queue_.front();
                callback_queue_.pop_front();
                queue_size_after_pop = callback_queue_.size();
            }

            ROCP_ERROR_IF(ioctl_monitor_diag_enabled())
                << fmt::format("DEBUG: ioctl monitor callback dequeue monitor={} id={} value={} "
                               "queue_size_after_pop={} tid={}",
                               static_cast<void*>(this),
                               item.id,
                               item.current_value,
                               queue_size_after_pop,
                               common::get_tid());

            signal_monitor_callback_t callback;
            {
                std::lock_guard<std::mutex> lock(subscriptions_mutex_);
                auto                        it = subscriptions_.find(item.id);
                if(it == subscriptions_.end())
                {
                    ROCP_ERROR_IF(ioctl_monitor_diag_enabled()) << fmt::format(
                        "DEBUG: ioctl monitor callback missing subscription monitor={} "
                        "id={} tid={}",
                        static_cast<void*>(this),
                        item.id,
                        common::get_tid());
                    continue;
                }
                callback = it->second.callback;
            }

            bool keep = false;
            if(callback)
            {
                try
                {
                    keep = callback(item.current_value);
                } catch(...)
                {
                    keep = false;
                }
            }

            ROCP_ERROR_IF(ioctl_monitor_diag_enabled())
                << fmt::format("DEBUG: ioctl monitor callback invoke monitor={} id={} value={} "
                               "keep={} tid={}",
                               static_cast<void*>(this),
                               item.id,
                               item.current_value,
                               keep,
                               common::get_tid());

            std::lock_guard<std::mutex> lock(subscriptions_mutex_);
            auto                        it = subscriptions_.find(item.id);
            if(it == subscriptions_.end())
            {
                ROCP_ERROR_IF(ioctl_monitor_diag_enabled())
                    << fmt::format("DEBUG: ioctl monitor callback post-invoke missing subscription "
                                   "monitor={} id={} tid={}",
                                   static_cast<void*>(this),
                                   item.id,
                                   common::get_tid());
                continue;
            }

            if(!keep)
            {
                ROCP_ERROR_IF(ioctl_monitor_diag_enabled())
                    << fmt::format("DEBUG: ioctl monitor callback unsubscribe-after-callback "
                                   "monitor={} id={} tid={}",
                                   static_cast<void*>(this),
                                   item.id,
                                   common::get_tid());
                subscriptions_.erase(it);
            }
            else
            {
                it->second.pending = false;
                ROCP_ERROR_IF(ioctl_monitor_diag_enabled())
                    << fmt::format("DEBUG: ioctl monitor callback keep monitor={} id={} "
                                   "pending=false tid={}",
                                   static_cast<void*>(this),
                                   item.id,
                                   common::get_tid());
            }
        }
    }

    SignalMonitorConfig cfg_ = {};
    SignalMonitorOps    ops_ = {};

    std::atomic<bool>                     running_{false};
    std::atomic<signal_subscription_id_t> next_id_{1};

    std::mutex               lifecycle_mutex_;
    std::thread              detector_thread_;
    std::vector<std::thread> worker_threads_;

    std::mutex                                                 subscriptions_mutex_;
    std::unordered_map<signal_subscription_id_t, Subscription> subscriptions_;

    std::mutex               queue_mutex_;
    std::condition_variable  queue_cv_;
    std::deque<CallbackItem> callback_queue_;
};
}  // namespace

namespace detail
{
std::shared_ptr<SignalMonitor>
create_ioctl_signal_monitor(const SignalMonitorConfig& cfg, SignalMonitorOps ops)
{
    if(!ops.load) ops.load = default_load_signal;

    // Populate default event-id support only when the runtime table provides it.
    if(!ops.get_event_id && has_event_id_support(get_amd_ext_table()))
    {
        ops.get_event_id = default_get_event_id;
    }

    // Use KFD wait-events if available; otherwise detector loop will remain in timed polling mode.
    if(!ops.wait_events && get_kfd_wait_fd() >= 0)
    {
        ops.wait_events = default_wait_events;
    }

    ROCP_ERROR_IF(ioctl_monitor_diag_enabled())
        << fmt::format("DEBUG: create_ioctl_signal_monitor poll_interval_us={} "
                       "active_spin_window_us={} callback_threads={} allow_poll_fallback={} "
                       "ops.get_event_id={} ops.wait_events={}",
                       cfg.poll_interval_us,
                       cfg.active_spin_window_us,
                       cfg.callback_threads,
                       cfg.allow_poll_fallback,
                       static_cast<bool>(ops.get_event_id),
                       static_cast<bool>(ops.wait_events));

    return std::make_shared<IoctlSignalMonitor>(cfg, std::move(ops));
}
}  // namespace detail
}  // namespace rocprofiler::hsa
