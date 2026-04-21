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
#include "lib/rocprofiler-sdk/hsa/queue_signal_subscription.hpp"
#include "lib/common/environment.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

namespace rocprofiler::hsa::test
{
TEST(signal_monitor, evaluate_condition_eq)
{
    EXPECT_TRUE(evaluate_signal_condition(HSA_SIGNAL_CONDITION_EQ, 0, 0));
    EXPECT_FALSE(evaluate_signal_condition(HSA_SIGNAL_CONDITION_EQ, -1, 0));
}

TEST(signal_monitor, evaluate_condition_lt)
{
    EXPECT_TRUE(evaluate_signal_condition(HSA_SIGNAL_CONDITION_LT, 0, 1));
    EXPECT_FALSE(evaluate_signal_condition(HSA_SIGNAL_CONDITION_LT, 2, 1));
}

TEST(signal_monitor, parse_backend_env)
{
    {
        common::env_store env{{{"ROCPROF_SIGNAL_MONITOR_BACKEND", "poll", 1}}};
        env.push();
        EXPECT_EQ(parse_signal_monitor_backend_env(), SignalMonitorBackend::poll);
        env.pop();
    }

    {
        common::env_store env{{{"ROCPROF_SIGNAL_MONITOR_BACKEND", "ioctl", 1}}};
        env.push();
        EXPECT_EQ(parse_signal_monitor_backend_env(), SignalMonitorBackend::ioctl);
        env.pop();
    }

    {
        common::env_store env{{{"ROCPROF_SIGNAL_MONITOR_BACKEND", "invalid", 1}}};
        env.push();
        EXPECT_EQ(parse_signal_monitor_backend_env(), SignalMonitorBackend::auto_select);
        env.pop();
    }

    {
        common::env_store env{{{"ROCPROF_SIGNAL_MONITOR_BACKEND", "", 1}}};
        env.push();
        EXPECT_EQ(parse_signal_monitor_backend_env(), SignalMonitorBackend::auto_select);
        env.pop();
    }
}

TEST(signal_monitor, polling_fires_once_for_eq)
{
    SignalMonitorConfig cfg{};
    cfg.poll_interval_us = 1;
    cfg.callback_threads = 1;

    std::atomic<hsa_signal_value_t> value{0};
    SignalMonitorOps ops{};
    ops.load = [&](hsa_signal_t) { return value.load(); };

    auto monitor = create_signal_monitor(SignalMonitorBackend::poll, cfg, ops);
    ASSERT_NE(monitor, nullptr);

    hsa_signal_t signal{};
    signal.handle = 1;

    std::mutex              mutex;
    std::condition_variable cv;
    int                     calls = 0;

    monitor->subscribe(signal, HSA_SIGNAL_CONDITION_EQ, 1, [&](hsa_signal_value_t) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            ++calls;
        }
        cv.notify_all();
        return false;
    });

    monitor->start();
    value.store(1);

    {
        std::unique_lock<std::mutex> lock(mutex);
        ASSERT_TRUE(cv.wait_for(lock, std::chrono::milliseconds{50}, [&]() { return calls == 1; }));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds{5});
    {
        std::lock_guard<std::mutex> lock(mutex);
        EXPECT_EQ(calls, 1);
    }

    monitor->stop();
}

TEST(signal_monitor, polling_unsubscribe_prevents_callback)
{
    SignalMonitorConfig cfg{};
    cfg.poll_interval_us = 1;
    cfg.callback_threads = 1;

    std::atomic<hsa_signal_value_t> value{0};
    SignalMonitorOps ops{};
    ops.load = [&](hsa_signal_t) { return value.load(); };

    auto monitor = create_signal_monitor(SignalMonitorBackend::poll, cfg, ops);
    ASSERT_NE(monitor, nullptr);

    hsa_signal_t signal{};
    signal.handle = 2;

    std::mutex              mutex;
    std::condition_variable cv;
    int                     calls = 0;

    auto id = monitor->subscribe(signal, HSA_SIGNAL_CONDITION_EQ, 1, [&](hsa_signal_value_t) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            ++calls;
        }
        cv.notify_all();
        return false;
    });

    ASSERT_TRUE(monitor->unsubscribe(id));

    monitor->start();
    value.store(1);

    {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait_for(lock, std::chrono::milliseconds{20}, [&]() { return calls != 0; });
    }

    {
        std::lock_guard<std::mutex> lock(mutex);
        EXPECT_EQ(calls, 0);
    }

    monitor->stop();
}

TEST(signal_monitor, ioctl_rechecks_after_waiting_eq)
{
    SignalMonitorConfig cfg{};
    cfg.poll_interval_us = 1;
    cfg.active_spin_window_us = 0;
    cfg.callback_threads = 1;
    cfg.allow_poll_fallback = false;

    std::atomic<hsa_signal_value_t> value{1};
    std::atomic<int>                wait_calls{0};
    SignalMonitorOps                ops{};
    ops.load = [&](hsa_signal_t) { return value.load(); };
    ops.get_event_id = [](hsa_signal_t signal, uint32_t& event_id) {
        event_id = static_cast<uint32_t>(signal.handle);
        return true;
    };
    ops.wait_events = [&](const std::vector<uint32_t>&, uint32_t) {
        wait_calls.fetch_add(1);
        value.store(-1);
        return true;
    };

    auto monitor = create_signal_monitor(SignalMonitorBackend::ioctl, cfg, ops);
    ASSERT_NE(monitor, nullptr);

    hsa_signal_t signal{};
    signal.handle = 0xA;

    std::mutex              mutex;
    std::condition_variable cv;
    int                     calls = 0;

    monitor->subscribe(signal, HSA_SIGNAL_CONDITION_EQ, -1, [&](hsa_signal_value_t) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            ++calls;
        }
        cv.notify_all();
        return false;
    });

    monitor->start();
    {
        std::unique_lock<std::mutex> lock(mutex);
        ASSERT_TRUE(cv.wait_for(lock, std::chrono::milliseconds{100}, [&]() { return calls == 1; }));
    }
    monitor->stop();

    EXPECT_GE(wait_calls.load(), 1);
}

TEST(signal_monitor, ioctl_backend_without_wait_ops_no_fallback_returns_null)
{
    SignalMonitorConfig cfg{};
    cfg.allow_poll_fallback = false;

    SignalMonitorOps ops{};
    ops.load = [](hsa_signal_t) { return hsa_signal_value_t{0}; };

    auto monitor = create_signal_monitor(SignalMonitorBackend::ioctl, cfg, ops);
    EXPECT_EQ(monitor, nullptr);
}

TEST(signal_monitor, ioctl_backend_without_wait_ops_falls_back_to_poll)
{
    SignalMonitorConfig cfg{};
    cfg.poll_interval_us = 1;
    cfg.callback_threads = 1;
    cfg.allow_poll_fallback = true;

    std::atomic<hsa_signal_value_t> value{0};
    SignalMonitorOps                ops{};
    ops.load = [&](hsa_signal_t) { return value.load(); };

    auto monitor = create_signal_monitor(SignalMonitorBackend::ioctl, cfg, ops);
    ASSERT_NE(monitor, nullptr);

    hsa_signal_t signal{};
    signal.handle = 0xB;

    std::mutex              mutex;
    std::condition_variable cv;
    int                     calls = 0;

    monitor->subscribe(signal, HSA_SIGNAL_CONDITION_EQ, 1, [&](hsa_signal_value_t) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            ++calls;
        }
        cv.notify_all();
        return false;
    });

    monitor->start();
    value.store(1);

    {
        std::unique_lock<std::mutex> lock(mutex);
        ASSERT_TRUE(cv.wait_for(lock, std::chrono::milliseconds{50}, [&]() { return calls == 1; }));
    }

    monitor->stop();
}

TEST(signal_monitor, auto_backend_prefers_ioctl_when_available)
{
    SignalMonitorConfig cfg{};
    cfg.poll_interval_us = 10;
    cfg.active_spin_window_us = 0;
    cfg.callback_threads = 1;

    std::atomic<hsa_signal_value_t> value{1};
    std::atomic<int>                wait_calls{0};
    SignalMonitorOps                ops{};
    ops.load = [&](hsa_signal_t) { return value.load(); };
    ops.get_event_id = [](hsa_signal_t signal, uint32_t& event_id) {
        event_id = static_cast<uint32_t>(signal.handle);
        return true;
    };
    ops.wait_events = [&](const std::vector<uint32_t>&, uint32_t) {
        wait_calls.fetch_add(1);
        value.store(-1);
        return true;
    };

    auto monitor = create_signal_monitor(SignalMonitorBackend::auto_select, cfg, ops);
    ASSERT_NE(monitor, nullptr);

    hsa_signal_t signal{};
    signal.handle = 0xC;

    std::mutex              mutex;
    std::condition_variable cv;
    int                     calls = 0;

    monitor->subscribe(signal, HSA_SIGNAL_CONDITION_EQ, -1, [&](hsa_signal_value_t) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            ++calls;
        }
        cv.notify_all();
        return false;
    });

    monitor->start();
    {
        std::unique_lock<std::mutex> lock(mutex);
        ASSERT_TRUE(cv.wait_for(lock, std::chrono::milliseconds{100}, [&]() { return calls == 1; }));
    }
    monitor->stop();

    EXPECT_GE(wait_calls.load(), 1);
}

TEST(signal_monitor, auto_backend_falls_back_to_poll)
{
    SignalMonitorConfig cfg{};
    cfg.poll_interval_us = 1;
    cfg.callback_threads = 1;

    std::atomic<hsa_signal_value_t> value{0};
    SignalMonitorOps                ops{};
    ops.load = [&](hsa_signal_t) { return value.load(); };

    auto monitor = create_signal_monitor(SignalMonitorBackend::auto_select, cfg, ops);
    ASSERT_NE(monitor, nullptr);

    hsa_signal_t signal{};
    signal.handle = 0xD;

    std::mutex              mutex;
    std::condition_variable cv;
    int                     calls = 0;

    monitor->subscribe(signal, HSA_SIGNAL_CONDITION_EQ, 1, [&](hsa_signal_value_t) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            ++calls;
        }
        cv.notify_all();
        return false;
    });

    monitor->start();
    value.store(1);
    {
        std::unique_lock<std::mutex> lock(mutex);
        ASSERT_TRUE(cv.wait_for(lock, std::chrono::milliseconds{50}, [&]() { return calls == 1; }));
    }
    monitor->stop();
}

TEST(signal_monitor, queue_adapter_invokes_async_handler_once)
{
    std::atomic<int> calls{0};
    auto             cb = [&](hsa_signal_value_t) {
        calls.fetch_add(1);
        return false;
    };

    auto ok = QueueSignalSubscription::test_invoke_for_unit(cb, -1);
    EXPECT_TRUE(ok);
    EXPECT_EQ(calls.load(), 1);
}

TEST(signal_monitor, queue_completion_requires_negative_one_transition)
{
    SignalMonitorConfig cfg{};
    cfg.poll_interval_us = 1;
    cfg.callback_threads = 1;

    std::atomic<hsa_signal_value_t> value{1};
    SignalMonitorOps                ops{};
    ops.load = [&](hsa_signal_t) { return value.load(); };

    auto monitor = create_signal_monitor(SignalMonitorBackend::poll, cfg, ops);
    ASSERT_NE(monitor, nullptr);

    hsa_signal_t signal{};
    signal.handle = 0xE;

    std::mutex              mutex;
    std::condition_variable cv;
    int                     calls = 0;

    monitor->subscribe(signal, HSA_SIGNAL_CONDITION_EQ, -1, [&](hsa_signal_value_t) {
        {
            std::lock_guard<std::mutex> lock(mutex);
            ++calls;
        }
        cv.notify_all();
        return false;
    });

    monitor->start();

    value.store(0);
    {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait_for(lock, std::chrono::milliseconds{20}, [&]() { return calls != 0; });
    }
    {
        std::lock_guard<std::mutex> lock(mutex);
        EXPECT_EQ(calls, 0);
    }

    value.store(-1);
    {
        std::unique_lock<std::mutex> lock(mutex);
        ASSERT_TRUE(cv.wait_for(lock, std::chrono::milliseconds{50}, [&]() { return calls == 1; }));
    }

    monitor->stop();
}
}  // namespace rocprofiler::hsa::test
