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

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

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
}  // namespace rocprofiler::hsa::test
