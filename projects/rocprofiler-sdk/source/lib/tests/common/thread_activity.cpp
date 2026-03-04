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

#include "lib/common/thread_activity.hpp"

#include <fmt/format.h>
#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <random>
#include <sstream>
#include <string>
#include <thread>

// namespace
// {
// }  // namespace

TEST(common, thread_activity)
{
    namespace thread_activity = ::rocprofiler::common::thread_activity;

    constexpr auto num_threads  = 4UL;
    constexpr auto min_interval = std::chrono::milliseconds{10};
    constexpr auto timeout      = std::chrono::milliseconds{500};

    const auto _initial_tasks = thread_activity::get_tasks(getpid());

    auto get_elapsed_time = [](auto _start_time) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - _start_time);
    };

    auto _busy_thread = [](auto& _started_threads, auto& _flag, std::atomic<uint64_t>& _result) {
        // increment started threads count so main thread knows when all threads are running
        ++_started_threads;

        // create rng to do busy work
        auto _random_gen = std::mt19937_64{std::random_device{}()};
        auto _rng = std::uniform_int_distribution<uint64_t>{std::numeric_limits<uint8_t>::max(),
                                                            std::numeric_limits<uint16_t>::max()};
        auto _random_value = [&_random_gen, &_rng]() { return _rng(_random_gen); }();

        // update shared value until flag is cleared
        while(_flag.load(std::memory_order_relaxed))
            _result += _random_value;
    };

    auto _sleeping_thread = [](auto&                     _started_threads,
                               auto&                     _flag,
                               std::chrono::milliseconds _sleep_duration) {
        // increment started threads count so main thread knows when all threads are running
        ++_started_threads;

        while(_flag.load(std::memory_order_relaxed))
            std::this_thread::yield();  // yield until flag is cleared to simulate sleeping thread

        // sleep for specified duration to simulate sleeping thread
        std::this_thread::sleep_for(_sleep_duration);
    };

    //=============================================================================================//
    //
    //      Busy thread tests
    //
    //=============================================================================================//

    auto _busy_value           = std::atomic<uint64_t>{0};
    auto _busy_flag            = std::atomic<bool>{true};
    auto _busy_threads         = std::vector<std::thread>{};
    auto _busy_started_threads = std::atomic<uint64_t>{0};
    for(size_t i = 0; i < num_threads; ++i)
    {
        _busy_threads.emplace_back(
            [&]() { _busy_thread(_busy_started_threads, _busy_flag, _busy_value); });
    }

    // wait for all threads to start and be detected in get_tasks before proceeding
    while(_busy_started_threads.load(std::memory_order_relaxed) < num_threads)
    {
        std::this_thread::sleep_for(std::chrono::microseconds{100});
    }

    // get current tasks again to include the newly started threads and remove the initial tasks
    // from the current tasks so we can focus on the new threads we just started
    auto _busy_tasks = thread_activity::get_tasks(getpid(), _initial_tasks);

    ASSERT_EQ(_busy_tasks.size(), num_threads);

    {
        auto _busy_start = std::chrono::steady_clock::now();
        auto _busy_activity =
            thread_activity::poll_tasks(_busy_tasks, nullptr, min_interval, timeout);
        auto _busy_elapsed = get_elapsed_time(_busy_start);

        EXPECT_GE(_busy_elapsed.count(), timeout.count());
        EXPECT_LE(_busy_activity.size(), num_threads);
        for(const auto& [task, status] : _busy_activity)
        {
            EXPECT_EQ(_initial_tasks.count(task), 0u);
            EXPECT_GT(_busy_tasks.count(task), 0u);
            EXPECT_GT(status, thread_activity::status::Sleeping)
                << fmt::format("Task {} status: {}", task, status);
            EXPECT_LT(status, thread_activity::status::Unknown)
                << fmt::format("Task {} status: {}", task, status);
        }
    }

    // release the threads
    _busy_flag.store(false, std::memory_order_relaxed);

    {
        auto _busy_start = std::chrono::steady_clock::now();
        auto _busy_activity =
            thread_activity::poll_tasks(_busy_tasks, nullptr, min_interval, timeout);
        auto _busy_elapsed = get_elapsed_time(_busy_start);

        EXPECT_LT(_busy_elapsed.count(), timeout.count());
        EXPECT_EQ(_busy_activity.size(), num_threads);
        for(const auto& [task, status] : _busy_activity)
        {
            EXPECT_EQ(_initial_tasks.count(task), 0u);
            EXPECT_GT(_busy_tasks.count(task), 0u);
            EXPECT_EQ(status, thread_activity::status::Gone)
                << fmt::format("Task {} status: {}", task, status);
        }
    }

    for(auto& itr : _busy_threads)
    {
        if(itr.joinable()) itr.join();
    }

    //=============================================================================================//
    //
    //      Sleeping thread tests
    //
    //=============================================================================================//

    auto _sleeping_flag            = std::atomic<bool>{true};
    auto _sleeping_started_threads = std::atomic<uint64_t>{0};
    auto _sleeping_threads         = std::vector<std::thread>{};

    for(size_t i = 0; i < num_threads; ++i)
    {
        _sleeping_threads.emplace_back(
            [&]() { _sleeping_thread(_sleeping_started_threads, _sleeping_flag, timeout * 2); });
    }

    // wait for all threads to start and be detected in get_tasks before proceeding
    while(_sleeping_started_threads.load(std::memory_order_relaxed) < num_threads)
        std::this_thread::yield();

    // set flag to false to exit loop and enter timed sleep in sleeping thread function
    _sleeping_flag.store(false, std::memory_order_relaxed);

    // get current tasks again to include the newly started threads and remove the initial tasks
    // from the current tasks so we can focus on the new threads we just started
    auto _sleeping_tasks = thread_activity::get_tasks(getpid(), _initial_tasks);

    ASSERT_EQ(_sleeping_tasks.size(), num_threads);

    {
        auto _sleeping_start = std::chrono::steady_clock::now();
        auto _sleeping_activity =
            thread_activity::poll_tasks(_sleeping_tasks, nullptr, min_interval, timeout);
        auto _sleeping_elapsed = get_elapsed_time(_sleeping_start);

        EXPECT_GE(_sleeping_elapsed.count(), timeout.count());
        EXPECT_LE(_sleeping_activity.size(), num_threads);
        for(const auto& [task, status] : _sleeping_activity)
        {
            EXPECT_EQ(_initial_tasks.count(task), 0u);
            EXPECT_GT(_sleeping_tasks.count(task), 0u);
            EXPECT_GT(status, thread_activity::status::StoppedOrTraced)
                << fmt::format("Task {} status: {}", task, status);
            EXPECT_LT(status, thread_activity::status::ActiveOnCPU)
                << fmt::format("Task {} status: {}", task, status);
        }
    }

    {
        auto _sleeping_activity =
            thread_activity::poll_tasks(_sleeping_tasks, nullptr, min_interval, timeout);

        EXPECT_EQ(_sleeping_activity.size(), num_threads);
        for(const auto& [task, status] : _sleeping_activity)
        {
            EXPECT_EQ(_initial_tasks.count(task), 0u);
            EXPECT_GT(_sleeping_tasks.count(task), 0u);
            EXPECT_EQ(status, thread_activity::status::Gone)
                << fmt::format("Task {} status: {}", task, status);
        }
    }

    // make sure threads are all joined
    for(auto& itr : _sleeping_threads)
    {
        if(itr.joinable()) itr.join();
    }
}
