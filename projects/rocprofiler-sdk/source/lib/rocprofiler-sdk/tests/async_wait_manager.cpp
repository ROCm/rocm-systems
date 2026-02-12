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

#include "lib/rocprofiler-sdk/hsa/async_wait_manager.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

using namespace rocprofiler::hsa;

class async_wait_manager_test : public ::testing::Test
{
protected:
    void TearDown() override { reset_async_shutdown(); }
};

// --- async_wait_manager basic state tests ---

TEST_F(async_wait_manager_test, initial_state_not_shutdown) { EXPECT_FALSE(is_async_shutdown()); }

TEST_F(async_wait_manager_test, notify_sets_shutdown_flag)
{
    notify_async_shutdown();
    EXPECT_TRUE(is_async_shutdown());
}

TEST_F(async_wait_manager_test, reset_clears_shutdown_flag)
{
    notify_async_shutdown();
    EXPECT_TRUE(is_async_shutdown());
    reset_async_shutdown();
    EXPECT_FALSE(is_async_shutdown());
}

// --- Atomic wait tests ---

TEST_F(async_wait_manager_test, atomic_completed_immediately)
{
    std::atomic<int> value{42};
    auto             result = wait_or_shutdown(value, 42, "test::atomic_completed_immediately");
    EXPECT_EQ(result, wait_result::completed);
}

TEST_F(async_wait_manager_test, atomic_completed_by_thread)
{
    std::atomic<int> value{0};

    std::thread setter([&value]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        value.store(1, std::memory_order_release);
    });

    auto result = wait_or_shutdown(value, 1, "test::atomic_completed_by_thread", 5'000'000'000ULL);
    EXPECT_EQ(result, wait_result::completed);

    setter.join();
}

TEST_F(async_wait_manager_test, atomic_timeout)
{
    std::atomic<int> value{0};

    auto result = wait_or_shutdown(value, 1, "test::atomic_timeout", 500'000'000ULL);
    EXPECT_EQ(result, wait_result::timeout);
}

TEST_F(async_wait_manager_test, atomic_shutdown_interrupts)
{
    std::atomic<int> value{0};

    std::thread shutdown_trigger([]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        notify_async_shutdown();
    });

    auto result = wait_or_shutdown(value, 1, "test::atomic_shutdown_interrupts", 30'000'000'000ULL);
    EXPECT_EQ(result, wait_result::shutdown);

    shutdown_trigger.join();
}

// --- Condition variable wait tests ---

TEST_F(async_wait_manager_test, cv_completed_immediately)
{
    std::mutex              mtx;
    std::condition_variable cv;
    bool                    ready = true;

    std::unique_lock<std::mutex> lock(mtx);
    auto                         result = wait_or_shutdown(
        cv, lock, [&] { return ready; }, "test::cv_completed_immediately");
    EXPECT_EQ(result, wait_result::completed);
}

TEST_F(async_wait_manager_test, cv_completed_by_thread)
{
    std::mutex              mtx;
    std::condition_variable cv;
    bool                    ready = false;

    std::thread notifier([&]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        {
            std::lock_guard<std::mutex> lock(mtx);
            ready = true;
        }
        cv.notify_one();
    });

    std::unique_lock<std::mutex> lock(mtx);
    auto                         result = wait_or_shutdown(
        cv, lock, [&] { return ready; }, "test::cv_completed_by_thread", 5'000'000'000ULL);
    EXPECT_EQ(result, wait_result::completed);

    notifier.join();
}

TEST_F(async_wait_manager_test, cv_timeout)
{
    std::mutex              mtx;
    std::condition_variable cv;
    bool                    ready = false;

    std::unique_lock<std::mutex> lock(mtx);
    auto                         result = wait_or_shutdown(
        cv, lock, [&] { return ready; }, "test::cv_timeout", 500'000'000ULL);
    EXPECT_EQ(result, wait_result::timeout);
}

TEST_F(async_wait_manager_test, cv_shutdown_interrupts)
{
    std::mutex              mtx;
    std::condition_variable cv;
    bool                    ready = false;

    std::thread shutdown_trigger([]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        notify_async_shutdown();
    });

    std::unique_lock<std::mutex> lock(mtx);
    auto                         result = wait_or_shutdown(
        cv, lock, [&] { return ready; }, "test::cv_shutdown_interrupts", 30'000'000'000ULL);
    EXPECT_EQ(result, wait_result::shutdown);

    shutdown_trigger.join();
}
