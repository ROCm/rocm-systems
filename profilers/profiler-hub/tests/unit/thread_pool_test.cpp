// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/thread_pool.hpp"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <future>
#include <memory>
#include <stdexcept>
#include <stop_token>
#include <thread>

namespace
{

using profiler_hub::common::thread_pool;
using namespace std::chrono_literals;

bool
finishes_within(std::chrono::milliseconds limit, std::function<void()> fn)
{
    auto done = std::async(std::launch::async, std::move(fn));
    return done.wait_for(limit) == std::future_status::ready;
}

class blocker
{
public:
    thread_pool::task_fn task()
    {
        return [this](const std::stop_token&) {
            m_started = true;
            m_release.get_future().wait();
        };
    }

    void wait_until_started() const
    {
        while(!m_started)
        {
            std::this_thread::sleep_for(1ms);
        }
    }

    void release() { m_release.set_value(); }

private:
    std::atomic<bool>  m_started{ false };
    std::promise<void> m_release;
};

TEST(thread_pool_test, zero_threads_is_rejected)
{
    EXPECT_THROW(thread_pool{ 0 }, std::invalid_argument);
}

TEST(thread_pool_test, a_submitted_task_runs_and_wait_returns)
{
    thread_pool       pool{ 2 };
    std::atomic<bool> ran{ false };

    const auto handle = pool.submit([&](const std::stop_token&) { ran = true; });
    handle.wait();

    EXPECT_TRUE(ran);
}

TEST(thread_pool_test, many_tasks_all_run)
{
    thread_pool                           pool{ 4 };
    std::atomic<int>                      count{ 0 };
    std::vector<thread_pool::task_handle> handles;
    for(int i = 0; i < 200; ++i)
    {
        handles.push_back(pool.submit([&](const std::stop_token&) { ++count; }));
    }
    for(const auto& handle : handles)
    {
        handle.wait();
    }

    EXPECT_EQ(count, 200);
}

TEST(thread_pool_test, tasks_run_on_several_workers_at_once)
{
    thread_pool       pool{ 2 };
    std::atomic<int>  arrived{ 0 };
    std::atomic<bool> both_seen{ false };

    const auto task = [&](const std::stop_token&) {
        ++arrived;
        const auto deadline = std::chrono::steady_clock::now() + 2s;
        while(arrived < 2 && std::chrono::steady_clock::now() < deadline)
        {
            std::this_thread::sleep_for(1ms);
        }
        if(arrived == 2) both_seen = true;
    };
    const auto first  = pool.submit(task);
    const auto second = pool.submit(task);
    first.wait();
    second.wait();

    EXPECT_TRUE(both_seen);
}

TEST(thread_pool_test, cancelling_a_pending_task_skips_it)
{
    thread_pool       pool{ 1 };
    blocker           first;
    std::atomic<bool> second_ran{ false };

    const auto first_handle = pool.submit(first.task());
    first.wait_until_started();
    const auto second_handle =
        pool.submit([&](const std::stop_token&) { second_ran = true; });

    EXPECT_TRUE(second_handle.cancel());
    first.release();
    first_handle.wait();
    second_handle.wait();

    EXPECT_FALSE(second_ran);
}

TEST(thread_pool_test, cancelling_a_finished_task_reports_false)
{
    thread_pool pool{ 1 };

    const auto handle = pool.submit([](const std::stop_token&) {});
    handle.wait();

    EXPECT_FALSE(handle.cancel());
}

TEST(thread_pool_test, cancelling_a_running_task_requests_a_stop)
{
    thread_pool       pool{ 1 };
    std::atomic<bool> started{ false };
    std::atomic<bool> saw_stop{ false };

    const auto handle = pool.submit([&](const std::stop_token& token) {
        started = true;
        while(!token.stop_requested())
        {
            std::this_thread::sleep_for(1ms);
        }
        saw_stop = true;
    });
    while(!started)
    {
        std::this_thread::sleep_for(1ms);
    }

    EXPECT_TRUE(handle.cancel());
    handle.wait();

    EXPECT_TRUE(saw_stop);
}

TEST(thread_pool_test, a_throwing_task_does_not_stop_the_worker)
{
    thread_pool       pool{ 1 };
    std::atomic<bool> second_ran{ false };

    const auto first = pool.submit(
        [](const std::stop_token&) { throw std::runtime_error("task failed"); });
    first.wait();
    const auto second = pool.submit([&](const std::stop_token&) { second_ran = true; });
    second.wait();

    EXPECT_TRUE(second_ran);
}

TEST(thread_pool_test,
     destroying_the_pool_cancels_queued_tasks_and_releases_their_waiters)
{
    auto              pool = std::make_unique<thread_pool>(1);
    blocker           first;
    std::atomic<bool> second_ran{ false };

    const auto first_handle = pool->submit(first.task());
    first.wait_until_started();
    const auto second_handle =
        pool->submit([&](const std::stop_token&) { second_ran = true; });

    std::thread destroyer{ [&] { pool.reset(); } };
    std::this_thread::sleep_for(50ms);
    first.release();
    destroyer.join();

    EXPECT_TRUE(finishes_within(2s, [&] { second_handle.wait(); }));
    EXPECT_TRUE(finishes_within(2s, [&] { first_handle.wait(); }));
    EXPECT_FALSE(second_ran);
    EXPECT_FALSE(second_handle.cancel());
}

TEST(thread_pool_test, destroying_an_idle_pool_does_not_hang)
{
    EXPECT_TRUE(finishes_within(2s, [] { thread_pool pool{ 4 }; }));
}

}  // namespace
