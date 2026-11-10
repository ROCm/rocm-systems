// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/thread_list/internal_thread_list.hpp"

#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <gtest/gtest.h>
#include <stdexcept>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace
{
using rocprofsys::thread_list::internal_thread_list;
using rocprofsys::thread_list::scoped_internal_thread;

constexpr pid_t k_tid_a = 1001;
constexpr pid_t k_tid_b = 1002;

constexpr auto k_stress_duration = std::chrono::milliseconds{ 200 };
constexpr int  k_stress_adders   = 4;

pid_t
this_tid()
{
    return static_cast<pid_t>(::syscall(SYS_gettid));
}

// Registers, queries, passes the creation gate and unregisters @p tid until @p stop.
void
churn_entries(internal_thread_list& list, const std::atomic<bool>& stop, pid_t tid)
{
    while(!stop.load())
    {
        list.add(tid);
        static_cast<void>(list.is_excluded(tid));
        {
            const auto scope = list.begin_creation();
        }
        list.remove(tid);
    }
}

// Prunes and probes the rescan gate until @p stop.
void
churn_rescan(internal_thread_list& list, const std::atomic<bool>& stop)
{
    while(!stop.load())
    {
        list.prune_exited();
        const auto scope = list.try_begin_rescan();
    }
}
}  // namespace

TEST(internal_thread_list, add_remove_is_excluded)
{
    internal_thread_list list;
    EXPECT_FALSE(list.is_excluded(k_tid_a));
    list.add(k_tid_a);
    list.add(k_tid_a);
    EXPECT_TRUE(list.is_excluded(k_tid_a));
    EXPECT_FALSE(list.is_excluded(k_tid_b));
    list.remove(k_tid_a);
    EXPECT_FALSE(list.is_excluded(k_tid_a));
    list.remove(k_tid_a);
    list.add(k_tid_a);
    EXPECT_TRUE(list.is_excluded(k_tid_a));
}

TEST(internal_thread_list, scoped_registers_and_removes)
{
    internal_thread_list list;
    {
        scoped_internal_thread scoped{ list };
        EXPECT_TRUE(list.is_excluded(this_tid()));
    }
    EXPECT_FALSE(list.is_excluded(this_tid()));
}

TEST(internal_thread_list, scoped_removes_on_exception)
{
    internal_thread_list list;
    EXPECT_THROW(
        {
            scoped_internal_thread scoped{ list };
            throw std::runtime_error{ "unwind" };
        },
        std::runtime_error);
    EXPECT_FALSE(list.is_excluded(this_tid()));
}

TEST(internal_thread_list, scoped_move_removes_once)
{
    internal_thread_list list;
    {
        scoped_internal_thread first{ list };
        scoped_internal_thread second{ std::move(first) };
        EXPECT_TRUE(list.is_excluded(this_tid()));
    }
    EXPECT_FALSE(list.is_excluded(this_tid()));
}

TEST(internal_thread_list, prune_exited_drops_dead_keeps_live)
{
    internal_thread_list list;
    pid_t                dead_tid = 0;
    std::thread{ [&dead_tid] { dead_tid = this_tid(); } }.join();
    ASSERT_NE(dead_tid, 0);

    list.add(dead_tid);
    list.add(this_tid());
    list.prune_exited();

    EXPECT_FALSE(list.is_excluded(dead_tid));
    EXPECT_TRUE(list.is_excluded(this_tid()));
}

TEST(internal_thread_list, rescan_blocks_while_creation_alive)
{
    internal_thread_list list;
    auto                 creation = list.begin_creation();
    EXPECT_FALSE(list.try_begin_rescan().owns_lock());

    std::atomic<bool> rescan_started{ false };
    auto              blocked = std::async(std::launch::async, [&] {
        auto scope = list.begin_rescan();
        rescan_started.store(true);
    });
    EXPECT_EQ(blocked.wait_for(std::chrono::milliseconds{ 100 }),
              std::future_status::timeout);
    EXPECT_FALSE(rescan_started.load());

    creation.unlock();
    blocked.get();
    EXPECT_TRUE(rescan_started.load());
    EXPECT_TRUE(list.try_begin_rescan().owns_lock());
}

TEST(internal_thread_list, creations_coexist_and_add_does_not_deadlock)
{
    internal_thread_list list;
    auto                 first  = list.begin_creation();
    auto                 second = list.begin_creation();
    EXPECT_TRUE(first.owns_lock());
    EXPECT_TRUE(second.owns_lock());

    // A child registers itself while its creator holds the gate.
    std::thread{ [&list] { list.add(k_tid_a); } }.join();
    EXPECT_TRUE(list.is_excluded(k_tid_a));
}

TEST(internal_thread_list, concurrent_operations)
{
    internal_thread_list     list;
    std::atomic<bool>        stop{ false };
    std::vector<std::thread> workers;
    workers.reserve(k_stress_adders + 1);
    for(int i = 0; i < k_stress_adders; ++i)
    {
        workers.emplace_back(churn_entries, std::ref(list), std::cref(stop), k_tid_a + i);
    }
    workers.emplace_back(churn_rescan, std::ref(list), std::cref(stop));

    std::this_thread::sleep_for(k_stress_duration);
    stop.store(true);
    for(auto& worker : workers)
    {
        worker.join();
    }
    SUCCEED();
}

TEST(internal_thread_list, reset_after_fork_leaves_usable_empty_list)
{
    internal_thread_list list;
    list.add(k_tid_a);
    list.reset_after_fork();
    EXPECT_FALSE(list.is_excluded(k_tid_a));
    list.add(k_tid_b);
    EXPECT_TRUE(list.is_excluded(k_tid_b));
    EXPECT_TRUE(list.try_begin_rescan().owns_lock());
    EXPECT_TRUE(list.begin_creation().owns_lock());
}
