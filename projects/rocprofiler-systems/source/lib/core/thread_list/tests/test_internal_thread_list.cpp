// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/thread_list/internal_thread_list.hpp"

#include <atomic>
#include <chrono>
#include <future>
#include <gtest/gtest.h>
#include <stdexcept>
#include <sys/syscall.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace
{
using rocprofsys::thread_list::internal_thread_list;
using rocprofsys::thread_list::scoped_internal_thread;

constexpr pid_t k_tid_a = 1001;
constexpr pid_t k_tid_b = 1002;

pid_t
this_tid()
{
    return static_cast<pid_t>(::syscall(SYS_gettid));
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
    try
    {
        scoped_internal_thread scoped{ list };
        throw std::runtime_error{ "unwind" };
    } catch(const std::runtime_error&)
    {}
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
    for(int i = 0; i < 4; ++i)
    {
        workers.emplace_back([&list, &stop, i] {
            const pid_t tid = k_tid_a + i;
            while(!stop.load())
            {
                list.add(tid);
                (void) list.is_excluded(tid);
                {
                    auto scope = list.begin_creation();
                }
                list.remove(tid);
            }
        });
    }
    workers.emplace_back([&list, &stop] {
        while(!stop.load())
        {
            list.prune_exited();
            auto scope = list.try_begin_rescan();
        }
    });
    std::this_thread::sleep_for(std::chrono::milliseconds{ 200 });
    stop.store(true);
    for(auto& worker : workers)
        worker.join();
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
