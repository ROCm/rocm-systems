/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hipfile-warnings.h"
#include "thread-pool.h"

#include <atomic>
#include <chrono>
#include <future>
#include <memory>

#include <gtest/gtest.h>

using namespace hipFile;
using namespace std::chrono_literals;

HIPFILE_WARN_NO_GLOBAL_CTOR_OFF

TEST(HipFileThreadPool, RunExecutesWork)
{
    ThreadPool        pool{1};
    auto              group = pool.makeTaskGroup();
    std::atomic<bool> ran{false};

    group->run([&ran]() { ran = true; });
    group->wait();

    ASSERT_TRUE(ran);
}

TEST(HipFileThreadPool, WaitBlocksUntilRunningWorkCompletes)
{
    ThreadPool        pool{1};
    auto              group = pool.makeTaskGroup();
    std::atomic<bool> gate{false};
    std::atomic<bool> started{false};

    group->run([&gate, &started]() {
        started.store(true);
        started.notify_all();
        gate.wait(false);
    });
    started.wait(false);

    auto wait_result = std::async(std::launch::async, [&group]() { group->wait(); });
    ASSERT_EQ(wait_result.wait_for(20ms), std::future_status::timeout);

    gate.store(true);
    gate.notify_all();
    ASSERT_EQ(wait_result.wait_for(1s), std::future_status::ready);
}

TEST(HipFileThreadPool, CancelSkipsPendingWork)
{
    ThreadPool        pool{1};
    auto              group = pool.makeTaskGroup();
    std::atomic<bool> gate{false};
    std::atomic<bool> started{false};
    std::atomic<bool> pending_ran{false};

    group->run([&gate, &started]() {
        started.store(true);
        started.notify_all();
        gate.wait(false);
    });
    started.wait(false);

    group->run([&pending_ran]() { pending_ran = true; });
    group->cancel();
    gate.store(true);
    gate.notify_all();
    group->wait();

    ASSERT_FALSE(pending_ran);
}

TEST(HipFileThreadPool, CancelDoesNotStopRunningWork)
{
    ThreadPool        pool{1};
    auto              group = pool.makeTaskGroup();
    std::atomic<bool> gate{false};
    std::atomic<bool> started{false};
    std::atomic<bool> completed{false};

    group->run([&gate, &started, &completed]() {
        started.store(true);
        started.notify_all();
        gate.wait(false);
        completed = true;
    });
    started.wait(false);

    group->cancel();
    gate.store(true);
    gate.notify_all();
    group->wait();

    ASSERT_TRUE(completed);
}

TEST(HipFileThreadPool, CanSubmitAfterCancel)
{
    ThreadPool        pool{1};
    auto              group = pool.makeTaskGroup();
    std::atomic<bool> ran{false};

    group->cancel();
    group->run([&ran]() { ran = true; });
    group->wait();

    ASSERT_TRUE(ran);
}

TEST(HipFileThreadPool, DestructorCancelsAndWaits)
{
    ThreadPool        pool{1};
    auto              group = pool.makeTaskGroup();
    std::atomic<bool> gate{false};
    std::atomic<bool> started{false};
    std::atomic<bool> pending_ran{false};

    group->run([&gate, &started]() {
        started.store(true);
        started.notify_all();
        gate.wait(false);
    });
    started.wait(false);

    group->run([&pending_ran]() { pending_ran = true; });

    auto destroy_result =
        std::async(std::launch::async, [owned_group = std::move(group)]() mutable { owned_group.reset(); });
    ASSERT_EQ(destroy_result.wait_for(20ms), std::future_status::timeout);

    gate.store(true);
    gate.notify_all();
    ASSERT_EQ(destroy_result.wait_for(1s), std::future_status::ready);
    ASSERT_FALSE(pending_ran);
}

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
