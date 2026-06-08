/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "context.h"
#include "hipfile-warnings.h"
#include "mthread-pool.h"
#include "thread-pool.h"

#include <atomic>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>

using namespace hipFile;
using ::testing::ByMove;
using ::testing::Return;
using ::testing::StrictMock;

// Put tests inside the macros to suppress the global constructor
// warnings
HIPFILE_WARN_NO_GLOBAL_CTOR_OFF

TEST(HipFileThreadPool, ConstructorUsesDefaultArenaConcurrency)
{
    ThreadPool pool{};
    ASSERT_GT(pool.threadCount(), 0);
}

TEST(HipFileThreadPool, ContextDefaultUsesThreadPool)
{
    IThreadPool *thread_pool = Context<IThreadPool>::get();

    ASSERT_NE(dynamic_cast<ThreadPool *>(thread_pool), nullptr);
    ASSERT_GT(thread_pool->threadCount(), 0);
}

TEST(HipFileThreadPool, ContextCanUseMockThreadPool)
{
    StrictMock<MThreadPool> thread_pool;
    auto                    task_group = std::make_unique<StrictMock<MTaskGroup>>();

    EXPECT_CALL(thread_pool, makeTaskGroup).WillOnce(Return(ByMove(std::move(task_group))));

    ASSERT_NE(Context<IThreadPool>::get()->makeTaskGroup(), nullptr);
}

TEST(HipFileThreadPool, TaskGroupWorkRuns)
{
    ThreadPool       pool{};
    auto             task_group = pool.makeTaskGroup();
    std::atomic<int> completed{0};

    task_group->run([&completed]() { completed.fetch_add(1, std::memory_order_relaxed); });
    task_group->run([&completed]() { completed.fetch_add(1, std::memory_order_relaxed); });

    task_group->wait();

    ASSERT_EQ(completed.load(std::memory_order_relaxed), 2);
}

TEST(HipFileThreadPool, TaskGroupCanOutliveThreadPoolObject)
{
    std::unique_ptr<ITaskGroup> task_group;
    std::atomic<int>            completed{0};

    {
        ThreadPool pool{};
        task_group = pool.makeTaskGroup();
        task_group->run([&completed]() { completed.fetch_add(1, std::memory_order_relaxed); });
    }

    task_group->wait();

    ASSERT_EQ(completed.load(std::memory_order_relaxed), 1);
}

HIPFILE_WARN_NO_GLOBAL_CTOR_ON
