// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/self_suppressing_mutex.hpp"
#include "common/trace_suppression.hpp"

#include <condition_variable>
#include <gtest/gtest.h>
#include <mutex>
#include <thread>

using rocprofsys::self_suppressing_mutex;
using rocprofsys::trace_suppression;

namespace
{
class SelfSuppressingMutexTest : public ::testing::Test
{
protected:
    self_suppressing_mutex m_mutex;
};
}  // namespace

TEST_F(SelfSuppressingMutexTest, SuppressesWhileHeldViaLockGuard)
{
    EXPECT_FALSE(trace_suppression::is_active());

    {
        std::lock_guard<self_suppressing_mutex> lock{ m_mutex };
        EXPECT_TRUE(trace_suppression::is_active());
    }

    EXPECT_FALSE(trace_suppression::is_active());
}

TEST_F(SelfSuppressingMutexTest, TryLockSuccessSuppressesUntilUnlock)
{
    ASSERT_TRUE(m_mutex.try_lock());
    EXPECT_TRUE(trace_suppression::is_active());

    m_mutex.unlock();
    EXPECT_FALSE(trace_suppression::is_active());
}

TEST_F(SelfSuppressingMutexTest, TryLockFailureLeavesCallingThreadUnsuppressed)
{
    std::mutex              cv_mutex;
    std::condition_variable holder_locked_cv;
    std::condition_variable release_cv;
    bool                    holder_locked = false;
    bool                    may_release   = false;

    std::thread holder([&]() {
        m_mutex.lock();
        {
            std::lock_guard<std::mutex> lock{ cv_mutex };
            holder_locked = true;
        }
        holder_locked_cv.notify_one();

        std::unique_lock<std::mutex> lock{ cv_mutex };
        release_cv.wait(lock, [&]() { return may_release; });
        m_mutex.unlock();
    });

    {
        std::unique_lock<std::mutex> lock{ cv_mutex };
        holder_locked_cv.wait(lock, [&]() { return holder_locked; });
    }

    EXPECT_FALSE(m_mutex.try_lock());
    EXPECT_FALSE(trace_suppression::is_active());

    {
        std::lock_guard<std::mutex> lock{ cv_mutex };
        may_release = true;
    }
    release_cv.notify_one();
    holder.join();
}
