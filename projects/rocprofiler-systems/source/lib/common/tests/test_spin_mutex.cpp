// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/spin_mutex.hpp"

#include <gtest/gtest.h>

#include <cstddef>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace
{
using rocprofsys::common::spin_mutex;

constexpr std::size_t k_thread_count = 8;
constexpr std::size_t k_iterations   = 10000;

TEST(spin_mutex, try_lock_fails_while_held)
{
    auto mutex = spin_mutex{};

    ASSERT_TRUE(mutex.try_lock());
    EXPECT_FALSE(mutex.try_lock());

    mutex.unlock();
    EXPECT_TRUE(mutex.try_lock());
    mutex.unlock();
}

TEST(spin_mutex, lock_guard_releases_on_scope_exit)
{
    auto mutex = spin_mutex{};

    {
        const std::lock_guard<spin_mutex> guard{ mutex };
        EXPECT_FALSE(mutex.try_lock());
    }

    EXPECT_TRUE(mutex.try_lock());
    mutex.unlock();
}

void
increment_locked(spin_mutex& mutex, std::size_t& counter)
{
    for(std::size_t i = 0; i < k_iterations; ++i)
    {
        const std::lock_guard<spin_mutex> guard{ mutex };
        ++counter;
    }
}

TEST(spin_mutex, provides_mutual_exclusion)
{
    auto        mutex   = spin_mutex{};
    std::size_t counter = 0;

    auto workers = std::vector<std::thread>{};
    for(std::size_t i = 0; i < k_thread_count; ++i)
    {
        workers.emplace_back(increment_locked, std::ref(mutex), std::ref(counter));
    }
    for(auto& worker : workers)
    {
        worker.join();
    }

    EXPECT_EQ(counter, k_thread_count * k_iterations);
}
}  // namespace
