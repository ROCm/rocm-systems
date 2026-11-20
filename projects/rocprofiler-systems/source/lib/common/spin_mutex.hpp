// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <thread>

namespace rocprofsys::inline common
{
/**
 * @brief Spin lock satisfying BasicLockable that never calls pthread_mutex_lock.
 *
 * Usable from code reachable through the pthread_mutex gotcha wrappers, where a
 * std::mutex would re-enter the wrapper. Waiters yield while spinning.
 */
class spin_mutex
{
public:
    spin_mutex() = default;

    spin_mutex(const spin_mutex&)            = delete;
    spin_mutex& operator=(const spin_mutex&) = delete;
    spin_mutex(spin_mutex&&)                 = delete;
    spin_mutex& operator=(spin_mutex&&)      = delete;

    void lock() noexcept
    {
        while(m_flag.test_and_set(std::memory_order_acquire))
        {
            while(m_flag.test(std::memory_order_relaxed))
            {
                std::this_thread::yield();
            }
        }
    }

    [[nodiscard]] bool try_lock() noexcept
    {
        return !m_flag.test_and_set(std::memory_order_acquire);
    }

    void unlock() noexcept { m_flag.clear(std::memory_order_release); }

private:
    std::atomic_flag m_flag;
};
}  // namespace rocprofsys::inline common
