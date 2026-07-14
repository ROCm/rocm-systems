// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/trace_suppression.hpp"

#include <mutex>

namespace rocprofsys
{
inline namespace common
{
/// std::mutex-compatible wrapper (satisfies BasicLockable) whose lock/unlock
/// calls are invisible to pthread mutex/rwlock tracing (see
/// trace_suppression). Without this, tracing a lock acquisition here would
/// recursively try to record that trace event through the very lock being
/// acquired, deadlocking on self-relock of a non-recursive mutex.
class self_suppressing_mutex
{
public:
    self_suppressing_mutex()  = default;
    ~self_suppressing_mutex() = default;

    self_suppressing_mutex(const self_suppressing_mutex&)            = delete;
    self_suppressing_mutex& operator=(const self_suppressing_mutex&) = delete;
    self_suppressing_mutex(self_suppressing_mutex&&)                 = delete;
    self_suppressing_mutex& operator=(self_suppressing_mutex&&)      = delete;

    void lock()
    {
        trace_suppression::enter();
        m_mutex.lock();
    }

    void unlock()
    {
        m_mutex.unlock();
        trace_suppression::exit();
    }

    [[nodiscard]] bool try_lock()
    {
        trace_suppression::enter();
        if(m_mutex.try_lock()) return true;
        trace_suppression::exit();
        return false;
    }

private:
    std::mutex m_mutex;
};
}  // namespace common
}  // namespace rocprofsys
