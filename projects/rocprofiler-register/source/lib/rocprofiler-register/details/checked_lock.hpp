// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

#include <mutex>
#include <thread>
#include <atomic>

namespace rocprofiler_register
{
namespace common
{
// A non-recursive mutex wrapper that:
//   - serializes concurrent callers (one proceeds, others block)
//   - detects recursive acquisition by the same thread and reports it via
//     the `recursive` flag rather than deadlocking or silently succeeding
//
// Usage (scoped_lock style):
//
//   static checked_mutex mtx;
//   checked_lock lk{mtx};
//   if(lk.recursive) return ROCP_REG_DEADLOCK;
//   // ... protected work ...

struct checked_mutex
{
    std::mutex              mtx    = {};
    std::atomic<std::thread::id> owner  = {};
};

struct checked_lock
{
    explicit checked_lock(checked_mutex& cm)
    : m_cm{ cm }
    {
        const auto self = std::this_thread::get_id();
        // Check for recursive entry before blocking on the mutex.
        // If this thread already owns the mutex the owner field equals self.
        recursive = (cm.owner.load(std::memory_order_acquire) == self);
        if(!recursive)
        {
            cm.mtx.lock();
            cm.owner.store(self, std::memory_order_release);
        }
    }

    ~checked_lock()
    {
        if(!recursive)
        {
            m_cm.owner.store(std::thread::id{}, std::memory_order_release);
            m_cm.mtx.unlock();
        }
    }

    checked_lock(const checked_lock&)            = delete;
    checked_lock& operator=(const checked_lock&) = delete;
    checked_lock(checked_lock&&)                 = delete;
    checked_lock& operator=(checked_lock&&)      = delete;

    // true  → the calling thread already held the lock (recursion detected)
    // false → the lock was freshly acquired
    bool recursive = false;

private:
    checked_mutex& m_cm;
};

}  // namespace common
}  // namespace rocprofiler_register
