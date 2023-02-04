// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include <atomic>
#include <mutex>

namespace omnitrace
{
namespace locking
{
/// simple mutex which spins on an atomic while trying to lock.
/// Provided for internal use for when there is low contention
/// but we want to avoid using pthread mutexes since those
/// are wrapped by library
struct atomic_mutex
{
    atomic_mutex()  = default;
    ~atomic_mutex() = default;

    atomic_mutex(const atomic_mutex&)     = delete;
    atomic_mutex(atomic_mutex&&) noexcept = delete;

    atomic_mutex& operator=(const atomic_mutex&) = delete;
    atomic_mutex& operator=(atomic_mutex&&) noexcept = delete;

    void lock();
    void unlock();
    bool try_lock();

private:
    std::atomic<int64_t> m_value = {};
};

struct atomic_lock
{
    atomic_lock(atomic_mutex&);
    atomic_lock(atomic_mutex&, std::defer_lock_t);
    ~atomic_lock();

    atomic_lock(const atomic_lock&)     = delete;
    atomic_lock(atomic_lock&&) noexcept = delete;

    atomic_lock& operator=(const atomic_lock&) = delete;
    atomic_lock& operator=(atomic_lock&&) noexcept = delete;

    bool owns_lock() const;

    void lock();
    void unlock();
    bool try_lock();

private:
    bool          m_owns = false;
    atomic_mutex& m_mutex;
};
}  // namespace locking
}  // namespace omnitrace
