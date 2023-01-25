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

#include "locking.hpp"

namespace omnitrace
{
namespace locking
{
void
atomic_mutex::lock()
{
    while(!try_lock())
    {}
}

void
atomic_mutex::unlock()
{
    if((m_value.load() & 1) == 1) ++m_value;
}

bool
atomic_mutex::try_lock()
{
    auto _targ = m_value.load(std::memory_order_relaxed);
    if((_targ & 1) == 0)
    {
        return (
            m_value.compare_exchange_strong(_targ, _targ + 1, std::memory_order_relaxed));
    }
    return false;
}

atomic_lock::atomic_lock(atomic_mutex& _v)
: m_mutex{ _v }
{
    lock();
}

atomic_lock::atomic_lock(atomic_mutex& _v, std::defer_lock_t)
: m_mutex{ _v }
{}

atomic_lock::~atomic_lock() { unlock(); }

bool
atomic_lock::owns_lock() const
{
    return m_owns;
}

void
atomic_lock::lock()
{
    if(!owns_lock())
    {
        m_mutex.lock();
        m_owns = true;
    }
}

void
atomic_lock::unlock()
{
    if(owns_lock())
    {
        m_mutex.unlock();
        m_owns = false;
    }
}

bool
atomic_lock::try_lock()
{
    if(!owns_lock())
    {
        m_owns = m_mutex.try_lock();
    }
    return m_owns;
}
}  // namespace locking
}  // namespace omnitrace
