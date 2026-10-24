// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
//

#include "finalization_gate.hpp"

#include "lib/common/scope_destructor.hpp"

#include <linux/futex.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <cerrno>
#include <limits>

namespace rocprofiler
{
namespace tool
{
namespace
{
// Returns false if `timeout_sec` (> 0) elapses before `word` is set; <= 0 waits indefinitely.
bool
wait_until_set(std::atomic<uint32_t>& word, int timeout_sec)
{
    // FUTEX_WAIT_BITSET takes an absolute CLOCK_MONOTONIC deadline, so compute it once and let
    // the kernel track the time remaining across wakeups.
    const auto bounded  = (timeout_sec > 0);
    auto       deadline = timespec{};
    if(bounded)
    {
        clock_gettime(CLOCK_MONOTONIC, &deadline);
        deadline.tv_sec += timeout_sec;
    }

    while(word.load(std::memory_order_acquire) == 0)
    {
        if(syscall(SYS_futex,
                   &word,
                   FUTEX_WAIT_BITSET,
                   0,
                   bounded ? &deadline : nullptr,
                   nullptr,
                   FUTEX_BITSET_MATCH_ANY) == -1 &&
           errno == ETIMEDOUT)
        {
            return false;
        }
    }
    return true;
}

// Wake every waiter: exit paths on several threads can be waiting at once.
void
set_and_wake_all(std::atomic<uint32_t>& word)
{
    word.store(1u, std::memory_order_release);
    syscall(SYS_futex, &word, FUTEX_WAKE, std::numeric_limits<int>::max(), nullptr, nullptr, 0);
}
}  // namespace

finalization_gate::status
finalization_gate::run_or_wait(const std::function<void()>& func, int timeout_sec)
{
    if(m_started.test_and_set(std::memory_order_acq_rel))
    {
        if(m_complete.load(std::memory_order_acquire) != 0) return status::completed;
        if(m_finalizer.load(std::memory_order_acquire) == std::this_thread::get_id())
            return status::reentered;
        return wait_until_set(m_complete, timeout_sec) ? status::completed : status::timed_out;
    }

    m_finalizer.store(std::this_thread::get_id(), std::memory_order_release);

    // Release the waiters even if `func` throws.
    auto _release = common::scope_destructor{[this]() { set_and_wake_all(m_complete); }};
    if(func) func();
    return status::ran;
}
}  // namespace tool
}  // namespace rocprofiler
