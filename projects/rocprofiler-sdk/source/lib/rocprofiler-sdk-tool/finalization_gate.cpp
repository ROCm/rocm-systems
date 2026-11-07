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
#include <unistd.h>

#include <limits>

namespace rocprofiler
{
namespace tool
{
namespace
{
void
wait_while_equal(std::atomic<uint32_t>& word, uint32_t expected)
{
    syscall(SYS_futex, &word, FUTEX_WAIT, expected, nullptr, nullptr, 0);
}

void
wake_all(std::atomic<uint32_t>& word)
{
    syscall(SYS_futex, &word, FUTEX_WAKE, std::numeric_limits<int>::max(), nullptr, nullptr, 0);
}
}  // namespace

finalization_gate::status
finalization_gate::run_or_wait(const std::function<void()>& func)
{
    auto expected = static_cast<uint32_t>(state::not_started);
    if(!m_state.compare_exchange_strong(
           expected, static_cast<uint32_t>(state::running), std::memory_order_acq_rel))
    {
        if(expected == static_cast<uint32_t>(state::complete)) return status::completed;
        if(m_finalizer.load(std::memory_order_acquire) == std::this_thread::get_id())
            return status::reentered;
        return wait_until_complete_or_reentered();
    }

    m_finalizer.store(std::this_thread::get_id(), std::memory_order_release);
    // A wait-only caller may already be sleeping on not_started.
    wake_all(m_state);

    // Release the waiters even if `func` throws.
    auto _release = common::scope_destructor{[this]() {
        m_state.store(static_cast<uint32_t>(state::complete), std::memory_order_release);
        wake_all(m_state);
    }};
    if(func) func();
    return status::ran;
}

finalization_gate::status
finalization_gate::run_or_wait(const std::function<void()>& func,
                               const std::function<void()>& after_complete)
{
    status result;
    try
    {
        result = run_or_wait(func);
    } catch(...)
    {
        if(after_complete) after_complete();
        throw;
    }
    if(result != status::reentered && after_complete) after_complete();
    return result;
}

finalization_gate::status
finalization_gate::wait_until_complete_or_reentered()
{
    while(true)
    {
        auto current = m_state.load(std::memory_order_acquire);
        if(current == static_cast<uint32_t>(state::complete)) return status::completed;
        if(current == static_cast<uint32_t>(state::running) &&
           m_finalizer.load(std::memory_order_acquire) == std::this_thread::get_id())
            return status::reentered;
        wait_while_equal(m_state, current);
    }
}
finalization_gate::status
finalize_once_and_join(finalization_gate&           request_gate,
                       finalization_gate&           completion_gate,
                       const std::function<void()>& request,
                       const std::function<void()>& join)
{
    using status = finalization_gate::status;
    auto result  = request_gate.run_or_wait(request);
    if(result == status::reentered) return result;
    if(completion_gate.wait_until_complete_or_reentered() == status::reentered)
        return status::reentered;
    if(join) join();
    return result;
}
}  // namespace tool
}  // namespace rocprofiler
