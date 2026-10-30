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

#pragma once

#include <signal.h>
#include <sys/types.h>
#include <array>
#include <atomic>
#include <cstddef>
#include <optional>
#include <utility>

namespace rocprofiler
{
namespace common
{
// PID ownership rejects an inherited eventfd even before any child atfork callback has run.
class fork_signal_state
{
public:
    bool usable(pid_t pid) const noexcept { return m_owner.load(std::memory_order_acquire) == pid; }
    void ready(pid_t pid) noexcept { m_owner.store(pid, std::memory_order_release); }

private:
    std::atomic<pid_t> m_owner{0};
    static_assert(std::atomic<pid_t>::is_always_lock_free);
};

// Signal handlers only read published immutable records. Retain at most a startup snapshot
// and the first application snapshot per signal; this preserves the tool's first-handler policy.
struct signal_disposition
{
    void (*handler)(int) = nullptr;
    std::optional<struct sigaction> action{};
    bool                            bootstrap = false;
    const signal_disposition*       previous  = nullptr;
};

template <size_t Size>
class signal_dispositions
{
public:
    const signal_disposition* load(int signo) const noexcept
    {
        return m_slots.at(signo).load(std::memory_order_acquire);
    }
    void save(int signo, signal_disposition value)
    {
        auto& slot     = m_slots.at(signo);
        auto* previous = slot.load(std::memory_order_acquire);
        if(previous && (!previous->bootstrap || value.bootstrap)) return;
        auto* next = new signal_disposition{std::move(value)};
        while(true)
        {
            next->previous = previous;
            if(slot.compare_exchange_weak(
                   previous, next, std::memory_order_release, std::memory_order_acquire))
                break;
            if(previous && (!previous->bootstrap || next->bootstrap))
            {
                delete next;
                return;
            }
        }
        // Published records deliberately have process lifetime, including across fork.
    }
    bool chained(int signo) const noexcept
    {
        const auto* state = load(signo);
        return state &&
               (state->action ? state->action->sa_handler != SIG_DFL : state->handler != SIG_DFL);
    }

private:
    std::array<std::atomic<const signal_disposition*>, Size> m_slots{};
    static_assert(std::atomic<const signal_disposition*>::is_always_lock_free);
};

}  // namespace common
}  // namespace rocprofiler
