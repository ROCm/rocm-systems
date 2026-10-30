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

#include <sys/types.h>
#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <utility>

namespace rocprofiler
{
namespace common
{
// Activity tracking is observational: it never blocks an application prepare handler.
// A lifecycle operation invalidates an admitted fork even if it finishes before the syscall.
class fork_lifecycle
{
public:
    class activity
    {
    public:
        explicit activity(fork_lifecycle& owner)
        : m_owner{&owner}
        {
            owner.m_state.fetch_add(epoch_step + 1, std::memory_order_acq_rel);
        }
        activity(activity&& rhs) noexcept
        : m_owner{std::exchange(rhs.m_owner, nullptr)}
        {}
        activity(const activity&) = delete;
        activity& operator=(const activity&) = delete;
        activity& operator=(activity&&) = delete;
        ~activity()
        {
            if(m_owner && !m_owner->abandoned())
                m_owner->m_state.fetch_add(epoch_step - 1, std::memory_order_release);
        }

    private:
        fork_lifecycle* m_owner = nullptr;
    };

    bool abandoned() const noexcept
    {
        return m_abandoned.load(std::memory_order_acquire) ||
               m_owner_pid.load(std::memory_order_acquire) != ::getpid();
    }
    bool try_prepare() noexcept
    {
        if(abandoned()) return false;
        auto value = m_state.load(std::memory_order_acquire);
        if((value & (activity_mask | fork_preparing)) != 0) return false;
        if(!m_state.compare_exchange_strong(
               value, value | fork_preparing, std::memory_order_acq_rel))
            return false;
        m_snapshot.store(value | fork_preparing, std::memory_order_release);
        return true;
    }
    bool child_eligible() const noexcept
    {
        const auto value = m_state.load(std::memory_order_acquire);
        return !m_abandoned.load(std::memory_order_acquire) && (value & fork_preparing) != 0 &&
               value == m_snapshot.load(std::memory_order_acquire);
    }
    void parent_complete() noexcept
    {
        // Preserve activities admitted by later prepare handlers or other parent threads.
        m_state.fetch_and(~fork_preparing, std::memory_order_release);
    }
    void child_complete(bool force_abandon = false) noexcept
    {
        if(force_abandon || !child_eligible())
            m_abandoned.store(true, std::memory_order_release);
        else
            m_owner_pid.store(::getpid(), std::memory_order_release);
        m_state.store(0, std::memory_order_release);
    }

private:
    static constexpr uint64_t epoch_step     = uint64_t{1} << 32;
    static constexpr uint64_t fork_preparing = uint64_t{1} << 31;
    static constexpr uint64_t activity_mask  = fork_preparing - 1;
    std::atomic<uint64_t>     m_state{0};
    std::atomic<uint64_t>     m_snapshot{0};
    std::atomic<bool>         m_abandoned{false};
    std::atomic<pid_t>        m_owner_pid{::getpid()};
    static_assert(std::atomic<uint64_t>::is_always_lock_free);
    static_assert(std::atomic<bool>::is_always_lock_free);
    static_assert(std::atomic<pid_t>::is_always_lock_free);
};

// A nested prepare must not flush SDK buffers or acquire the registry a second time. The parent
// of that nested fork leaves the outer prepare intact; its child abandons the inherited session.
struct fork_sequence
{
    bool begin() noexcept { return ++depth == 1; }
    bool finish() noexcept { return --depth == 0; }
    bool nested() const noexcept { return depth > 1; }

    uint32_t depth    = 0;
    bool     admitted = false;
};
}  // namespace common
}  // namespace rocprofiler
