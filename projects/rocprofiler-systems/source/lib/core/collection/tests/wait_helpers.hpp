// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "core/collection/fork_gate.hpp"
#include "core/collection/scheduler.hpp"

#include <pthread.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <future>
#include <latch>
#include <thread>

namespace rocprofsys::collection::testing
{
/** Generous bound for waits that are expected to succeed. */
inline constexpr auto k_timeout = std::chrono::seconds{ 30 };

/** Counts calls; wait() returns once `wanted` calls were counted. Thread-safe. */
class call_counter
{
public:
    explicit call_counter(int wanted)
    : m_wanted{ wanted }
    {}

    void operator()()
    {
        if(++m_count == m_wanted)
        {
            m_reached.set_value();
        }
    }

    /** @return false on timeout */
    [[nodiscard]] bool wait() const
    {
        return m_reached_future.wait_for(k_timeout) == std::future_status::ready;
    }

    [[nodiscard]] int count() const { return m_count.load(); }

private:
    int                m_wanted;
    std::atomic<int>   m_count{ 0 };
    std::promise<void> m_reached;
    std::future<void>  m_reached_future = m_reached.get_future();
};

/** Lets an unrelated scheduler tick a few times, so a thread that should stay idle has
 * time to misbehave. Only for checks that something does not happen. */
inline void
let_other_threads_run()
{
    constexpr int  k_heartbeat_ticks = 20;
    constexpr auto k_heartbeat       = std::chrono::milliseconds{ 1 };

    call_counter heartbeat{ k_heartbeat_ticks };
    scheduler    clock{ "heartbeat" };
    clock.start(k_heartbeat, [&heartbeat](std::uint64_t) {
        heartbeat();
        return true;
    });
    static_cast<void>(heartbeat.wait());
}

/** Whether a poll on another thread would get a pass right now. */
[[nodiscard]] inline bool
pass_granted_elsewhere(fork_gate& gate)
{
    bool granted = false;
    std::jthread{ [&gate, &granted] {
        granted = static_cast<bool>(fork_gate::pass{ gate });
    } }.join();
    return granted;
}

/** Spins until a fork is counted by the gate (another thread is inside block()).
 * @return false when none was counted within k_timeout */
[[nodiscard]] inline bool
wait_for_fork_in_progress(fork_gate& gate)
{
    const auto give_up = std::chrono::steady_clock::now() + k_timeout;
    while(pass_granted_elsewhere(gate))
    {
        if(std::chrono::steady_clock::now() >= give_up)
        {
            return false;
        }
        std::this_thread::yield();
    }
    return true;
}

/** While it lives, creating a thread fails (std::system_error): the default thread stack
 * is set far beyond what can be mapped. Create and destroy it while no other thread
 * starts threads. */
class failing_thread_creation
{
public:
    failing_thread_creation()
    {
        constexpr std::size_t k_unmappable_stack = std::size_t{ 1 } << 50U;
        pthread_getattr_default_np(&m_saved);
        pthread_attr_t huge{};
        pthread_attr_init(&huge);
        pthread_attr_setstacksize(&huge, k_unmappable_stack);
        pthread_setattr_default_np(&huge);
        pthread_attr_destroy(&huge);
    }
    ~failing_thread_creation()
    {
        pthread_setattr_default_np(&m_saved);
        pthread_attr_destroy(&m_saved);
    }

    failing_thread_creation(const failing_thread_creation&)            = delete;
    failing_thread_creation& operator=(const failing_thread_creation&) = delete;
    failing_thread_creation(failing_thread_creation&&)                 = delete;
    failing_thread_creation& operator=(failing_thread_creation&&)      = delete;

private:
    pthread_attr_t m_saved{};
};

/** Counts a latch down when destroyed, so threads waiting on it can be joined. */
class latch_releaser
{
public:
    explicit latch_releaser(std::latch& gate)
    : m_gate{ gate }
    {}
    ~latch_releaser() { m_gate.count_down(); }

    latch_releaser(const latch_releaser&)            = delete;
    latch_releaser& operator=(const latch_releaser&) = delete;
    latch_releaser(latch_releaser&&)                 = delete;
    latch_releaser& operator=(latch_releaser&&)      = delete;

private:
    std::latch& m_gate;
};
}  // namespace rocprofsys::collection::testing
