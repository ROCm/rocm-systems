// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <condition_variable>
#include <mutex>

namespace rocprofsys::collection
{
/** Keeps polls out of fork(): counts passes and forks. Polls never exclude each other.
 * One gate per process sampler state; a forked child replaces the whole state. */
class fork_gate
{
public:
    /** RAII: one poll. Converts to false while a fork is in progress: skip the work. */
    class pass
    {
    public:
        /** Tag: wait for a fork in progress instead of converting to false. */
        struct wait_t
        {
            explicit wait_t() = default;
        };
        static constexpr wait_t k_wait{};

        explicit pass(fork_gate& gate);
        /** Always entered. Waits while a fork is in progress, unless the calling
         * thread already holds a pass on the gate (that fork waits for it). */
        pass(fork_gate& gate, wait_t);
        /** Leaves the gate if it was entered. */
        ~pass();

        pass(const pass&)            = delete;
        pass& operator=(const pass&) = delete;

        [[nodiscard]] explicit operator bool() const noexcept { return m_entered; }

    private:
        fork_gate& m_gate;
        /** The calling thread's innermost held gate and its depth before this pass;
         * restored when this pass leaves. */
        const fork_gate* m_outer_gate  = nullptr;
        int              m_outer_depth = 0;
        bool             m_entered     = false;
    };

    /** Before fork(): counts the fork and waits until no pass is held. The passes held
     * by the calling thread (a poll that forks, nested passes included) are discounted
     * first. Nests; several threads may fork. */
    void block();
    /** After fork() in the parent. Restores the discounted passes. */
    void unblock();

private:
    /** @return false while a fork is in progress */
    [[nodiscard]] bool try_enter();
    /** Waits until no fork is in progress (no wait when the calling thread already
     * holds a pass on this gate), then enters. */
    void enter();
    /** Restores the calling thread's held gate and depth to the given outer ones. */
    void leave(const fork_gate* outer_gate, int outer_depth);

    std::mutex              m_mutex;
    std::condition_variable m_idle;
    int                     m_polls = 0;
    int                     m_forks = 0;
};
}  // namespace rocprofsys::collection
