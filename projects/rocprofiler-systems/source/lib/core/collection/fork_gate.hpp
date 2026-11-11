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
        explicit pass(fork_gate& gate);
        /** Leaves the gate if it was entered. */
        ~pass();

        pass(const pass&)            = delete;
        pass& operator=(const pass&) = delete;

        [[nodiscard]] explicit operator bool() const noexcept { return m_entered; }

    private:
        fork_gate& m_gate;
        bool       m_entered = false;
    };

    /** Before fork(): counts the fork and waits until no pass is held. A pass held by
     * the calling thread (a poll that forks) is discounted first. Nests; several
     * threads may fork. */
    void block();
    /** After fork() in the parent. Restores a discounted pass. */
    void unblock();

private:
    /** @return false while a fork is in progress */
    [[nodiscard]] bool try_enter();
    void               leave();

    std::mutex              m_mutex;
    std::condition_variable m_idle;
    int                     m_polls = 0;
    int                     m_forks = 0;
};
}  // namespace rocprofsys::collection
