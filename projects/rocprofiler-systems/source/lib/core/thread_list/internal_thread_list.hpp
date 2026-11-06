// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <mutex>
#include <shared_mutex>
#include <sys/types.h>
#include <unordered_set>

namespace rocprofsys::thread_list
{
/**
 * @brief Thread-safe set of system TIDs that must never be sampled.
 *
 * Two locks with separate jobs: `m_gate` orders creation of excluded threads (shared)
 * against a sampler rescan (exclusive); `m_mutex` is a leaf lock protecting the set, so
 * `add()`, `remove()`, `is_excluded()` and `prune_exited()` never wait on the gate.
 */
class internal_thread_list
{
public:
    using creation_scope = std::shared_lock<std::shared_mutex>;
    using rescan_scope   = std::unique_lock<std::shared_mutex>;

    /// Marks @p tid as never-sampled. Idempotent. Never touches the gate.
    void add(pid_t tid);

    /// Forgets @p tid. No-op if it is not in the list.
    void remove(pid_t tid);

    /// True when @p tid must not be sampled.
    [[nodiscard("the exclusion decision must be used")]] bool is_excluded(
        pid_t tid) const;

    /// Drops every TID that is no longer listed in /proc/self/task (thread exited).
    /// If /proc/self/task cannot be read, nothing is dropped.
    void prune_exited();

    /// Held by a creator of an excluded thread from before the real pthread_create
    /// until the child's TID is added. Many creators may hold it at once.
    [[nodiscard("the scope releases the gate when destroyed")]] creation_scope
    begin_creation() const;

    /// Held by the sampler for a whole rescan. Blocks while any creation_scope lives.
    [[nodiscard("the scope releases the gate when destroyed")]] rescan_scope
    begin_rescan() const;

    /// Non-blocking begin_rescan(). The returned scope does not own the gate
    /// (`owns_lock() == false`) when a creation is in flight.
    [[nodiscard("the scope releases the gate when destroyed")]] rescan_scope
    try_begin_rescan() const;

    /// Fork child only: re-initializes both locks and clears the set. The child
    /// inherits locks in whatever state another parent thread left them in.
    void reset_after_fork() noexcept;

private:
    mutable std::shared_mutex m_gate;
    mutable std::mutex        m_mutex;
    std::unordered_set<pid_t> m_tids;
};

/// Registers the calling thread in a list for its lifetime. Normal exit removes the
/// entry; every other exit path is covered by internal_thread_list::prune_exited().
class scoped_internal_thread
{
public:
    explicit scoped_internal_thread(internal_thread_list& list);
    ~scoped_internal_thread();

    scoped_internal_thread(const scoped_internal_thread&)            = delete;
    scoped_internal_thread& operator=(const scoped_internal_thread&) = delete;
    scoped_internal_thread(scoped_internal_thread&& other) noexcept;
    scoped_internal_thread& operator=(scoped_internal_thread&&) = delete;

private:
    internal_thread_list* m_list;  // non-owning; null after move
    pid_t                 m_tid;
};
}  // namespace rocprofsys::thread_list
