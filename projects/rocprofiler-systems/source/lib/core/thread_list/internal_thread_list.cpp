// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/thread_list/internal_thread_list.hpp"

#include "core/state.hpp"

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <memory>
#include <string>
#include <sys/syscall.h>
#include <system_error>
#include <unistd.h>
#include <utility>

namespace rocprofsys::thread_list
{
namespace
{
constexpr auto k_task_dir = "/proc/self/task";

// Marks the calling thread Internal while it takes one of the list's locks, so the
// lock calls are not seen by the profiling hooks.
[[nodiscard]] state::thread::scoped_guard
internal_scope()
{
    return state::thread::scoped(state::thread::Internal);
}

pid_t
current_tid() noexcept
{
    return static_cast<pid_t>(::syscall(SYS_gettid));
}

// Returns false when the directory cannot be listed.
bool
list_live_tids(std::unordered_set<pid_t>& live)
{
    std::error_code error;
    for(const auto& entry : std::filesystem::directory_iterator{ k_task_dir, error })
    {
        const auto name = entry.path().filename().string();
        pid_t      tid  = 0;
        const auto res  = std::from_chars(name.data(), name.data() + name.size(), tid);
        if(res.ec == std::errc{}) live.insert(tid);
    }
    return !error;
}
}  // namespace

void
internal_thread_list::add(pid_t tid)
{
    const auto      guard = internal_scope();
    std::lock_guard lock{ m_mutex };
    m_tids.insert(tid);
}

void
internal_thread_list::remove(pid_t tid)
{
    const auto      guard = internal_scope();
    std::lock_guard lock{ m_mutex };
    m_tids.erase(tid);
}

bool
internal_thread_list::is_excluded(pid_t tid) const
{
    const auto      guard = internal_scope();
    std::lock_guard lock{ m_mutex };
    return m_tids.contains(tid);
}

void
internal_thread_list::prune_exited()
{
    // The listing is read while m_mutex is held: every entry present was added before
    // the read, so its thread existed then and is in the snapshot unless it exited. A
    // racing add() waits on m_mutex and lands after the prune.
    const auto                guard = internal_scope();
    std::lock_guard           lock{ m_mutex };
    std::unordered_set<pid_t> live;
    if(!list_live_tids(live)) return;
    std::erase_if(m_tids, [&live](pid_t tid) { return !live.contains(tid); });
}

internal_thread_list::creation_scope
internal_thread_list::begin_creation() const
{
    const auto guard = internal_scope();
    return creation_scope{ m_gate };
}

internal_thread_list::rescan_scope
internal_thread_list::begin_rescan() const
{
    const auto guard = internal_scope();
    return rescan_scope{ m_gate };
}

internal_thread_list::rescan_scope
internal_thread_list::try_begin_rescan() const
{
    const auto guard = internal_scope();
    return rescan_scope{ m_gate, std::try_to_lock };
}

void
internal_thread_list::reset_after_fork() noexcept
{
    // Mirrors cpu-snap's fork-child handling: the locks may be held by threads that do
    // not exist in the child, so they are re-constructed instead of unlocked.
    std::construct_at(&m_gate);
    std::construct_at(&m_mutex);
    m_tids.clear();
}

scoped_internal_thread::scoped_internal_thread(internal_thread_list& list)
: m_list{ &list }
, m_tid{ current_tid() }
{
    m_list->add(m_tid);
}

scoped_internal_thread::~scoped_internal_thread()
{
    if(m_list != nullptr) m_list->remove(m_tid);
}

scoped_internal_thread::scoped_internal_thread(scoped_internal_thread&& other) noexcept
: m_list{ std::exchange(other.m_list, nullptr) }
, m_tid{ other.m_tid }
{}
}  // namespace rocprofsys::thread_list
