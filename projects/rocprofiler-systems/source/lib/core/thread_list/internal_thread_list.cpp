// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "core/thread_list/internal_thread_list.hpp"

#include "common/static_object.hpp"
#include "core/state.hpp"
#include "logger/debug.hpp"

#include <charconv>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <sys/syscall.h>
#include <system_error>
#include <unistd.h>
#include <unordered_set>
#include <utility>
#include <vector>

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
        if(res.ec == std::errc{})
        {
            live.insert(tid);
        }
    }
    return !error;
}

// Erases the TIDs in @p tids that are not in /proc/self/task and returns them. Erases
// nothing when the directory cannot be listed.
std::vector<pid_t>
erase_exited(std::unordered_set<pid_t>& tids)
{
    std::unordered_set<pid_t> live;
    if(!list_live_tids(live))
    {
        return {};
    }
    std::vector<pid_t> exited;
    std::erase_if(tids, [&live, &exited](pid_t tid) {
        const bool is_gone = !live.contains(tid);
        if(is_gone)
        {
            exited.push_back(tid);
        }
        return is_gone;
    });
    return exited;
}
}  // namespace

void
internal_thread_list::add(pid_t tid)
{
    const auto guard    = internal_scope();
    bool       inserted = false;
    {
        const std::lock_guard lock{ m_mutex };
        inserted = m_tids.insert(tid).second;
    }
    if(inserted)
    {
        LOG_DEBUG("Thread {} excluded from sampling", tid);
    }
}

void
internal_thread_list::remove(pid_t tid)
{
    const auto guard   = internal_scope();
    bool       removed = false;
    {
        const std::lock_guard lock{ m_mutex };
        removed = m_tids.erase(tid) > 0;
    }
    if(removed)
    {
        LOG_DEBUG("Thread {} included in sampling again", tid);
    }
}

bool
internal_thread_list::is_excluded(pid_t tid) const
{
    const auto            guard = internal_scope();
    const std::lock_guard lock{ m_mutex };
    return m_tids.contains(tid);
}

void
internal_thread_list::prune_exited()
{
    // The listing is read while m_mutex is held: every entry present was added before
    // the read, so its thread existed then and is in the snapshot unless it exited. A
    // racing add() waits on m_mutex and lands after the prune.
    const auto         guard = internal_scope();
    std::vector<pid_t> pruned;
    {
        const std::lock_guard lock{ m_mutex };
        pruned = erase_exited(m_tids);
    }
    for(const auto tid : pruned)
    {
        LOG_DEBUG("Thread {} exited, dropped from the exclusion list", tid);
    }
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

pid_t
current_tid() noexcept
{
    return static_cast<pid_t>(::syscall(SYS_gettid));
}

std::string
thread_name(pid_t tid)
{
    std::ifstream comm{ std::string{ k_task_dir } + "/" + std::to_string(tid) + "/comm" };
    std::string   name;
    std::getline(comm, name);
    return name;
}

internal_thread_list&
get_internal_thread_list()
{
    static auto*& s_list =
        common::static_object<internal_thread_list>::construct(common::do_not_destroy{});
    return *s_list;
}

scoped_internal_thread::scoped_internal_thread(internal_thread_list& list)
: m_list{ &list }
, m_tid{ current_tid() }
{
    m_list->add(m_tid);
}

scoped_internal_thread::~scoped_internal_thread()
{
    if(m_list != nullptr)
    {
        m_list->remove(m_tid);
    }
}

scoped_internal_thread::scoped_internal_thread(scoped_internal_thread&& other) noexcept
: m_list{ std::exchange(other.m_list, nullptr) }
, m_tid{ other.m_tid }
{}
}  // namespace rocprofsys::thread_list
