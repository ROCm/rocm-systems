// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "cpu_snap/sampling/thread_tracker.hpp"

#include <algorithm>
#include <exception>
#include <string>
#include <utility>

#include <unistd.h>

#include "cpu_snap/error.hpp"
#include "cpu_snap/sampling/thread_comm.hpp"

namespace cpu_snap::sampling::internal
{

thread_source
proc_thread_source()
{
    return []() -> std::optional<std::vector<thread_entry>> {
        const auto tids = utils::try_list_self_thread_tids();
        if(!tids) return std::nullopt;
        std::vector<thread_entry> out;
        out.reserve(tids->size());
        for(const pid_t tid : *tids)
        {
            out.push_back({ tid, utils::read_thread_start_ticks(tid) });
        }
        return out;
    };
}

thread_tracker::thread_tracker(thread_source source, log_sink logger, clock_kind clock,
                               bool                       include_main_thread,
                               std::function<bool(pid_t)> is_excluded)
: m_source{ std::move(source) }
, m_logger{ std::move(logger) }
, m_clock{ clock }
, m_include_main_thread{ include_main_thread }
, m_is_excluded{ std::move(is_excluded) }
{}

void
thread_tracker::attach_initial(triggers::i_trigger& trigger, pid_t exclude_tid)
{
    const auto threads = m_source();
    if(!threads)
    {
        throw environment_error("sampler: cannot enumerate /proc/self/task");
    }
    for(const auto& entry : *threads)
    {
        if(!m_include_main_thread && entry.tid == static_cast<pid_t>(::getpid()))
        {
            continue;
        }
        try_attach(trigger, entry, exclude_tid);
    }
}

std::optional<std::vector<thread_entry>>
thread_tracker::poll()
{
    auto current = m_source();
    if(!current)
    {
        if(!std::exchange(m_listing_failed, true))
        {
            emit(m_logger, m_clock, log_level::warn,
                 "cannot enumerate /proc/self/task; thread rescan failed");
        }
        return std::nullopt;
    }
    m_listing_failed = false;
    return current;
}

rescan_result
thread_tracker::rescan(triggers::i_trigger& trigger, streaming_pipeline* pipeline,
                       pid_t exclude_tid, const std::vector<thread_entry>& current)
{
    rescan_result result;

    std::unordered_map<pid_t, std::optional<std::uint64_t>> live;
    for(const auto& entry : current)
    {
        live.emplace(entry.tid, entry.start_ticks);
    }

    const auto recycled = [&](pid_t tid) {
        const auto known = m_tid_start_ticks.find(tid);
        if(known == m_tid_start_ticks.end()) return false;
        const auto& now = live.at(tid);
        return now && *now != known->second;
    };

    std::vector<pid_t> tracked;
    trigger.collect_tids(tracked);
    for(const pid_t tid : tracked)
    {
        if(live.contains(tid) && !recycled(tid) && !is_excluded(tid))
        {
            continue;
        }
        if(pipeline)
        {
            pipeline->on_tid_exit(tid);
        }
        trigger.detach_tid(tid);
        m_tid_start_ticks.erase(tid);
    }

    std::erase_if(m_logged_attach_failures,
                  [&live](pid_t tid) { return !live.contains(tid); });
    for(const auto& entry : current)
    {
        try
        {
            try_attach(trigger, entry, exclude_tid);
            m_logged_attach_failures.erase(entry.tid);
        } catch(const std::exception& e)
        {
            ++result.failed_attaches;
            log_attach_failure(entry.tid, e.what());
        } catch(...)
        {
            ++result.failed_attaches;
            log_attach_failure(entry.tid, "unknown error");
        }
    }
    return result;
}

void
thread_tracker::reset() noexcept
{
    m_tid_start_ticks.clear();
    m_logged_attach_failures.clear();
}

void
thread_tracker::try_attach(triggers::i_trigger& trigger, const thread_entry& entry,
                           pid_t exclude_tid)
{
    if(entry.tid == exclude_tid || is_excluded(entry.tid) ||
       trigger.is_tracked(entry.tid))
    {
        return;
    }
    trigger.attach_tid(entry.tid);
    if(entry.start_ticks)
    {
        m_tid_start_ticks[entry.tid] = *entry.start_ticks;
    }
}

bool
thread_tracker::is_excluded(pid_t tid) const
{
    return m_is_excluded && m_is_excluded(tid);
}

void
thread_tracker::log_attach_failure(pid_t tid, std::string_view what)
{
    if(m_logged_attach_failures.insert(tid).second)
    {
        emit(m_logger, m_clock, log_level::warn, [&] {
            return "cannot attach to tid " + std::to_string(tid) + ": " +
                   std::string(what);
        });
    }
}

}  // namespace cpu_snap::sampling::internal
