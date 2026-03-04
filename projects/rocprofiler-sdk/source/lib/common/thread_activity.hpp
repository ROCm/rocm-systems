// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc.
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

#include "lib/common/logging.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <chrono>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <sys/poll.h>
#include <sys/types.h>
#include <unistd.h>  // sysconf

namespace rocprofiler
{
namespace common
{
namespace thread_activity
{
struct task_info;

namespace operators
{
bool
operator<(const task_info& lhs, const task_info& rhs);
}  // namespace operators

enum class status
{
    // should never be seen
    Dead,
    // thread no longer exists
    Gone,
    // terminated but not yet reaped by parent
    Zombie,
    // usually I/O operations that cannot be interrupted
    BlockedUninterruptible,
    // a state for kernel threads associated with offlined CPUs, uncommon for user processes
    Parked,
    // not valid since the 2.6.xx kernel
    Paging,
    // either by a job control signal (e.g., SIGSTOP) or because it is being traced by a debugger
    StoppedOrTraced,
    // Interruptible sleep, waiting for an event to complete. This is the most common sleeping state
    // for user threads
    Sleeping,
    // Waiting in the run queue for CPU time.
    RunnableWaiting,
    // Actively executing on a CPU. This is the only state where we can be sure the thread is
    // actively running and not just waiting to run
    ActiveOnCPU,
    Unknown,
};

struct task_info
{
    uint64_t    id   = 0;
    std::string path = {};

    operator bool() const;

    friend bool operator<(const task_info& lhs, const task_info& rhs)
    {
        return operators::operator<(lhs, rhs);
    }
};

struct task_schedstat
{
    uint64_t run_time_ns  = 0;
    uint64_t run_queue_ns = 0;
    uint64_t timeslices   = 0;
};

struct task_stat
{
    uint64_t utime_ticks = 0;
    uint64_t stime_ticks = 0;
    status   state       = status::Unknown;
};

struct sample
{
    task_info                     task      = {};
    std::optional<task_schedstat> schedstat = std::nullopt;
    std::optional<task_stat>      stat      = std::nullopt;

    operator bool() const;
};

using task_set_t        = std::set<task_info>;
using task_status_map_t = std::map<task_info, status>;
// return true from predicate if we should stop polling, false to keep polling until timeout
using poll_tasks_predicate_t = std::function<bool(const task_status_map_t&)>;

task_set_t
get_tasks(pid_t pid = getpid(), const task_set_t& exclude_tasks = {});

sample
get_sample(const task_info& task);

// if predicate is not provided, will keep polling until all tasks are either Gone or Unknown, or
// until timeout
task_status_map_t
poll_tasks(task_set_t&               tasks,
           poll_tasks_predicate_t    predicate    = nullptr,
           std::chrono::milliseconds min_interval = std::chrono::milliseconds{1},
           std::chrono::milliseconds timeout      = std::chrono::milliseconds{1000});

namespace operators
{
bool
operator==(const task_info& lhs, const task_info& rhs);

bool
operator<(const task_info& lhs, const task_info& rhs);

bool
operator<(task_schedstat lhs, task_schedstat rhs);

bool
operator<(task_stat lhs, task_stat rhs);

task_schedstat&
operator-=(task_schedstat& lhs, task_schedstat rhs);

task_stat&
operator-=(task_stat& lhs, task_stat rhs);

sample&
operator-=(sample& lhs, const sample& rhs);

task_schedstat
operator-(task_schedstat lhs, task_schedstat rhs);

task_stat
operator-(task_stat lhs, task_stat rhs);

sample
operator-(const sample& lhs, const sample& rhs);
}  // namespace operators
}  // namespace thread_activity
}  // namespace common
}  // namespace rocprofiler

// Bring operators into the common namespace for easier use
using namespace ::rocprofiler::common::thread_activity::operators;

namespace std
{
template <>
struct hash<rocprofiler::common::thread_activity::task_info>
{
    std::size_t operator()(const rocprofiler::common::thread_activity::task_info& t) const noexcept
    {
        return std::hash<uint64_t>{}(t.id) ^ std::hash<std::string>{}(t.path);
    }
};
}  // namespace std

namespace fmt
{
template <>
struct formatter<::rocprofiler::common::thread_activity::status>
{
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename Ctx>
    auto format(const ::rocprofiler::common::thread_activity::status& val, Ctx& ctx) const
    {
        using status = ::rocprofiler::common::thread_activity::status;
        switch(val)
        {
            case status::Dead: return fmt::format_to(ctx.out(), "Dead");
            case status::Paging: return fmt::format_to(ctx.out(), "Paging");
            case status::Parked: return fmt::format_to(ctx.out(), "Parked");
            case status::Gone: return fmt::format_to(ctx.out(), "Gone");
            case status::Zombie: return fmt::format_to(ctx.out(), "Zombie");
            case status::BlockedUninterruptible:
                return fmt::format_to(ctx.out(), "BlockedUninterruptible");
            case status::StoppedOrTraced: return fmt::format_to(ctx.out(), "StoppedOrTraced");
            case status::Sleeping: return fmt::format_to(ctx.out(), "Sleeping");
            case status::RunnableWaiting: return fmt::format_to(ctx.out(), "RunnableWaiting");
            case status::ActiveOnCPU: return fmt::format_to(ctx.out(), "ActiveOnCPU");
            case status::Unknown: return fmt::format_to(ctx.out(), "Unknown");
        }

        ROCP_CI_LOG(INFO) << fmt::format("Unknown status value: {}", static_cast<int>(val));
        return fmt::format_to(ctx.out(), "Unknown");
    }
};

template <>
struct formatter<::rocprofiler::common::thread_activity::task_info>
{
    template <typename ParseContext>
    constexpr auto parse(ParseContext& ctx)
    {
        return ctx.begin();
    }

    template <typename Ctx>
    auto format(const ::rocprofiler::common::thread_activity::task_info& val, Ctx& ctx) const
    {
        return fmt::format_to(ctx.out(), "{{id: {}, path: {}}}", val.id, val.path);
    }
};
}  // namespace fmt
