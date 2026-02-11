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

#include "lib/common/thread_activity.hpp"
#include "lib/common/filesystem.hpp"
#include "lib/common/logging.hpp"

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <sstream>
#include <string>

namespace rocprofiler
{
namespace common
{
namespace thread_activity
{
namespace
{
namespace fs = ::rocprofiler::common::filesystem;

std::optional<task_schedstat>
read_schedstat(const task_info& task)
{
    auto _inp = fs::path{task.path} / "schedstat";
    if(!fs::exists(_inp)) return std::nullopt;

    auto _data = task_schedstat{};
    if(auto ifs = std::ifstream{_inp}; ifs.is_open())
    {
        auto line = std::string{};
        std::getline(ifs, line);
        if(!ifs) return std::nullopt;

        auto iss = std::istringstream{line};
        if(!(iss >> _data.run_time_ns >> _data.run_queue_ns >> _data.timeslices))
            return std::nullopt;

        return _data;
    }

    return std::nullopt;
}

std::optional<task_stat>
read_stat(const task_info& task)
{
    auto _inp = fs::path{task.path} / "stat";
    if(!fs::exists(_inp)) return std::nullopt;

    auto _data = task_stat{};
    if(auto ifs = std::ifstream{_inp}; ifs.is_open())
    {
        auto line = std::string{};
        std::getline(ifs, line);
        if(!ifs || line.empty()) return std::nullopt;

        // Find the last ')' that ends comm
        auto rparen = line.rfind(')');
        if(rparen == std::string::npos) return std::nullopt;

        // After ") " should be: state then the rest of fields
        // Example: "12345 (my thread) R 1 2 3 ... "
        if(rparen + 2 >= line.size()) return std::nullopt;

        auto _get_status = [&task](char _state) {
            switch(_state)
            {
                // 'R' == runnable but we saw no CPU in our window
                case 'R': return status::RunnableWaiting;
                case 'S': return status::Sleeping;
                case 'D': return status::BlockedUninterruptible;
                case 'T':
                case 't': return status::StoppedOrTraced;
                case 'Z': return status::Zombie;
                case 'X': return status::Dead;
                case 'W': return status::Paging;
                case 'P': return status::Parked;
            }

            ROCP_CI_LOG(INFO) << fmt::format(
                "Unknown thread state for {} ({}): '{}'", task.id, task.path, _state);
            return status::Unknown;
        };

        // state is the char after ") "
        auto state  = line.at(rparen + 2);
        _data.state = _get_status(state);

        // The remainder after state+space contains fields starting from ppid (field 4)
        // We need utime (field 14) and stime (field 15), counting from the beginning.
        // Since we already consumed fields 1-3 (pid, comm, state),
        // the remainder begins at field 4.
        auto rest = std::string{};
        if(rparen + 4 <= line.size()) rest = line.substr(rparen + 4);

        auto iss = std::istringstream{rest};

        // Fields 4..13: 10 numbers to skip (ppid..cmajflt)
        // Then field 14: utime, field 15: stime.
        for(int i = 0; i < 10; ++i)
        {
            uint64_t skip = 0;
            if(!(iss >> skip)) return std::nullopt;
        }

        if(!(iss >> _data.utime_ticks >> _data.stime_ticks)) return std::nullopt;

        return _data;
    }

    return std::nullopt;
}

status
classify(const sample& _data)
{
    // Strongest signals first
    if(_data.schedstat.has_value() && _data.schedstat->run_time_ns > 0)
    {
        return status::ActiveOnCPU;
    }
    else if(_data.stat.has_value() && (_data.stat->utime_ticks + _data.stat->stime_ticks) > 0)
    {
        return status::ActiveOnCPU;
    }
    else if(_data.schedstat.has_value() && _data.schedstat->run_queue_ns > 0)
    {
        return status::RunnableWaiting;
    }
    else if(_data.stat.has_value())
    {
        // If no CPU delta, use state as a hint
        return _data.stat->state;
    }
    else if(!_data.task)
    {
        // task disappeared
        return status::Gone;
    }

    return status::Unknown;
}

bool
default_poll_tasks_predicate(const std::map<task_info, status>& data)
{
    uint64_t gone    = 0;
    uint64_t unknown = 0;
    for(const auto& [task, status] : data)
    {
        ROCP_TRACE << fmt::format("Task {} status: {}", task, status);
        if(status == status::Unknown)
            ++unknown;
        else if(status == status::Gone)
            ++gone;
    }

    ROCP_TRACE << fmt::format(
        "default_poll_tasks_predicate: {}/{} tasks are Gone ({}) or Unknown ({})",
        gone + unknown,
        data.size(),
        gone,
        unknown);

    // return true to stop because all tasks are either gone or unknown, false to keep polling until
    // timeout
    return ((gone + unknown) == data.size());
}
}  // namespace

namespace operators
{
bool
operator==(const task_info& lhs, const task_info& rhs)
{
    return std::tie(lhs.id, lhs.path) == std::tie(rhs.id, rhs.path);
}

bool
operator<(const task_info& lhs, const task_info& rhs)
{
    return std::tie(lhs.id, lhs.path) < std::tie(rhs.id, rhs.path);
}

bool
operator<(task_schedstat lhs, task_schedstat rhs)
{
    return std::tie(lhs.run_time_ns, lhs.run_queue_ns, lhs.timeslices) <
           std::tie(rhs.run_time_ns, rhs.run_queue_ns, rhs.timeslices);
}

bool
operator<(task_stat lhs, task_stat rhs)
{
    auto _lhs_ticks = lhs.utime_ticks + lhs.stime_ticks;
    auto _rhs_ticks = rhs.utime_ticks + rhs.stime_ticks;
    return std::tie(_lhs_ticks, lhs.state) < std::tie(_rhs_ticks, rhs.state);
}

task_schedstat&
operator-=(task_schedstat& lhs, task_schedstat rhs)
{
    if(lhs < rhs) std::swap(lhs, rhs);

    lhs.run_time_ns -= rhs.run_time_ns;
    lhs.run_queue_ns -= rhs.run_queue_ns;
    lhs.timeslices -= rhs.timeslices;

    return lhs;
}

task_stat&
operator-=(task_stat& lhs, task_stat rhs)
{
    if(lhs < rhs) std::swap(lhs, rhs);

    lhs.utime_ticks -= rhs.utime_ticks;
    lhs.stime_ticks -= rhs.stime_ticks;
    lhs.state = std::min(lhs.state, rhs.state);  // keep the "less active" state
    return lhs;
}

sample&
operator-=(sample& lhs, const sample& rhs)
{
    if(lhs.task == rhs.task)
    {
        if(lhs.schedstat && rhs.schedstat)
        {
            *lhs.schedstat -= *rhs.schedstat;
        }

        if(lhs.stat && rhs.stat)
        {
            *lhs.stat -= *rhs.stat;
        }
    }
    else
    {
        ROCP_CI_LOG(INFO) << fmt::format(
            "operator-(sample, sample) called on two different tasks ({} vs {})",
            lhs.task.id,
            rhs.task.id);
    }

    return lhs;
}

task_schedstat
operator-(task_schedstat lhs, task_schedstat rhs)
{
    return (lhs -= rhs);
}

task_stat
operator-(task_stat lhs, task_stat rhs)
{
    return (lhs -= rhs);
}

sample
operator-(const sample& lhs, const sample& rhs)
{
    if(lhs.task == rhs.task)
    {
        auto _ret = lhs;  // make a copy
        return (_ret -= rhs);
    }

    // return a sample with only the task info if tasks don't match
    return sample{lhs.task, std::nullopt, std::nullopt};
}
}  // namespace operators

task_info::operator bool() const
{
    if(id == 0 || path.empty()) return false;
    return fs::exists(fs::path{path} / "stat");
}

sample::operator bool() const { return (task && (schedstat.has_value() || stat.has_value())); }

std::set<task_info>
get_tasks(pid_t _pid, const std::set<task_info>& exclude_tasks)
{
    auto _tasks = std::set<task_info>{};

    ROCP_TRACE << fmt::format("Reading tasks in /proc/{}/task/*", _pid);
    for(const auto& itr : fs::directory_iterator{fs::path{fmt::format("/proc/{}/task", _pid)}})
    {
        if(auto path = fs::path{itr}; fs::exists(path / "stat") || fs::exists(path / "schedstat"))
        {
            ROCP_TRACE << fmt::format("- Reading task info from {}", path.string());

            auto _id   = std::stoull(path.filename().string());
            auto _task = task_info{_id, path.string()};
            if(exclude_tasks.count(_task) == 0u)
            {
                ROCP_TRACE << fmt::format(
                    "   - Including task {} ({}) in get_tasks result", _id, path.string());
                _tasks.emplace(_task);
            }
            else
            {
                ROCP_TRACE << fmt::format(
                    "   - Excluding task {} ({}) from get_tasks result", _id, path.string());
            }
        }
    }

    ROCP_TRACE << fmt::format("Found {} tasks in /proc/{}/task", _tasks.size(), _pid);
    return _tasks;
}

sample
get_sample(const task_info& task)
{
    return sample{task, read_schedstat(task), read_stat(task)};
}

std::map<task_info, status>
poll_tasks(std::set<task_info>&      tasks,
           poll_tasks_predicate_t    predicate,
           std::chrono::milliseconds min_interval,
           std::chrono::milliseconds timeout)
{
    auto _data       = std::map<task_info, status>{};
    auto _start_time = std::chrono::steady_clock::now();

    if(!predicate) predicate = &default_poll_tasks_predicate;

    auto get_elapsed_time = [&_start_time]() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - _start_time);
    };

    auto get_tasks = [&tasks]() {
        auto _tasks = tasks;
        for(const auto& task : tasks)
        {
            if(task) _tasks.emplace(task);
        }
        return _tasks;
    };

    auto get_samples = [](const auto& _tasks) {
        auto _samples = std::unordered_map<task_info, sample>{};
        _samples.reserve(_tasks.size());
        for(const auto& itr : _tasks)
        {
            auto sample = get_sample(itr);
            if(sample) _samples.emplace(itr, sample);
        }
        return _samples;
    };

    auto _base_samples = get_samples(tasks);
    do
    {
        std::this_thread::sleep_for(min_interval);
        auto _current_tasks   = get_tasks();
        auto _current_samples = get_samples(_current_tasks);

        _data.clear();

        // update status for tasks that disappeared since last check
        for(auto itr : tasks)
        {
            if(_current_tasks.count(itr) == 0u || _current_samples.count(itr) == 0u)
            {
                _data.emplace(itr, status::Gone);
            }
        }

        for(const auto& itr : _current_samples)
        {
            const auto& task   = itr.first;
            const auto& sample = itr.second;

            if(_base_samples.count(task) != 0u)
            {
                auto _delta = sample - _base_samples.at(task);
                auto status = classify(_delta);
                _data.emplace(task, status);
            }
            else
            {
                ROCP_CI_LOG(INFO) << fmt::format(
                    "New task detected! Not in baseline tasks: {} ({})", task.id, task.path);
                _data.emplace(task, status::Unknown);
            }
        }

        if(bool should_stop = predicate(_data); should_stop) break;

    } while(get_elapsed_time() < timeout);

    return _data;
}

}  // namespace thread_activity
}  // namespace common
}  // namespace rocprofiler
