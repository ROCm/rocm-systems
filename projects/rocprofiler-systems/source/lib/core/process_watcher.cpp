// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "process_watcher.hpp"
#include "common.hpp"
#include "debug.hpp"
#include "mproc.hpp"

#include <algorithm>
#include <fstream>
#include <set>
#include <sstream>

#include <sys/wait.h>

namespace rocprofsys
{

namespace
{
std::set<pid_t>
read_children_from_proc(pid_t ppid)
{
    std::set<pid_t> children;
    if(ppid <= 0) return children;

    std::ostringstream path_stream;
    path_stream << "/proc/" << ppid << "/task/" << ppid << "/children";
    std::string   path = path_stream.str();
    std::ifstream ifs{ path };

    if(!ifs)
    {
        ROCPROFSYS_VERBOSE_F(3, "Warning! Cannot read '%s'\n", path.c_str());
        return children;
    }

    pid_t child_pid = -1;
    while(ifs >> child_pid)
    {
        if(child_pid > 0) children.insert(child_pid);
    }

    return children;
}
}  // namespace

const char*
process_event_to_string(process_event event)
{
    switch(event)
    {
        case process_event::SPAWNED: return "SPAWNED";
        case process_event::EXITED: return "EXITED";
        case process_event::SIGNALED: return "SIGNALED";
        case process_event::STOPPED: return "STOPPED";
    }
    return "UNKNOWN";
}

process_watcher::process_watcher(pid_t target_pid)
: m_target_pid(target_pid)
{
    ROCPROFSYS_VERBOSE_F(2, "process_watcher created for PID %d\n", m_target_pid);
}

process_watcher::~process_watcher() { stop(); }

void
process_watcher::start()
{
    if(m_running.exchange(true, std::memory_order_acq_rel))
    {
        ROCPROFSYS_VERBOSE_F(2, "process_watcher already running for PID %d\n",
                             m_target_pid);
        return;
    }

    m_stop_requested.store(false, std::memory_order_release);

    m_monitor_thread = std::thread([this]() { monitor_loop(); });

    ROCPROFSYS_VERBOSE_F(1,
                         "process_watcher started for PID %d (poll interval: %ld ms)\n",
                         m_target_pid, m_poll_interval.count());
}

void
process_watcher::stop()
{
    if(!m_running.exchange(false, std::memory_order_acq_rel))
    {
        return;
    }

    m_stop_requested.store(true, std::memory_order_release);
    m_cv.notify_all();

    if(m_monitor_thread.joinable()) m_monitor_thread.join();

    ROCPROFSYS_VERBOSE_F(1, "process_watcher stopped for PID %d\n", m_target_pid);
}

void
process_watcher::set_poll_interval(poll_interval_t interval)
{
    std::lock_guard<std::mutex> lock{ m_mutex };
    m_poll_interval = interval;
    m_cv.notify_all();
}

process_watcher::poll_interval_t
process_watcher::get_poll_interval() const
{
    std::lock_guard<std::mutex> lock{ m_mutex };
    return m_poll_interval;
}

void
process_watcher::register_callback(callback_t callback)
{
    std::lock_guard<std::mutex> lock{ m_mutex };
    m_callbacks.emplace_back(std::move(callback));
}

void
process_watcher::clear_callbacks()
{
    std::lock_guard<std::mutex> lock{ m_mutex };
    m_callbacks.clear();
}

std::vector<process_info>
process_watcher::get_active_processes() const
{
    std::lock_guard<std::mutex> lock{ m_mutex };
    std::vector<process_info>   result;
    result.reserve(m_processes.size());

    for(const auto& [pid, info] : m_processes)
    {
        if(info.is_running()) result.push_back(info);
    }
    return result;
}

std::vector<process_info>
process_watcher::get_all_processes() const
{
    std::lock_guard<std::mutex> lock{ m_mutex };
    std::vector<process_info>   result;
    result.reserve(m_processes.size());

    for(const auto& [pid, info] : m_processes)
    {
        result.push_back(info);
    }
    return result;
}

std::optional<process_info>
process_watcher::get_process(pid_t pid) const
{
    std::lock_guard<std::mutex> lock{ m_mutex };
    auto                        itr = m_processes.find(pid);
    if(itr != m_processes.end()) return itr->second;
    return std::nullopt;
}

size_t
process_watcher::active_count() const
{
    std::lock_guard<std::mutex> lock{ m_mutex };
    size_t                      count = 0;
    for(const auto& [pid, info] : m_processes)
    {
        if(info.is_running()) ++count;
    }
    return count;
}

pid_t
process_watcher::add_process(pid_t pid)
{
    if(pid <= 0)
    {
        ROCPROFSYS_VERBOSE_F(1, "Invalid PID %d passed to add_process\n", pid);
        return -1;
    }

    process_info info;
    info.pid        = pid;
    info.ppid       = m_target_pid;
    info.start_time = std::chrono::steady_clock::now();
    info.name       = read_process_name(pid);

    bool is_new = false;
    {
        std::lock_guard<std::mutex> lock{ m_mutex };
        auto [itr, inserted] = m_processes.emplace(pid, info);
        is_new               = inserted;
        if(!inserted)
        {
            ROCPROFSYS_VERBOSE_F(2, "Process %d already tracked\n", pid);
        }
    }

    if(is_new)
    {
        notify_callbacks(process_event::SPAWNED, info);
        ROCPROFSYS_VERBOSE_F(1, "Added process %d (%s) to watcher\n", pid,
                             info.name.c_str());
    }

    return pid;
}

void
process_watcher::remove_process(pid_t pid)
{
    std::lock_guard<std::mutex> lock{ m_mutex };
    auto                        erased = m_processes.erase(pid);
    if(erased > 0)
    {
        ROCPROFSYS_VERBOSE_F(2, "Removed process %d from watcher\n", pid);
    }
}

int
process_watcher::wait_for_all(int opts)
{
    int max_exit_code = 0;

    std::vector<pid_t> pids;
    {
        std::lock_guard<std::mutex> lock{ m_mutex };
        for(const auto& [pid, info] : m_processes)
        {
            if(info.is_running()) pids.push_back(pid);
        }
    }

    for(auto pid : pids)
    {
        int ec = wait_for(pid, opts);
        if(ec > max_exit_code) max_exit_code = ec;
    }

    return max_exit_code;
}

int
process_watcher::wait_for(pid_t pid, int opts)
{
    int status = mproc::wait_pid(pid, opts);

    process_info info_copy;
    {
        std::lock_guard<std::mutex> lock{ m_mutex };
        auto                        itr = m_processes.find(pid);
        if(itr != m_processes.end())
        {
            itr->second.exit_status = status;
            itr->second.end_time    = std::chrono::steady_clock::now();
            info_copy               = itr->second;
        }
    }

    return mproc::diagnose_status(pid, status, get_verbose());
}

void
process_watcher::monitor_loop()
{
    ROCPROFSYS_VERBOSE_F(2, "Monitor thread started for PID %d\n", m_target_pid);

    while(!m_stop_requested.load(std::memory_order_acquire))
    {
        poll_children();
        check_process_status();

        std::unique_lock<std::mutex> lock{ m_mutex };
        m_cv.wait_for(lock, m_poll_interval, [this]() {
            return m_stop_requested.load(std::memory_order_acquire);
        });
    }

    ROCPROFSYS_VERBOSE_F(2, "Monitor thread exiting for PID %d\n", m_target_pid);
}

void
process_watcher::poll_children()
{
    auto children = read_children_from_proc(m_target_pid);

    std::vector<std::pair<pid_t, process_info>> new_processes;

    {
        std::lock_guard<std::mutex> lock{ m_mutex };

        for(auto child_pid : children)
        {
            if(m_processes.find(child_pid) == m_processes.end())
            {
                process_info info;
                info.pid        = child_pid;
                info.ppid       = m_target_pid;
                info.start_time = std::chrono::steady_clock::now();
                info.name       = read_process_name(child_pid);

                m_processes.emplace(child_pid, info);
                new_processes.emplace_back(child_pid, info);

                ROCPROFSYS_VERBOSE_F(2, "Detected new child process %d (%s)\n", child_pid,
                                     info.name.c_str());
            }
        }
    }

    for(const auto& [pid, info] : new_processes)
    {
        notify_callbacks(process_event::SPAWNED, info);
    }
}

void
process_watcher::check_process_status()
{
    std::vector<std::pair<pid_t, process_info*>> to_check;

    {
        std::lock_guard<std::mutex> lock{ m_mutex };
        for(auto& [pid, info] : m_processes)
        {
            if(info.is_running()) to_check.emplace_back(pid, &info);
        }
    }

    for(auto& [pid, info_ptr] : to_check)
    {
        int   status = 0;
        pid_t result = waitpid(pid, &status, WNOHANG | WUNTRACED);

        if(result > 0)
        {
            process_event event = process_event::EXITED;

            if(WIFSIGNALED(status))
            {
                event = process_event::SIGNALED;
            }
            else if(WIFSTOPPED(status))
            {
                event = process_event::STOPPED;
            }

            process_info info_copy;
            {
                std::lock_guard<std::mutex> lock{ m_mutex };
                info_ptr->exit_status = status;
                info_ptr->end_time    = std::chrono::steady_clock::now();
                info_copy             = *info_ptr;
            }

            notify_callbacks(event, info_copy);

            int ec = mproc::diagnose_status(pid, status, get_verbose());
            ROCPROFSYS_VERBOSE_F(
                1, "Process %d (%s) %s with code %d (duration: %ld ms)\n", pid,
                info_copy.name.c_str(), process_event_to_string(event), ec,
                info_copy.duration().count());
        }
    }
}

void
process_watcher::notify_callbacks(process_event event, const process_info& info)
{
    std::vector<callback_t> callbacks;
    {
        std::lock_guard<std::mutex> lock{ m_mutex };
        callbacks = m_callbacks;
    }

    for(const auto& cb : callbacks)
    {
        if(cb)
        {
            try
            {
                cb(event, info);
            } catch(const std::exception& e)
            {
                ROCPROFSYS_VERBOSE_F(0, "Exception in process_watcher callback: %s\n",
                                     e.what());
            }
        }
    }
}

std::string
process_watcher::read_process_name(pid_t pid) const
{
    std::ostringstream path_stream;
    path_stream << "/proc/" << pid << "/comm";
    std::string   path = path_stream.str();
    std::ifstream ifs{ path };

    if(!ifs) return {};

    std::string name;
    std::getline(ifs, name);

    // Remove trailing newline if present
    if(!name.empty() && name.back() == '\n') name.pop_back();

    return name;
}

}  // namespace rocprofsys
