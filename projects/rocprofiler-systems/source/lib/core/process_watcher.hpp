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

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

namespace rocprofsys
{
enum class process_event
{
    SPAWNED,
    EXITED,
    SIGNALED,
    STOPPED
};

struct process_info
{
    pid_t       pid         = -1;
    pid_t       ppid        = -1;
    int         exit_status = 0;
    std::string name        = {};

    std::chrono::steady_clock::time_point                start_time = {};
    std::optional<std::chrono::steady_clock::time_point> end_time   = std::nullopt;

    bool is_running() const { return !end_time.has_value(); }

    std::chrono::milliseconds duration() const
    {
        auto _end = end_time.value_or(std::chrono::steady_clock::now());
        return std::chrono::duration_cast<std::chrono::milliseconds>(_end - start_time);
    }

    std::string to_string() const
    {
        std::stringstream _ss;
        _ss << "pid: " << pid << ", ppid: " << ppid << ", name: " << name
            << ", start_time: " << start_time.time_since_epoch().count() << ", end_time: "
            << end_time.value_or(std::chrono::steady_clock::now())
                   .time_since_epoch()
                   .count()
            << ", duration: " << duration().count();
        return _ss.str();
    }
};

/**
 * @brief Monitors child processes spawned by a target process.
 *
 * The process_watcher polls /proc/<pid>/task/<pid>/children to discover
 * new child processes and uses waitpid() to detect process termination.
 * Callbacks are invoked for SPAWNED, EXITED, SIGNALED, and STOPPED events.
 */
class process_watcher
{
public:
    using callback_t      = std::function<void(process_event, const process_info&)>;
    using poll_interval_t = std::chrono::milliseconds;

    explicit process_watcher(pid_t target_pid = getpid());
    ~process_watcher();

    process_watcher(const process_watcher&)            = delete;
    process_watcher& operator=(const process_watcher&) = delete;
    process_watcher(process_watcher&&)                 = delete;
    process_watcher& operator=(process_watcher&&)      = delete;

    void start();
    void stop();

    bool is_running() const { return m_running.load(std::memory_order_acquire); }

    void            set_poll_interval(poll_interval_t interval);
    poll_interval_t get_poll_interval() const;

    void register_callback(callback_t callback);
    void clear_callbacks();

    std::vector<process_info>   get_active_processes() const;
    std::vector<process_info>   get_all_processes() const;
    std::optional<process_info> get_process(pid_t pid) const;
    size_t                      active_count() const;

    pid_t add_process(pid_t pid);
    void  remove_process(pid_t pid);

    int wait_for_all(int opts = 0);
    int wait_for(pid_t pid, int opts = 0);

    pid_t get_target_pid() const { return m_target_pid; }

private:
    void        monitor_loop();
    void        poll_children();
    void        check_process_status();
    void        notify_callbacks(process_event event, const process_info& info);
    std::string read_process_name(pid_t pid) const;

    pid_t             m_target_pid     = -1;
    poll_interval_t   m_poll_interval  = std::chrono::milliseconds{ 100 };
    std::atomic<bool> m_running        = { false };
    std::atomic<bool> m_stop_requested = { false };

    mutable std::mutex      m_mutex          = {};
    std::condition_variable m_cv             = {};
    std::thread             m_monitor_thread = {};

    std::map<pid_t, process_info> m_processes = {};
    std::vector<callback_t>       m_callbacks = {};
};

const char*
process_event_to_string(process_event event);

}  // namespace rocprofsys
