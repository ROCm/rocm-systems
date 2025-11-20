// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "buffer_storage.hpp"
#include "PTL/Task.hh"
#include "PTL/TaskGroup.hh"
#include "PTL/ThreadPool.hh"
#include "debug.hpp"
#include "library/runtime.hpp"
#include <chrono>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unistd.h>

using namespace std::chrono_literals;

namespace rocprofsys
{
namespace trace_cache
{

flush_worker_t::flush_worker_t(worker_function_t            worker_function,
                               worker_synchronization_ptr_t worker_synchronization_ptr,
                               std::string                  filepath)

: m_worker_function(std::move(worker_function))
, m_worker_synchronization(std::move(worker_synchronization_ptr))
, m_filepath(std::move(filepath))
{}

void
flush_worker_t::start(const pid_t& current_pid)
{
    if(m_worker_synchronization->is_running)
    {
        std::stringstream _ss;
        _ss << "Flush worker is already running";
        throw std::runtime_error(_ss.str());
    }

    m_ofs = std::ofstream{ m_filepath, std::ios::binary | std::ios::out };

    if(!m_ofs.good())
    {
        std::stringstream _ss;
        _ss << "Error opening file for writing: " << m_filepath;
        throw std::runtime_error(_ss.str());
    }

    m_worker_synchronization->origin_pid = current_pid;
    m_worker_synchronization->is_running = true;

    ROCPROFSYS_SCOPED_SAMPLING_ON_CHILD_THREADS(false);

    m_flushing_thread = std::make_unique<std::thread>([&]() {
        std::mutex _shutdown_condition_mutex;
        while(m_worker_synchronization->is_running)
        {
            m_worker_function(m_ofs, false);
            std::unique_lock _lock{ _shutdown_condition_mutex };
            m_worker_synchronization->is_running_condition.wait_for(
                _lock, CACHE_FILE_FLUSH_TIMEOUT,
                [&]() { return !m_worker_synchronization->is_running; });
        }

        m_worker_function(m_ofs, true);
        m_ofs.close();
        m_worker_synchronization->exit_finished = true;
        m_worker_synchronization->exit_finished_condition.notify_one();
    });
}
void
flush_worker_t::stop(const pid_t& current_pid)
{
    const bool flushing_thread_exist = m_flushing_thread != nullptr;
    const bool worker_is_running =
        m_worker_synchronization != nullptr && m_worker_synchronization->is_running;

    if(flushing_thread_exist && worker_is_running)
    {
        ROCPROFSYS_DEBUG("Buffer storage shutting down..\n");
        m_worker_synchronization->is_running = false;
        m_worker_synchronization->is_running_condition.notify_all();

        const bool thread_is_created_in_this_process =
            current_pid == m_worker_synchronization->origin_pid;
        if(!thread_is_created_in_this_process)
        {
            ROCPROFSYS_DEBUG(
                "Buffer storage is not created in same process as shutting down..\n");
            return;
        }

        std::mutex       _exit_mutex;
        std::unique_lock _exit_lock{ _exit_mutex };
        m_worker_synchronization->exit_finished_condition.wait(
            _exit_lock, [&]() { return m_worker_synchronization->exit_finished.load(); });

        if(m_flushing_thread->joinable())
        {
            m_flushing_thread->join();
            m_flushing_thread.reset();
        }
    }
}

}  // namespace trace_cache
}  // namespace rocprofsys
