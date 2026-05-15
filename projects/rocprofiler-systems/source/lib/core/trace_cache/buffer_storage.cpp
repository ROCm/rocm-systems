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

#include "logger/debug.hpp"

#include <memory>
#include <mutex>
#include <sstream>
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
        LOG_WARNING("Flush worker is already running for pid={}", current_pid);
        throw std::runtime_error("Flush worker is already running");
    }

    LOG_DEBUG("Starting flush worker for pid={}, filepath={}", current_pid, m_filepath);

    m_ofs = std::ofstream{ m_filepath, std::ios::binary | std::ios::out };

    if(!m_ofs.good())
    {
        LOG_CRITICAL("Failed to open file for writing: {}", m_filepath);
        std::stringstream _ss;
        _ss << "Error opening file for writing: " << m_filepath;
        throw std::runtime_error(_ss.str());
    }

    m_worker_synchronization->origin_pid = current_pid;
    m_worker_synchronization->is_running = true;

    m_flushing_thread = std::make_unique<std::thread>([&]() {
        LOG_TRACE("Flush worker thread started for pid={}",
                  m_worker_synchronization->origin_pid);
        std::mutex _shutdown_condition_mutex;
        while(m_worker_synchronization->is_running)
        {
            m_worker_function(m_ofs, false);
            std::unique_lock _lock{ _shutdown_condition_mutex };
            m_worker_synchronization->is_running_condition.wait_for(
                _lock, CACHE_FILE_FLUSH_TIMEOUT,
                [&]() { return !m_worker_synchronization->is_running; });
        }

        LOG_TRACE("Flush worker thread performing final flush");
        m_worker_function(m_ofs, true);
        m_ofs.close();
        m_worker_synchronization->exit_finished = true;
        m_worker_synchronization->exit_finished_condition.notify_one();
        LOG_TRACE("Flush worker thread exiting");
    });

    LOG_DEBUG("Flush worker started successfully for pid={}", current_pid);
}
void
flush_worker_t::stop(const pid_t& current_pid)
{
    LOG_DEBUG("Stopping flush worker for pid={}", current_pid);

    const bool flushing_thread_exist = m_flushing_thread != nullptr;
    const bool worker_is_running =
        m_worker_synchronization != nullptr && m_worker_synchronization->is_running;

    if(flushing_thread_exist && worker_is_running)
    {
        LOG_TRACE("Signaling flush worker to stop");
        m_worker_synchronization->is_running = false;
        m_worker_synchronization->is_running_condition.notify_all();

        const bool thread_is_created_in_this_process =
            current_pid == m_worker_synchronization->origin_pid;
        if(!thread_is_created_in_this_process)
        {
            LOG_DEBUG("Flush worker was created in different process, skipping join");
            return;
        }

        LOG_TRACE("Waiting for flush worker thread to finish");
        std::mutex       _exit_mutex;
        std::unique_lock _exit_lock{ _exit_mutex };
        m_worker_synchronization->exit_finished_condition.wait(
            _exit_lock, [&]() { return m_worker_synchronization->exit_finished.load(); });

        if(m_flushing_thread->joinable())
        {
            m_flushing_thread->join();
            m_flushing_thread.reset();
            LOG_TRACE("Flush worker thread joined successfully");
        }

        LOG_DEBUG("Flush worker stopped for pid={}", current_pid);
    }
    else
    {
        LOG_TRACE("Flush worker not running or thread doesn't exist, nothing to stop");
    }
}

}  // namespace trace_cache
}  // namespace rocprofsys
