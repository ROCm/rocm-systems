// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

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

    m_worker_synchronization->origin_pid    = current_pid;
    m_worker_synchronization->exit_finished = false;
    m_worker_synchronization->is_running    = true;

    m_flushing_thread = std::make_unique<std::thread>([&]() {
        LOG_TRACE("Flush worker thread started for pid={}",
                  m_worker_synchronization->origin_pid);
        while(m_worker_synchronization->is_running)
        {
            m_worker_function(m_ofs, false);
            std::unique_lock _lock{ m_worker_synchronization->is_running_mutex };
            m_worker_synchronization->is_running_condition.wait_for(
                _lock, CACHE_FILE_FLUSH_TIMEOUT,
                [&]() { return !m_worker_synchronization->is_running; });
        }

        LOG_TRACE("Flush worker thread performing final flush");
        m_worker_function(m_ofs, true);
        m_ofs.close();
        {
            std::lock_guard _lock{ m_worker_synchronization->exit_finished_mutex };
            m_worker_synchronization->exit_finished = true;
        }
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
        const bool thread_is_created_in_this_process =
            current_pid == m_worker_synchronization->origin_pid;

        if(!thread_is_created_in_this_process)
        {
            // Child process after fork(): the flush thread does not exist here
            // and the inherited mutex may be locked by a thread that is gone.
            // Skip all mutex/CV/join logic — a plain atomic store is enough.
            LOG_DEBUG("Flush worker was created in different process, skipping join");
            m_worker_synchronization->is_running.store(false, std::memory_order_release);
            return;
        }

        LOG_TRACE("Signaling flush worker to stop");
        {
            std::lock_guard _lock{ m_worker_synchronization->is_running_mutex };
            m_worker_synchronization->is_running = false;
        }
        m_worker_synchronization->is_running_condition.notify_all();

        LOG_TRACE("Waiting for flush worker thread to finish");
        std::unique_lock _exit_lock{ m_worker_synchronization->exit_finished_mutex };
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
