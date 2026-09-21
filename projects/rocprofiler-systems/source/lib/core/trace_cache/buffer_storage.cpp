// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "buffer_storage.hpp"

#include "logger/debug.hpp"

#include <condition_variable>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <stop_token>
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

    m_flushing_thread = std::jthread{ [this](std::stop_token stoken) {
        LOG_TRACE("Flush worker thread started for pid={}",
                  m_worker_synchronization->origin_pid);

        // Local cv/mutex pair: kept off the worker_synchronization struct so
        // the destructor runs on this thread's stack frame, never during TLS
        // teardown of a foreign thread (TSan DTLS_Destroy deadlock).
        std::condition_variable_any cv;
        std::mutex                  mu;

        while(!stoken.stop_requested())
        {
            m_worker_function(m_ofs, false);
            std::unique_lock _lock{ mu };
            cv.wait_for(_lock, stoken, CACHE_FILE_FLUSH_TIMEOUT,
                        [&]() { return stoken.stop_requested(); });
        }

        LOG_TRACE("Flush worker thread performing final flush");
        m_worker_function(m_ofs, true);
        m_ofs.close();
        m_worker_synchronization->is_running = false;
        m_worker_synchronization->exit_finished = true;
        LOG_TRACE("Flush worker thread exiting");
    } };

    m_worker_synchronization->is_running = true;

    LOG_DEBUG("Flush worker started successfully for pid={}", current_pid);
}

void
flush_worker_t::stop(const pid_t& current_pid)
{
    LOG_DEBUG("Stopping flush worker for pid={}", current_pid);

    const bool flushing_thread_exist = m_flushing_thread.joinable();
    const bool worker_is_running =
        m_worker_synchronization != nullptr && m_worker_synchronization->is_running;

    if(!flushing_thread_exist || !worker_is_running)
    {
        LOG_TRACE("Flush worker not running or thread doesn't exist, nothing to stop");
        return;
    }

    const bool thread_is_created_in_this_process =
        current_pid == m_worker_synchronization->origin_pid;

    if(!thread_is_created_in_this_process)
    {
        // Child process after fork(): the flush thread does not exist in this
        // address space. Calling detach() or join() would invoke
        // pthread_detach / pthread_join on a thread id that is invalid in the
        // child (UB, and ThreadSanitizer CHECK-fails inside its thread
        // registry). Re-construct the jthread in place to drop the inherited
        // handle without running ~jthread, so no pthread call is made and the
        // member is left non-joinable.
        LOG_DEBUG("Flush worker was created in different process, skipping join");
        m_worker_synchronization->is_running.store(false, std::memory_order_release);
        std::construct_at(&m_flushing_thread);
        return;
    }

    LOG_TRACE("Signaling flush worker to stop");
    m_flushing_thread.request_stop();

    LOG_TRACE("Waiting for flush worker thread to finish");
    m_flushing_thread.join();

    LOG_DEBUG("Flush worker stopped for pid={}", current_pid);
}

}  // namespace trace_cache
}  // namespace rocprofsys
