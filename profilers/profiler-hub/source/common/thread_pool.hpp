// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <condition_variable>
#include <cstddef>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <stop_token>
#include <thread>
#include <vector>

namespace profiler_hub::common
{

namespace detail
{
struct task_control_block;
}

/**
 * Fixed-size pool of worker threads executing type-erased, cooperatively
 * cancellable tasks from a shared FIFO queue.
 */
class thread_pool
{
public:
    enum class task_state : uint8_t
    {
        pending,
        running,
        completed,
        cancelled,
    };

    /** @brief Callable signature for tasks: polls @p token to stop early. */
    using task_fn = std::function<void(std::stop_token token)>;

    /** @brief Handle to a submitted task, returned by submit(). */
    class task_handle
    {
    public:
        /** @brief Blocks until the task finishes running or is cancelled. */
        void wait() const;

        /**
         * @brief Cancels the task.
         *
         * If the task has not started running yet, it is skipped entirely.
         * If it is already running, a cooperative stop is requested via
         * the task's std::stop_token; the task must poll the token to
         * honor it, otherwise it runs to completion regardless.
         *
         * @return true if the task was pending or running (skipped or stop
         *         requested), false if it had already completed/cancelled.
         */
        [[nodiscard]] bool cancel() const;

    private:
        friend class thread_pool;
        explicit task_handle(std::shared_ptr<detail::task_control_block> block);

        std::shared_ptr<detail::task_control_block> m_block;
    };

    /** @param num_threads Number of worker threads to start. Must be > 0. */
    explicit thread_pool(size_t num_threads);
    ~thread_pool();

    thread_pool(const thread_pool&)            = delete;
    thread_pool& operator=(const thread_pool&) = delete;
    thread_pool(thread_pool&&)                 = delete;
    thread_pool& operator=(thread_pool&&)      = delete;

    /** @brief Queues @p task for execution on the next available worker. */
    [[nodiscard]] task_handle submit(task_fn task);

private:
    void worker_loop(const std::stop_token& pool_stop_token);

    std::vector<std::jthread>                               m_workers;
    std::deque<std::shared_ptr<detail::task_control_block>> m_queue;
    std::mutex                                              m_queue_mutex;
    std::condition_variable_any                             m_queue_cv;
};

}  // namespace profiler_hub::common
