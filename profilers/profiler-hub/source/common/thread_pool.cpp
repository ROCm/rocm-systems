// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "thread_pool.hpp"

#include "debug.hpp"

#include <exception>
#include <stdexcept>
#include <utility>

namespace profiler_hub::common
{

namespace detail
{
struct task_control_block
{
    thread_pool::task_fn fn;
    std::stop_source     stop_src;

    mutable std::mutex              mutex;
    mutable std::condition_variable cv;
    thread_pool::task_state         state{ thread_pool::task_state::pending };
};
}  // namespace detail

thread_pool::task_handle::task_handle(std::shared_ptr<detail::task_control_block> block)
: m_block{ std::move(block) }
{}

void
thread_pool::task_handle::wait() const
{
    std::unique_lock lock{ m_block->mutex };
    m_block->cv.wait(lock, [this] {
        return m_block->state == task_state::completed ||
               m_block->state == task_state::cancelled;
    });
}

bool
thread_pool::task_handle::cancel() const
{
    std::scoped_lock lock{ m_block->mutex };
    switch(m_block->state)
    {
        case task_state::pending: m_block->state = task_state::cancelled; break;
        case task_state::running: m_block->stop_src.request_stop(); break;
        case task_state::completed:
        case task_state::cancelled: return false;
    }
    m_block->cv.notify_all();
    return true;
}

thread_pool::task_state
thread_pool::task_handle::state() const
{
    std::scoped_lock lock{ m_block->mutex };
    return m_block->state;
}

thread_pool::thread_pool(size_t                num_threads,
                         std::function<void()> on_thread_start,
                         std::function<void()> on_thread_stop)
: m_on_thread_start{ std::move(on_thread_start) }
, m_on_thread_stop{ std::move(on_thread_stop) }
{
    if(num_threads == 0)
    {
        throw std::invalid_argument("thread_pool requires at least one worker thread");
    }

    m_workers.reserve(num_threads);
    for(size_t i = 0; i < num_threads; ++i)
    {
        m_workers.emplace_back(
            [this](const std::stop_token& token) { worker_loop(token); });
    }
}

thread_pool::~thread_pool()
{
    for(auto& worker : m_workers)
    {
        worker.request_stop();
    }
    m_queue_cv.notify_all();
}

thread_pool::task_handle
thread_pool::submit(task_fn task)
{
    auto block = std::make_shared<detail::task_control_block>();
    block->fn  = std::move(task);

    [[maybe_unused]] size_t queue_size = 0;
    {
        std::scoped_lock lock{ m_queue_mutex };
        m_queue.push_back(block);
        queue_size = m_queue.size();
    }
    LOG_DEBUG("thread_pool: task submitted (queue size {})", queue_size);
    m_queue_cv.notify_one();

    return task_handle{ block };
}

void
thread_pool::worker_loop(const std::stop_token& pool_stop_token)
{
    if(m_on_thread_start) m_on_thread_start();

    while(!pool_stop_token.stop_requested())
    {
        std::shared_ptr<detail::task_control_block> block;

        {
            std::unique_lock lock{ m_queue_mutex };
            m_queue_cv.wait(lock, pool_stop_token, [this] { return !m_queue.empty(); });

            if(m_queue.empty()) continue;

            block = std::move(m_queue.front());
            m_queue.pop_front();
        }

        std::stop_token task_token = block->stop_src.get_token();
        {
            std::scoped_lock task_lock{ block->mutex };
            if(block->state == task_state::cancelled) continue;
            block->state = task_state::running;
        }

        LOG_DEBUG("thread_pool: worker picked up task for execution");
        try
        {
            block->fn(task_token);
        } catch(const std::exception& e)
        {
            LOG_ERROR("thread_pool task threw: {}", e.what());
        } catch(...)
        {
            LOG_ERROR("thread_pool task threw a non-std::exception");
        }

        {
            std::scoped_lock task_lock{ block->mutex };
            block->state = task_state::completed;
        }
        block->cv.notify_all();
    }

    if(m_on_thread_stop) m_on_thread_stop();
}

}  // namespace profiler_hub::common
