/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "thread-pool.h"

#include <memory>
#include <oneapi/tbb/task_arena.h>
#include <oneapi/tbb/task_group.h>
#include <stdexcept>
#include <utility>

namespace hipFile {

struct ThreadPool::ThreadPoolStorage {
    tbb::task_arena arena;
};

class ThreadPool::TaskGroup : public ITaskGroup {
public:
    explicit TaskGroup(std::shared_ptr<ThreadPoolStorage> _storage) : storage{std::move(_storage)}
    {
    }

    ~TaskGroup() noexcept override
    {
        try {
            cancel_impl();
            wait_impl();
        }
        catch (...) {
            // Explicit wait() preserves task failures. Destructors cannot report them.
        }
    }

    void run(std::function<void()> work) override
    {
        if (!work) {
            throw std::invalid_argument("Task group work item cannot be empty");
        }

        storage->arena.execute([this, task = std::move(work)]() mutable { tasks.run(std::move(task)); });
    }

    void cancel() override
    {
        cancel_impl();
    }

    void wait() override
    {
        wait_impl();
    }

private:
    void cancel_impl()
    {
        storage->arena.execute([this]() { tasks.cancel(); });
    }

    void wait_impl()
    {
        storage->arena.execute([this]() { tasks.wait(); });
    }

    std::shared_ptr<ThreadPoolStorage> storage;
    tbb::task_group                    tasks;
};

ThreadPool::ThreadPool() : storage{std::make_shared<ThreadPoolStorage>()}
{
}

ThreadPool::~ThreadPool() noexcept = default;

std::unique_ptr<ITaskGroup>
ThreadPool::makeTaskGroup()
{
    return std::make_unique<TaskGroup>(storage);
}

std::size_t
ThreadPool::threadCount() const noexcept
{
    return static_cast<std::size_t>(storage->arena.max_concurrency());
}

}
