/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "context.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <thread>

namespace hipFile {

/// @brief Manage a set of tasks on a thread pool
/// Task groups manage a set of tasks. Multiple task groups can share the
/// same thread pool.
class ITaskGroup {
public:
    virtual ~ITaskGroup() = default;

    /// @brief Run a task function on the thread pool
    ///
    /// @param task Task function to run
    virtual void run(std::function<void()> task) = 0;
    /// @brief Cancel all tasks in task group
    virtual void cancel() = 0;
    /// @brief Wait until no tasks are outstanding in task group
    virtual void wait() = 0;
};

class IThreadPool {
public:
    virtual ~IThreadPool() = default;

    /// @brief Construct a task group on the thread pool
    virtual std::unique_ptr<ITaskGroup> makeTaskGroup() = 0;
};

class ThreadPool : public IThreadPool {
public:
    /// @brief Construct a threadpool
    ///
    /// @param workers Number of worker threads
    explicit ThreadPool(size_t workers = std::thread::hardware_concurrency());
    ~ThreadPool() override;

    ThreadPool(const ThreadPool &)            = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;

    ThreadPool(ThreadPool &&)            = delete;
    ThreadPool &operator=(ThreadPool &&) = delete;

    /// @brief Construct a task group on the threadpool
    std::unique_ptr<ITaskGroup> makeTaskGroup() override;

private:
    struct Impl;

    std::unique_ptr<Impl> impl;
};

HIPFILE_CONTEXT_DEFAULT_IMPL(IThreadPool, ThreadPool);

}
