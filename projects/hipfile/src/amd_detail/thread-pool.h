/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <functional>
#include <memory>

namespace hipFile {

class ITaskGroup {
public:
    virtual ~ITaskGroup() = default;

    virtual void run(std::function<void()> work) = 0;
    virtual void cancel()                        = 0;
    virtual void wait()                          = 0;
};

class IThreadPool {
public:
    virtual ~IThreadPool() = default;

    virtual std::unique_ptr<ITaskGroup> makeTaskGroup() = 0;

    virtual std::size_t threadCount() const noexcept = 0;
};

class ThreadPool : public IThreadPool {
public:
    ThreadPool();
    ~ThreadPool() noexcept override;

    ThreadPool(const ThreadPool &)            = delete;
    ThreadPool &operator=(const ThreadPool &) = delete;
    ThreadPool(ThreadPool &&)                 = delete;
    ThreadPool &operator=(ThreadPool &&)      = delete;

    std::unique_ptr<ITaskGroup> makeTaskGroup() override;

    std::size_t threadCount() const noexcept override;

private:
    struct ThreadPoolStorage;
    class TaskGroup;

    std::shared_ptr<ThreadPoolStorage> storage;
};

}
