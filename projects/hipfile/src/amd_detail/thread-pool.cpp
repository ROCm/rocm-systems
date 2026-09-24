/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "taskflow/taskflow.hpp"
#include "thread-pool.h"

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <exception>
#include <mutex>
#include <utility>

namespace hipFile {

namespace {

    class TaskflowTaskGroup final : public ITaskGroup {
    public:
        explicit TaskflowTaskGroup(std::shared_ptr<tf::Executor> _executor)
            : executor{std::move(_executor)}, state{std::make_shared<State>()}
        {
        }

        ~TaskflowTaskGroup() override
        {
            cancel();
            wait();
        }

        void run(std::function<void()> task) override
        {
            uint64_t task_generation = 0;
            auto     task_state      = state;

            {
                std::lock_guard<std::mutex> lock{task_state->mutex};

                task_generation = task_state->generation;
                task_state->outstanding++;
            }

            try {
                executor->silent_async([task_state, task_generation, work = std::move(task)]() mutable {
                    Completion            completion{task_state};
                    std::function<void()> local_work;
                    local_work.swap(work);

                    {
                        std::lock_guard<std::mutex> lock{task_state->mutex};
                        if (task_generation != task_state->generation) {
                            return;
                        }
                    }

                    local_work();
                });
            }
            catch (...) {
                finish(task_state);
                throw;
            }
        }

        void cancel() override
        {
            std::lock_guard<std::mutex> lock{state->mutex};

            state->generation++;
        }

        void wait() override
        {
            std::unique_lock<std::mutex> lock{state->mutex};

            state->cv.wait(lock, [this]() { return state->outstanding == 0; });
        }

    private:
        struct State {
            std::mutex              mutex;
            std::condition_variable cv;
            uint64_t                generation  = 0;
            size_t                  outstanding = 0;
        };

        class Completion {
        public:
            explicit Completion(std::shared_ptr<State> _state) : state{std::move(_state)}
            {
            }

            ~Completion()
            {
                finish(state);
            }

            Completion(const Completion &)            = delete;
            Completion &operator=(const Completion &) = delete;

            Completion(Completion &&)            = delete;
            Completion &operator=(Completion &&) = delete;

        private:
            std::shared_ptr<State> state;
        };

        static void finish(const std::shared_ptr<State> &state)
        {
            {
                std::lock_guard<std::mutex> lock{state->mutex};

                state->outstanding--;
            }
            state->cv.notify_all();
        }

        std::shared_ptr<tf::Executor> executor;
        std::shared_ptr<State>        state;
    };

}

struct ThreadPool::Impl {
    explicit Impl(size_t workers) : executor{std::make_shared<tf::Executor>(std::max<size_t>(workers, 1))}
    {
    }

    std::shared_ptr<tf::Executor> executor;
};

ThreadPool::ThreadPool(size_t workers) : impl{std::make_unique<Impl>(workers)}
{
}

ThreadPool::~ThreadPool() = default;

std::unique_ptr<ITaskGroup>
ThreadPool::makeTaskGroup()
{
    return std::make_unique<TaskflowTaskGroup>(impl->executor);
}

}
