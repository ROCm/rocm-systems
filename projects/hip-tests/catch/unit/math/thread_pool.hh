/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#pragma once

#include <atomic>
#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

class ThreadPool {
 public:
  ThreadPool(size_t thread_count = std::thread::hardware_concurrency())
      : thread_count_(thread_count == 0 ? 1 : thread_count), stop_(false), active_tasks_(0) {
    workers_.reserve(thread_count_);
    for (size_t i = 0; i < thread_count_; ++i) {
      workers_.emplace_back([this] { this->worker_thread(); });
    }
  }

  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      stop_ = true;
    }
    condition_.notify_all();
    for (std::thread& worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

  // Submits a task to the thread pool.
  template <typename T> void Post(T&& task) {
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      if (stop_) {
        return;  // Don't accept new tasks after shutdown
      }
      tasks_.emplace(std::forward<T>(task));
      ++active_tasks_;
    }
    condition_.notify_one();
  }

  // Busy waits for the number of active tasks to reach zero.
  void Wait() const {
    while (active_tasks_.load(std::memory_order_relaxed)) {
    }
  }

  size_t thread_count() const { return thread_count_; }

 private:
  void worker_thread() {
    while (true) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });

        if (stop_ && tasks_.empty()) {
          return;
        }

        if (!tasks_.empty()) {
          task = std::move(tasks_.front());
          tasks_.pop();
        }
      }

      if (task) {
        struct TaskGuard {
          std::atomic<size_t>& counter;
          ~TaskGuard() { --counter; }
        } guard{active_tasks_};

        try {
          task();
        } catch (...) {
          std::terminate();
        }
      }
    }
  }

  const size_t thread_count_;
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> tasks_;

  std::mutex queue_mutex_;
  std::condition_variable condition_;
  bool stop_;

  std::atomic<size_t> active_tasks_;
};

inline ThreadPool thread_pool{};
