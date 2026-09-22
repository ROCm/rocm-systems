/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
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
