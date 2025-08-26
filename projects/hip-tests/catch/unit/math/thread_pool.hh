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
#include <thread>

#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>

// This is a simple wrapper around boost::asio::thread_pool that keeps track of the number of
// currently active tasks using an atomic counter.
class ThreadPool {
 public:
  ThreadPool(size_t thread_count = std::thread::hardware_concurrency())
      : thread_count_(thread_count) {}

  ~ThreadPool() { thread_pool_.join(); }

  // Submits a task to the thread pool and increments the number of active tasks. The task is
  // wrapped in a lambda that decrements the number of active tasks upon completion.
  template <typename T> void Post(T&& task) {
    ++active_tasks_;
    auto&& task_wrapper = [task, this] {
      task();
      --active_tasks_;
    };
    boost::asio::post(thread_pool_, task_wrapper);
  }

  // Busy waits for the number of active tasks to reach zero.
  void Wait() const {
    while (active_tasks_.load(std::memory_order_relaxed))
      ;
  }

  size_t thread_count() const { return thread_count_; }

 private:
  const size_t thread_count_;
  boost::asio::thread_pool thread_pool_{thread_count_};
  std::atomic<size_t> active_tasks_;
};

inline ThreadPool thread_pool{};
