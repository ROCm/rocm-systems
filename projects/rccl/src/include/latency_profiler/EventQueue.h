/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <condition_variable>
#include <deque>

namespace latency_profiler {

// A multi-producer, single consumer queue.
// This queue is designed to be used in scenarios where multiple producers
// are pushing items into the queue, and a single consumer is waiting for
// items to be available.

template <class Element>
class EventQueue {
 private:
  std::deque<std::unique_ptr<Element>> queue_;
  std::condition_variable cv_;
  mutable std::mutex mutex_;

 public:
  void push(std::unique_ptr<Element> item) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      queue_.push_back(std::move(item));
    }
    cv_.notify_one();
  }

  std::unique_ptr<Element> waitPop() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (queue_.empty()) {
      cv_.wait(lock, [this] { return !queue_.empty(); });
    }
    std::unique_ptr<Element> item = std::move(queue_.front());
    queue_.pop_front();

    return item;
  }
};
} // namespace latency_profiler
