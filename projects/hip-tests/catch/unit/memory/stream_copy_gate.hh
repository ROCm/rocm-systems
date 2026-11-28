/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

#include <hip_test_common.hh>

namespace HipTest {
// Keeps a copy behind prior stream work, with a bounded escape if an async API
// incorrectly waits for that work on the calling host thread.
class StreamCopyGate {
 public:
  explicit StreamCopyGate(hipStream_t stream,
                          std::chrono::milliseconds timeout = std::chrono::seconds(2))
      : blocked_(std::make_shared<std::atomic_bool>(true)) {
    // The callback can remain queued after this scope exits. Own its state
    // separately so cleanup need not wait behind unrelated earlier work.
    auto callback_state = std::make_unique<std::shared_ptr<std::atomic_bool>>(blocked_);
    HIP_CHECK(hipStreamAddCallback(
        stream,
        [](hipStream_t, hipError_t, void* opaque) {
          std::unique_ptr<std::shared_ptr<std::atomic_bool>> state(
              static_cast<std::shared_ptr<std::atomic_bool>*>(opaque));
          while ((*state)->load()) std::this_thread::yield();
        },
        callback_state.get(), 0));
    callback_state.release();
    try {
      watchdog_ = std::thread([this, timeout] {
        std::unique_lock lock(mutex_);
        if (!returned_.wait_for(lock, timeout, [this] { return done_; })) {
          timed_out_ = true;
          blocked_->store(false);
        }
      });
    } catch (...) {
      blocked_->store(false);
      throw;
    }
  }
  StreamCopyGate(const StreamCopyGate&) = delete;
  StreamCopyGate& operator=(const StreamCopyGate&) = delete;
  StreamCopyGate(StreamCopyGate&&) = delete;
  StreamCopyGate& operator=(StreamCopyGate&&) = delete;
  ~StreamCopyGate() { release(); }
  void release() {
    {
      std::lock_guard lock(mutex_);
      done_ = true;
    }
    returned_.notify_one();
    blocked_->store(false);
    if (watchdog_.joinable()) {
      watchdog_.join();
    }
  }
  bool timed_out() const { return timed_out_; }

 private:
  std::shared_ptr<std::atomic_bool> blocked_;
  std::mutex mutex_;
  std::condition_variable returned_;
  bool done_ = false;
  std::atomic_bool timed_out_{false};
  std::thread watchdog_;
};
}  // namespace HipTest
