// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/kmd/linux/libc_passthrough.h"
#include "util/intrusive_list.h"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cerrno>
#include <ctime>
#include <limits>
#include <mutex>
#include <poll.h>

namespace rocjitsu {

/// @brief Broadcast state changes through interruptible libc waits.
/// @details Each sleeper owns a stack semaphore, so waking one cannot consume
/// another's notification. Registration precedes the generation check to close
/// the check/sleep race. No application descriptor is allocated or borrowed.
/// Signal-driven thread exit can unwind libc setup/waits and blocked registration
/// or normal unregistration before mutex acquisition. Delivery after acquisition
/// (including the lock call's return path before guard adoption), during list
/// mutation or notification, or during noexcept cleanup is not supported.
class SyncobjNotification {
public:
  uint32_t load(std::memory_order order) const { return generation_.load(order); }

  int wait_until(uint32_t expected, const timespec *deadline) {
    Waiter waiter(*this);
    if (waiter.initialize() != 0)
      return -1;
    if (waiter.register_waiter() != 0)
      return -1;
    const int result = [&] {
      if (generation_.load(std::memory_order_acquire) != expected)
        return 0;

      if (libc_passthrough().sem_clockwait) {
        // Even an unbounded wait uses the timed API: unlike sem_wait, its wait
        // is interruptible with SA_RESTART, matching the other DRM wait paths.
        const timespec unlimited{std::numeric_limits<time_t>::max(), 0};
        return libc_passthrough().sem_clockwait(waiter.semaphore(), CLOCK_MONOTONIC,
                                                deadline ? deadline : &unlimited);
      }

      // Older libc versions lack sem_clockwait. Bounded descriptor-free polling
      // retains monotonic deadlines and EINTR without depending on wall-clock
      // adjustments or raising the minimum libc version for ordinary simulation.
      while (generation_.load(std::memory_order_acquire) == expected) {
        timespec delay{0, 1'000'000};
        if (deadline) {
          timespec now{};
          if (libc_passthrough().clock_gettime(CLOCK_MONOTONIC, &now) != 0)
            return -1;
          const time_t seconds = deadline->tv_sec - now.tv_sec;
          const long nanoseconds = deadline->tv_nsec - now.tv_nsec;
          if (seconds < 0 || (seconds == 0 && nanoseconds <= 0)) {
            errno = ETIMEDOUT;
            return -1;
          }
          if (seconds == 0)
            delay.tv_nsec = std::min(delay.tv_nsec, nanoseconds);
          else if (seconds == 1 && nanoseconds < 0)
            delay.tv_nsec = std::min(delay.tv_nsec, 1'000'000'000 + nanoseconds);
        }
        if (::ppoll(nullptr, 0, &delay, nullptr) < 0)
          return -1;
      }
      return 0;
    }();
    const int saved_errno = errno;
    // A runtime may deliver a deferred signal while reacquiring the mutex.
    // Keep normal unlinking outside noexcept destruction so forced unwinding
    // can still invoke the destructor's cleanup backstop.
    if (waiter.unregister_waiter() != 0)
      return -1;
    errno = saved_errno;
    return result;
  }

  void notify_all() {
    std::lock_guard lock(mutex_);
    generation_.fetch_add(1, std::memory_order_release);
    for (auto iterator = waiters_.begin(); iterator != waiters_.end(); ++iterator)
      static_cast<Waiter *>(iterator.node_pointer())->notify();
  }

private:
  class Waiter : public util::IListNode<Waiter> {
  public:
    explicit Waiter(SyncobjNotification &owner) : owner_(owner) {}
    ~Waiter() {
      const int saved_errno = errno;
      if (registered_) {
        std::lock_guard lock(owner_.mutex_);
        owner_.waiters_.erase(util::IntrusiveList<Waiter>::Iterator(*this));
      }
      if (initialized_)
        libc_passthrough().sem_destroy(&semaphore_);
      errno = saved_errno;
    }
    Waiter(const Waiter &) = delete;
    Waiter &operator=(const Waiter &) = delete;

    int initialize() {
      const int result = libc_passthrough().sem_init(&semaphore_, 0, 0);
      initialized_ = result == 0;
      return result;
    }
    int register_waiter() {
      // std::mutex::lock reaches a noexcept libc declaration. Keep a signal's
      // forced unwind able to reach the still-unregistered waiter's destructor.
      const int result = libc_passthrough().pthread_mutex_lock(owner_.mutex_.native_handle());
      if (result != 0) {
        errno = result;
        return -1;
      }
      std::lock_guard lock(owner_.mutex_, std::adopt_lock);
      owner_.waiters_.push_front(*this);
      registered_ = true;
      return 0;
    }
    int unregister_waiter() {
      const int result = libc_passthrough().pthread_mutex_lock(owner_.mutex_.native_handle());
      if (result != 0) {
        errno = result;
        return -1;
      }
      std::lock_guard lock(owner_.mutex_, std::adopt_lock);
      owner_.waiters_.erase(util::IntrusiveList<Waiter>::Iterator(*this));
      registered_ = false;
      return 0;
    }
    sem_t *semaphore() { return &semaphore_; }
    void notify() {
      if (notified_)
        return;
      const int result = libc_passthrough().sem_post(&semaphore_);
      assert(result == 0);
      (void)result;
      notified_ = true;
    }

  private:
    SyncobjNotification &owner_;
    sem_t semaphore_{};
    bool initialized_ = false;
    bool registered_ = false;
    bool notified_ = false;
  };

  std::atomic<uint32_t> generation_{0};
  std::mutex mutex_;
  util::IntrusiveList<Waiter> waiters_;
};

} // namespace rocjitsu
