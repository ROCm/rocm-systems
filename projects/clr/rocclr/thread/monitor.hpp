/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef MONITOR_HPP_
#define MONITOR_HPP_

#include "top.hpp"
#include "utils/flags.hpp"
#include <condition_variable>
#include <mutex>
#include <atomic>
#include <tuple>
#include <utility>
#include "os/os.hpp"

namespace amd {

class alignas(64) Monitor {
 public:
  explicit Monitor() {
    // Relaxed: the Monitor is still under construction and cannot be observed by
    // another thread yet, so no ordering is required. seq_cst (the default)
    // would emit a locked xchg per store, which shows up in hot construction
    // paths (e.g. amd::Event has two Monitors built per object).
    waits_.store(0, std::memory_order_relaxed);  // 0 waiting thread initially
    notifyState_.store(notifyState::notNotified,
                       std::memory_order_relaxed);  // initially not notified
  }

  //! Try to acquire the lock, return true if successful, false if failed.
  bool tryLock() { return mutex_.try_lock(); }

  //! Acquire the lock or suspend the calling thread.
  void lock() { mutex_.lock(); }

  //! Release the lock and wake a single waiting thread if any.
  void unlock() { mutex_.unlock(); }

  // GCC 12+ emits a false -Wstringop-overflow when it inlines atomic ops on
  // class members through multiple call frames and loses size provenance.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstringop-overflow"
#endif

  /*! \brief Give up the lock and go to sleep.
   *
   *  Calling wait() causes the current thread to go to sleep until
   *  another thread calls notify()/notifyAll().
   *
   *  \note The monitor must be owned before calling wait().
   */
  void wait() {
    assert(waits_.load(std::memory_order_acquire) >= 0 && "Error: waits_.load() < 0");
    std::unique_lock lk(mutex_, std::adopt_lock);

    int c = 0;
    while (unlikely(notifyState_.load(std::memory_order_acquire) == notifyState::allNotified)) {
      lk.unlock();
      // NotifyAll() processing already in progress, don't enter now.
      // The new wait() shoule be processed by next notifyAll().
      if (c < maxReadSpinIter_) {
        Os::spinPause();
        c++;
      }
      // and then SMP friendly
      else {
        Os::yield();
      }
      lk.lock();
    }
    waits_.fetch_add(1, std::memory_order_acq_rel);

    lk.unlock();
    // fast path
    c = 0;
    while (c < maxCount_ &&
           (notifyState_.load(std::memory_order_acquire) == notifyState::notNotified)) {
      // First, be SMT friendly
      if (c < maxReadSpinIter_) {
        Os::spinPause();
      }
      // and then SMP friendly
      else {
        Os::yield();
      }
      c++;
    }
    assert(c <= maxCount_ && "Error: c > maxCount_");

    lk.lock();

    if (c == maxCount_) {
      // In case notify() is called between loop and here
      notifyState expextedNotifyState = notifyState::oneNotified;
      if (notifyState_.load(std::memory_order_acquire) != notifyState::allNotified &&
          !notifyState_.compare_exchange_strong(expextedNotifyState, notifyState::notNotified,
                                                std::memory_order_acq_rel,
                                                std::memory_order_acquire)) {
        // Still not notified, so enter slow path
        cv_.wait(lk);  // slow path
        expextedNotifyState = notifyState::oneNotified;
        // To reset notifyState::oneNotified to notifyState::notNotified state if notifyState_ is
        // notifyState::oneNotified.
        // This will happen when notify() is called during cv_.wait(lk). Will do nothing
        // if notifyState_ is notifyState::notNoftifed or notifyState::allNotified.
        notifyState_.compare_exchange_strong(expextedNotifyState, notifyState::notNotified,
                                             std::memory_order_acq_rel, std::memory_order_acquire);
      }
    }
    // the mutex is locked again before exiting...
    lk.release();  // Release the ownership so that the caller should unlock the mutex
    if (waits_.fetch_sub(1, std::memory_order_acq_rel) == 1) {
      // No waiter indicates that notify() or notifyAll() processing has ended
      notifyState_.store(notifyState::notNotified, std::memory_order_release);
    }
  }

  /*! \brief Wake up a single thread waiting on this monitor.
   *
   *  \note The monitor need be owned before calling notify().
   */
  void notify() {
    // If notifyState_ is notifyState::oneNotified or notifyState::allNotified, this will be
    // skipped.
    if (notifyState_.load(std::memory_order_acquire) == notifyState::notNotified &&
        waits_.load(std::memory_order_acquire) > 0) {
      notifyState_.store(notifyState::oneNotified, std::memory_order_release);
      cv_.notify_one();
    }
  }

  /*! \brief Wake up all threads that are waiting on this monitor.
   *
   *  \note The monitor need be owned before calling notifyAll().
   */
  void notifyAll() {
    // If notifyState_ is notifyState::allNotified, this will be skipped.  So notifyAll()
    // can still be called if notify() is just called as notifyAll() covers notify()
    if (notifyState_.load(std::memory_order_acquire) != notifyState::allNotified &&
        waits_.load(std::memory_order_acquire) > 0) {
      // One notification is enough
      notifyState_.store(notifyState::allNotified, std::memory_order_release);
      cv_.notify_all();
    }
  }

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

 private:
  enum class notifyState : uint32_t { notNotified = 0, oneNotified = 1, allNotified = 2 };

  // The members are split across separate cache lines to avoid false sharing.
  // mutex_ is written on every lock()/unlock(); the spin atomics (waits_ and
  // notifyState_) are read in the spin loop of wait() and written by notify*();
  // cv_ is only touched on the slow path. Keeping the three groups on distinct
  // cache lines stops the spin loop from being invalidated by lock/unlock or
  // condition-variable traffic. alignas(64) on the class also rounds sizeof up
  // to a whole number of cache lines, so adjacent Monitors never share a line.

  //! Cache line 0: the mutex (hot, written on every lock/unlock).
  alignas(64) std::mutex mutex_;

  //! Cache line 1: the spin-wait state plus the read-only spin tunables.
  alignas(64) std::atomic<int> waits_;
  std::atomic<notifyState> notifyState_;
  const int maxCount_{55};  //!< Max count of spins in wait()
  const int maxReadSpinIter_{50};

  //! Cache line 2: the condition variable (cold, slow path only).
  alignas(64) std::condition_variable cv_;
};

static_assert(alignof(Monitor) == 64,
              "Monitor must be aligned to a 64-byte cache line so that adjacent "
              "Monitors never share a line (false sharing).");
static_assert(sizeof(Monitor) % 64 == 0,
              "Monitor must occupy a whole number of cache lines so that its hot "
              "members never straddle a line and a following object can never "
              "share its trailing line (false sharing).");

class ScopedLock : StackObject {
 public:
  explicit ScopedLock(Monitor& lock) : lock_(&lock) { lock_->lock(); }

  explicit ScopedLock(Monitor* lock) : lock_(lock) {
    if (lock_) {
      lock_->lock();
    }
  }

  ~ScopedLock() {
    if (lock_) lock_->unlock();
  }

 private:
  Monitor* lock_;
};

}  // namespace amd

#endif /*MONITOR_HPP_*/
