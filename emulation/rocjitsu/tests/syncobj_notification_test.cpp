// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/syncobj_notification.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <pthread.h>
#include <thread>

class SyncobjNotificationTest : public ::testing::Test {
protected:
  void SetUp() override {
    rocjitsu::libc_passthrough().resolve();
    saved_ = rocjitsu::libc_passthrough();
  }
  void TearDown() override { rocjitsu::libc_passthrough() = saved_; }

  static timespec deadline_after(std::chrono::milliseconds duration) {
    timespec deadline{};
    clock_gettime(CLOCK_MONOTONIC, &deadline);
    deadline.tv_nsec += duration.count() * 1'000'000;
    deadline.tv_sec += deadline.tv_nsec / 1'000'000'000;
    deadline.tv_nsec %= 1'000'000'000;
    return deadline;
  }

  static int pause_setup() {
    setup_entered_.store(true, std::memory_order_release);
    const timespec timeout{2, 0};
    return ppoll(nullptr, 0, &timeout, nullptr);
  }

  void check_setup_thread_exit(bool during_initialization, bool fallback) {
    rocjitsu::libc_passthrough() = saved_;
    if (fallback)
      rocjitsu::libc_passthrough().sem_clockwait = nullptr;
    if (during_initialization) {
      rocjitsu::libc_passthrough().sem_init = +[](sem_t *, int, unsigned) { return pause_setup(); };
    } else if (fallback) {
      rocjitsu::libc_passthrough().clock_gettime =
          +[](clockid_t, timespec *) { return pause_setup(); };
    } else {
      rocjitsu::libc_passthrough().sem_clockwait =
          +[](sem_t *, clockid_t, const timespec *) { return pause_setup(); };
    }

    struct sigaction action {
    }, old_action{};
    action.sa_handler = +[](int) { pthread_exit(nullptr); };
    sigemptyset(&action.sa_mask);
    ASSERT_EQ(sigaction(SIGUSR1, &action, &old_action), 0);
    rocjitsu::SyncobjNotification notification;
    const timespec deadline = deadline_after(std::chrono::seconds(3));
    bool unwound = false;
    bool returned = false;
    setup_entered_.store(false, std::memory_order_release);
    std::thread waiter([&] {
      class Cleanup {
      public:
        explicit Cleanup(bool &unwound) : unwound_(unwound) {}
        ~Cleanup() { unwound_ = true; }

      private:
        bool &unwound_;
      } cleanup(unwound);
      (void)notification.wait_until(0, &deadline);
      returned = true;
    });
    const auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (!setup_entered_.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < limit)
      std::this_thread::yield();
    const bool entered = setup_entered_.load(std::memory_order_acquire);
    if (entered) {
      EXPECT_EQ(pthread_kill(waiter.native_handle(), SIGUSR1), 0);
    }
    waiter.join();
    EXPECT_EQ(sigaction(SIGUSR1, &old_action, nullptr), 0);
    rocjitsu::libc_passthrough() = saved_;
    EXPECT_TRUE(entered);
    EXPECT_TRUE(unwound);
    EXPECT_FALSE(returned);
    // No stack waiter may remain registered after forced unwinding.
    notification.notify_all();
  }

  rocjitsu::LibcPassthrough saved_;
  static inline std::atomic<bool> setup_entered_{false};
};

TEST_F(SyncobjNotificationTest, DeadlineSurvivesMissingClockwaitWrapper) {
  rocjitsu::libc_passthrough().sem_clockwait = nullptr;
  rocjitsu::SyncobjNotification notification;
  const timespec deadline = deadline_after(std::chrono::milliseconds(20));
  const auto start = std::chrono::steady_clock::now();
  EXPECT_EQ(notification.wait_until(0, &deadline), -1);
  EXPECT_EQ(errno, ETIMEDOUT);
  EXPECT_GE(std::chrono::steady_clock::now() - start, std::chrono::milliseconds(19));
}

TEST_F(SyncobjNotificationTest, NotificationSurvivesMissingClockwaitWrapper) {
  rocjitsu::libc_passthrough().sem_clockwait = nullptr;
  rocjitsu::SyncobjNotification notification;
  const timespec deadline = deadline_after(std::chrono::seconds(2));
  int result = -1;
  std::thread waiter([&] { result = notification.wait_until(0, &deadline); });
  notification.notify_all();
  waiter.join();
  EXPECT_EQ(result, 0);
}

TEST_F(SyncobjNotificationTest, UnboundedWaitReturnsEintrWithSaRestart) {
  for (bool fallback : {false, true}) {
    rocjitsu::libc_passthrough().sem_clockwait = fallback ? nullptr : saved_.sem_clockwait;
    rocjitsu::SyncobjNotification notification;
    struct sigaction action {
    }, old_action{};
    action.sa_handler = +[](int) {};
    sigemptyset(&action.sa_mask);
    action.sa_flags = SA_RESTART;
    ASSERT_EQ(sigaction(SIGUSR1, &action, &old_action), 0);
    std::atomic<bool> done{false};
    int result = 0;
    int error = 0;
    std::thread waiter([&] {
      result = notification.wait_until(0, nullptr);
      error = errno;
      done.store(true, std::memory_order_release);
    });
    const auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (!done.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < limit) {
      EXPECT_EQ(pthread_kill(waiter.native_handle(), SIGUSR1), 0);
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    // Bound a regression where SA_RESTART incorrectly restarts the wait.
    notification.notify_all();
    waiter.join();
    EXPECT_EQ(sigaction(SIGUSR1, &old_action, nullptr), 0);
    EXPECT_EQ(result, -1);
    EXPECT_EQ(error, EINTR);
  }
}

TEST_F(SyncobjNotificationTest, ThreadExitDuringInitializationCleansUp) {
  check_setup_thread_exit(true, false);
}

TEST_F(SyncobjNotificationTest, ThreadExitBeforeSemaphoreSleepUnregistersWaiter) {
  check_setup_thread_exit(false, false);
}

TEST_F(SyncobjNotificationTest, ThreadExitBeforeFallbackSleepUnregistersWaiter) {
  check_setup_thread_exit(false, true);
}
