// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/syncobj_notification.h"

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <fstream>
#include <pthread.h>
#include <sys/syscall.h>
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
  static int (*forward_clock)(clockid_t, timespec *) = nullptr;
  forward_clock = saved_.clock_gettime;
  setup_entered_.store(false, std::memory_order_release);
  rocjitsu::libc_passthrough().clock_gettime = +[](clockid_t clock, timespec *time) {
    setup_entered_.store(true, std::memory_order_release);
    return forward_clock(clock, time);
  };
  rocjitsu::SyncobjNotification notification;
  const timespec deadline = deadline_after(std::chrono::seconds(2));
  int result = -1;
  std::thread waiter([&] { result = notification.wait_until(0, &deadline); });
  const auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(1);
  while (!setup_entered_.load(std::memory_order_acquire) &&
         std::chrono::steady_clock::now() < limit)
    std::this_thread::yield();
  const bool entered = setup_entered_.load(std::memory_order_acquire);
  const auto wake_start = std::chrono::steady_clock::now();
  notification.notify_all();
  waiter.join();
  ASSERT_TRUE(entered);
  EXPECT_EQ(result, 0);
  EXPECT_LT(std::chrono::steady_clock::now() - wake_start, std::chrono::milliseconds(500));
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

TEST_F(SyncobjNotificationTest, ThreadExitDuringContendedRegistrationUnwinds) {
  static std::atomic<bool> first_sleep{false}, in_post{false}, release_post{false};
  static int (*forward_wait)(sem_t *, clockid_t, const timespec *) = nullptr;
  static int (*forward_clock)(clockid_t, timespec *) = nullptr;
  static int (*forward_post)(sem_t *) = nullptr;
  first_sleep = false;
  in_post = false;
  release_post = false;
  forward_wait = saved_.sem_clockwait;
  forward_clock = saved_.clock_gettime;
  forward_post = saved_.sem_post;
  if (forward_wait) {
    rocjitsu::libc_passthrough().sem_clockwait =
        +[](sem_t *semaphore, clockid_t clock, const timespec *deadline) {
          first_sleep.store(true, std::memory_order_release);
          return forward_wait(semaphore, clock, deadline);
        };
  } else {
    rocjitsu::libc_passthrough().clock_gettime = +[](clockid_t clock, timespec *time) {
      first_sleep.store(true, std::memory_order_release);
      return forward_clock(clock, time);
    };
  }
  rocjitsu::libc_passthrough().sem_post = +[](sem_t *semaphore) {
    in_post.store(true, std::memory_order_release);
    const auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    while (!release_post.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < limit)
      std::this_thread::yield();
    return forward_post(semaphore);
  };
  struct sigaction action {
  }, old_action{};
  action.sa_handler = +[](int) { pthread_exit(nullptr); };
  sigemptyset(&action.sa_mask);
  ASSERT_EQ(sigaction(SIGUSR1, &action, &old_action), 0);

  const auto await = [](const std::atomic<bool> &flag) {
    const auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (!flag.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < limit)
      std::this_thread::yield();
    return flag.load(std::memory_order_acquire);
  };
  rocjitsu::SyncobjNotification notification;
  const timespec deadline = deadline_after(std::chrono::seconds(4));
  std::thread first([&] { (void)notification.wait_until(0, &deadline); });
  const bool first_entered = await(first_sleep);
  std::thread notifier([&] { notification.notify_all(); });
  const bool mutex_held = await(in_post);

  std::atomic<pid_t> tid{0};
  std::atomic<bool> done{false};
  bool unwound = false;
  bool returned = false;
  const uint32_t expected = notification.load(std::memory_order_acquire);
  std::thread waiter([&] {
    class Cleanup {
    public:
      Cleanup(bool &unwound, std::atomic<bool> &done) : unwound_(unwound), done_(done) {}
      ~Cleanup() {
        unwound_ = true;
        done_.store(true, std::memory_order_release);
      }

    private:
      bool &unwound_;
      std::atomic<bool> &done_;
    } cleanup(unwound, done);
    tid.store(gettid(), std::memory_order_release);
    (void)notification.wait_until(expected, &deadline);
    returned = true;
  });
  bool blocked = false;
  const auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(1);
  while (!blocked && std::chrono::steady_clock::now() < limit) {
    const pid_t waiter_tid = tid.load(std::memory_order_acquire);
    if (waiter_tid > 0) {
      std::ifstream state("/proc/self/task/" + std::to_string(waiter_tid) + "/syscall");
      long number = -1;
      blocked = (state >> number) && number == SYS_futex;
    }
    std::this_thread::yield();
  }
  const int signal_result = blocked ? pthread_kill(waiter.native_handle(), SIGUSR1) : -1;
  const bool exited_while_locked = blocked && await(done);
  release_post.store(true, std::memory_order_release);
  waiter.join();
  notifier.join();
  first.join();
  EXPECT_EQ(sigaction(SIGUSR1, &old_action, nullptr), 0);
  EXPECT_TRUE(first_entered);
  EXPECT_TRUE(mutex_held);
  EXPECT_TRUE(blocked);
  EXPECT_EQ(signal_result, 0);
  EXPECT_TRUE(exited_while_locked);
  EXPECT_TRUE(unwound);
  EXPECT_FALSE(returned);
  notification.notify_all();
}
