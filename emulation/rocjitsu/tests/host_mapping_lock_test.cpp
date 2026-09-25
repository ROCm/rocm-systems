// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/host_mapping_lock.h"

#include <gtest/gtest.h>

#include <atomic>
#include <barrier>
#include <chrono>
#include <future>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

TEST(HostMappingLockTest, ChildResetsLockHeldByVanishedThread) {
  std::barrier held(2);
  std::barrier release(2);
  std::jthread owner([&] {
    util::ObservableSharedMutex::ExclusiveGuard lock =
        rocjitsu::host_mapping_lock().lock_exclusive();
    held.arrive_and_wait();
    release.arrive_and_wait();
  });
  held.arrive_and_wait();
  const pid_t child = fork();
  if (child == 0) {
    alarm(5);
    rocjitsu::reset_host_mapping_lock_after_fork();
    {
      util::ObservableSharedMutex::ExclusiveGuard lock =
          rocjitsu::host_mapping_lock().lock_exclusive();
    }
    _exit(0);
  }
  // Releasing the parent's lock cannot release the child's inherited copy.
  release.arrive_and_wait();
  owner.join();
  ASSERT_GE(child, 0);
  int status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFEXITED(status)) << "child status: " << status;
  EXPECT_EQ(WEXITSTATUS(status), 0);
  // The child's reset must also leave the parent's lock usable.
  util::ObservableSharedMutex::ExclusiveGuard lock = rocjitsu::host_mapping_lock().lock_exclusive();
}

TEST(HostMappingLockTest, BatchKeepsMappingStableAndAmortizesNestedAccesses) {
  using namespace std::chrono_literals;

  // Isolate the test so an implementation bug that deadlocks a nested access
  // becomes a bounded failure rather than hanging the whole suite.
  const pid_t child = fork();
  ASSERT_GE(child, 0);
  if (child == 0) {
    alarm(10);
    rocjitsu::reset_host_mapping_lock_after_fork();

    std::atomic<bool> writer_acquired{false};
    std::thread writer;
    {
      const rocjitsu::HostMappingBatchGuard outer_batch;
      const rocjitsu::HostMappingBatchGuard inner_batch;
      writer = std::thread([&] {
        auto lock = rocjitsu::host_mapping_lock().lock_exclusive();
        writer_acquired.store(true, std::memory_order_release);
      });

      const auto deadline = std::chrono::steady_clock::now() + 5s;
      while (rocjitsu::host_mapping_lock().blocked_writers() == 0) {
        if (std::chrono::steady_clock::now() >= deadline)
          _exit(1);
        std::this_thread::yield();
      }

      auto access = rocjitsu::lock_host_mapping_for_access();
      if (writer_acquired.load(std::memory_order_acquire))
        _exit(2);
    }
    writer.join();
    if (!writer_acquired.load(std::memory_order_acquire))
      _exit(3);

    auto exclusive = rocjitsu::host_mapping_lock().lock_exclusive();
    auto ordinary_access = std::async(std::launch::async, [] {
      auto access = rocjitsu::lock_host_mapping_for_access();
      return true;
    });
    if (ordinary_access.wait_for(100ms) != std::future_status::timeout)
      _exit(4);
    exclusive.unlock();
    if (ordinary_access.wait_for(2s) != std::future_status::ready || !ordinary_access.get())
      _exit(5);
    _exit(0);
  }

  int status = 0;
  ASSERT_EQ(waitpid(child, &status, 0), child);
  ASSERT_TRUE(WIFEXITED(status)) << "child status: " << status;
  EXPECT_EQ(WEXITSTATUS(status), 0);
}
