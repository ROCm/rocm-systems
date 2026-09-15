// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/host_mapping_lock.h"

#include <gtest/gtest.h>

#include <barrier>
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
