/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "unit/file_lock_test.h"

#include <gtest/gtest.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>

#include "src/cuid_file.h"

// ---------------------------------------------------------------------------
// TestFileLockBasic
// ---------------------------------------------------------------------------

TestFileLockBasic::TestFileLockBasic() {
  SetTitle("File Lock — Basic Acquire/Release");
  SetDescription(
      "Verify exclusive and shared locks can be acquired and "
      "released, and that RAII destructor releases the lock.");
}

void TestFileLockBasic::SetUp() {}

void TestFileLockBasic::Run() {
  const std::string test_file = "/tmp/cuid_test_lock_basic";

  {
    CuidFileLock lock(test_file, CuidLockType::EXCLUSIVE);
    EXPECT_FALSE(lock.is_locked());
    EXPECT_TRUE(lock.acquire());
    EXPECT_TRUE(lock.is_locked());
    lock.release();
    EXPECT_FALSE(lock.is_locked());
  }

  {
    CuidFileLock lock(test_file, CuidLockType::SHARED);
    EXPECT_FALSE(lock.is_locked());
    EXPECT_TRUE(lock.acquire());
    EXPECT_TRUE(lock.is_locked());
  }

  unlink((test_file + ".lock").c_str());
}

// ---------------------------------------------------------------------------
// TestFileLockRAII
// ---------------------------------------------------------------------------

TestFileLockRAII::TestFileLockRAII() {
  SetTitle("File Lock — RAII Auto-Release");
  SetDescription(
      "Verify that the destructor releases the lock so a subsequent "
      "try_acquire on the same file succeeds immediately.");
}

void TestFileLockRAII::SetUp() {}

void TestFileLockRAII::Run() {
  const std::string test_file = "/tmp/cuid_test_lock_raii";

  {
    CuidFileLock lock(test_file, CuidLockType::EXCLUSIVE);
    EXPECT_TRUE(lock.acquire());
    EXPECT_TRUE(lock.is_locked());
  }

  {
    CuidFileLock lock(test_file, CuidLockType::EXCLUSIVE);
    EXPECT_TRUE(lock.try_acquire());
    EXPECT_TRUE(lock.is_locked());
  }

  unlink((test_file + ".lock").c_str());
}

// ---------------------------------------------------------------------------
// TestFileLockMultipleShared
// ---------------------------------------------------------------------------

TestFileLockMultipleShared::TestFileLockMultipleShared() {
  SetTitle("File Lock — Multiple Shared Locks");
  SetDescription("Verify that two shared locks on the same file can be held simultaneously.");
}

void TestFileLockMultipleShared::SetUp() {}

void TestFileLockMultipleShared::Run() {
  const std::string test_file = "/tmp/cuid_test_lock_shared";

  {
    CuidFileLock lock1(test_file, CuidLockType::SHARED);
    CuidFileLock lock2(test_file, CuidLockType::SHARED);

    EXPECT_TRUE(lock1.acquire());
    EXPECT_TRUE(lock2.try_acquire());
    EXPECT_TRUE(lock1.is_locked());
    EXPECT_TRUE(lock2.is_locked());
  }

  unlink((test_file + ".lock").c_str());
}

// ---------------------------------------------------------------------------
// TestFileLockExclusiveBlocks
// ---------------------------------------------------------------------------

TestFileLockExclusiveBlocks::TestFileLockExclusiveBlocks() {
  SetTitle("File Lock — Exclusive Lock Blocks");
  SetDescription(
      "Verify that a try_acquire fails while another process holds an "
      "exclusive lock, and succeeds once that process exits.");
}

void TestFileLockExclusiveBlocks::SetUp() {}

void TestFileLockExclusiveBlocks::Run() {
  const std::string test_file = "/tmp/cuid_test_lock_exclusive";
  unlink((test_file + ".lock").c_str());

  pid_t pid = fork();
  if (pid == 0) {
    CuidFileLock lock(test_file, CuidLockType::EXCLUSIVE);
    if (!lock.acquire()) {
      _exit(1);
    }
    usleep(500000);
    _exit(0);
  } else if (pid > 0) {
    usleep(100000);

    CuidFileLock lock(test_file, CuidLockType::EXCLUSIVE);
    EXPECT_FALSE(lock.try_acquire());

    int status;
    waitpid(pid, &status, 0);
    EXPECT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0);

    EXPECT_TRUE(lock.try_acquire());
  } else {
    FAIL() << "fork() failed";
  }

  unlink((test_file + ".lock").c_str());
}

// ---------------------------------------------------------------------------
// TestFileLockTimeout
// ---------------------------------------------------------------------------

TestFileLockTimeout::TestFileLockTimeout() {
  SetTitle("File Lock — Acquire With Timeout");
  SetDescription(
      "Verify that acquire_with_timeout returns false after approximately the "
      "requested timeout when another process holds an exclusive lock.");
}

void TestFileLockTimeout::SetUp() {}

void TestFileLockTimeout::Run() {
  const std::string test_file = "/tmp/cuid_test_lock_timeout";
  unlink((test_file + ".lock").c_str());

  pid_t pid = fork();
  if (pid == 0) {
    CuidFileLock lock(test_file, CuidLockType::EXCLUSIVE);
    if (!lock.acquire()) {
      _exit(1);
    }
    sleep(2);
    _exit(0);
  } else if (pid > 0) {
    usleep(100000);

    CuidFileLock lock(test_file, CuidLockType::EXCLUSIVE);

    auto start = std::chrono::steady_clock::now();
    bool acquired = lock.acquire_with_timeout(1);
    auto elapsed = std::chrono::steady_clock::now() - start;
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();

    EXPECT_FALSE(acquired);
    EXPECT_GE(elapsed_ms, 900);
    EXPECT_LE(elapsed_ms, 1500);

    int status;
    waitpid(pid, &status, 0);
    EXPECT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0);

    EXPECT_TRUE(lock.acquire_with_timeout(1));
  } else {
    FAIL() << "fork() failed";
  }

  unlink((test_file + ".lock").c_str());
}

// ---------------------------------------------------------------------------
// TestFileLockTimeoutSpecialCases
// ---------------------------------------------------------------------------

TestFileLockTimeoutSpecialCases::TestFileLockTimeoutSpecialCases() {
  SetTitle("File Lock — Timeout Special Cases");
  SetDescription(
      "Verify that timeout=0 behaves like try_acquire and timeout=-1 behaves "
      "like an infinite-wait acquire.");
}

void TestFileLockTimeoutSpecialCases::SetUp() {}

void TestFileLockTimeoutSpecialCases::Run() {
  const std::string test_file = "/tmp/cuid_test_lock_timeout_special";
  unlink((test_file + ".lock").c_str());

  {
    CuidFileLock lock(test_file, CuidLockType::EXCLUSIVE);
    EXPECT_TRUE(lock.acquire_with_timeout(0));
    EXPECT_TRUE(lock.is_locked());
  }

  {
    CuidFileLock lock(test_file, CuidLockType::EXCLUSIVE);
    EXPECT_TRUE(lock.acquire_with_timeout(-1));
    EXPECT_TRUE(lock.is_locked());
  }

  unlink((test_file + ".lock").c_str());
}
