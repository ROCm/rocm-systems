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
#include "include/amd_cuid.h"
#include "src/cuid_file.h"
#include "src/cuid_util.h"
#include <atomic>
#include <chrono>
#include <fcntl.h>
#include <gtest/gtest.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

// ============================================================================
// CuidFileLock Tests
// ============================================================================

// Test basic lock acquisition and release
void test_file_lock_basic() {
  const std::string test_file = "/tmp/cuid_test_lock_basic";

  // Test exclusive lock
  {
    CuidFileLock lock(test_file, CuidLockType::EXCLUSIVE);
    EXPECT_FALSE(lock.is_locked());
    EXPECT_TRUE(lock.acquire());
    EXPECT_TRUE(lock.is_locked());
    lock.release();
    EXPECT_FALSE(lock.is_locked());
  }

  // Test shared lock
  {
    CuidFileLock lock(test_file, CuidLockType::SHARED);
    EXPECT_FALSE(lock.is_locked());
    EXPECT_TRUE(lock.acquire());
    EXPECT_TRUE(lock.is_locked());
    // Lock should be released by destructor
  }

  // Clean up
  unlink((test_file + ".lock").c_str());
}

// Test RAII - lock released on scope exit
void test_file_lock_raii() {
  const std::string test_file = "/tmp/cuid_test_lock_raii";

  {
    CuidFileLock lock(test_file, CuidLockType::EXCLUSIVE);
    EXPECT_TRUE(lock.acquire());
    EXPECT_TRUE(lock.is_locked());
    // Scope ends here - destructor should release lock
  }

  // Now we should be able to acquire again immediately
  {
    CuidFileLock lock(test_file, CuidLockType::EXCLUSIVE);
    EXPECT_TRUE(lock.try_acquire()); // Should succeed immediately
    EXPECT_TRUE(lock.is_locked());
  }

  // Clean up
  unlink((test_file + ".lock").c_str());
}

// Test that multiple shared locks can be acquired simultaneously
void test_file_lock_multiple_shared() {
  const std::string test_file = "/tmp/cuid_test_lock_shared";

  CuidFileLock lock1(test_file, CuidLockType::SHARED);
  CuidFileLock lock2(test_file, CuidLockType::SHARED);

  EXPECT_TRUE(lock1.acquire());
  EXPECT_TRUE(
      lock2.try_acquire()); // Should succeed - shared locks are compatible

  EXPECT_TRUE(lock1.is_locked());
  EXPECT_TRUE(lock2.is_locked());

  // Clean up
  unlink((test_file + ".lock").c_str());
}

// Test that exclusive lock blocks other locks (using fork for true
// multi-process test)
void test_file_lock_exclusive_blocks() {
  const std::string test_file = "/tmp/cuid_test_lock_exclusive";

  // Clean up from any previous run
  unlink((test_file + ".lock").c_str());

  pid_t pid = fork();

  if (pid == 0) {
    // Child process: acquire exclusive lock and hold it
    CuidFileLock lock(test_file, CuidLockType::EXCLUSIVE);
    if (!lock.acquire()) {
      _exit(1); // Failed to acquire lock
    }

    // Hold lock for 500ms
    usleep(500000);
    _exit(0);
  } else if (pid > 0) {
    // Parent process: wait a bit, then try to acquire
    usleep(100000); // 100ms - let child acquire first

    CuidFileLock lock(test_file, CuidLockType::EXCLUSIVE);

    // try_acquire should fail because child holds exclusive lock
    EXPECT_FALSE(lock.try_acquire());

    // Wait for child to finish
    int status;
    waitpid(pid, &status, 0);
    EXPECT_TRUE(WIFEXITED(status));
    EXPECT_EQ(WEXITSTATUS(status), 0);

    // Now lock should be available
    EXPECT_TRUE(lock.try_acquire());
  } else {
    FAIL() << "fork() failed";
  }

  // Clean up
  unlink((test_file + ".lock").c_str());
}

// Test timeout functionality
void test_file_lock_timeout() {
  const std::string test_file = "/tmp/cuid_test_lock_timeout";

  // Clean up from any previous run
  unlink((test_file + ".lock").c_str());

  pid_t pid = fork();

  if (pid == 0) {
    // Child process: acquire exclusive lock and hold it for 2 seconds
    CuidFileLock lock(test_file, CuidLockType::EXCLUSIVE);
    if (!lock.acquire()) {
      _exit(1);
    }
    sleep(2);
    _exit(0);
  } else if (pid > 0) {
    // Parent process: wait a bit, then try to acquire with timeout
    usleep(100000); // 100ms - let child acquire first

    CuidFileLock lock(test_file, CuidLockType::EXCLUSIVE);

    // 1 second timeout should fail (child holds for 2 seconds)
    auto start = std::chrono::steady_clock::now();
    bool acquired = lock.acquire_with_timeout(1);
    auto elapsed = std::chrono::steady_clock::now() - start;

    EXPECT_FALSE(acquired);
    // Should have waited approximately 1 second
    EXPECT_GE(
        std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(),
        900);
    EXPECT_LE(
        std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count(),
        1500);

    // Wait for child to finish
    int status;
    waitpid(pid, &status, 0);

    // Now should be able to acquire
    EXPECT_TRUE(lock.acquire_with_timeout(1));
  } else {
    FAIL() << "fork() failed";
  }

  // Clean up
  unlink((test_file + ".lock").c_str());
}

// Test timeout with 0 (non-blocking) and -1 (infinite)
void test_file_lock_timeout_special_cases() {
  const std::string test_file = "/tmp/cuid_test_lock_timeout_special";

  // Clean up
  unlink((test_file + ".lock").c_str());

  // Test timeout=0 (should behave like try_acquire)
  {
    CuidFileLock lock(test_file, CuidLockType::EXCLUSIVE);
    EXPECT_TRUE(lock.acquire_with_timeout(0)); // Should succeed immediately
    EXPECT_TRUE(lock.is_locked());
  }

  // Test timeout=-1 (should behave like acquire - infinite wait)
  {
    CuidFileLock lock(test_file, CuidLockType::EXCLUSIVE);
    EXPECT_TRUE(lock.acquire_with_timeout(-1)); // Should succeed
    EXPECT_TRUE(lock.is_locked());
  }

  // Clean up
  unlink((test_file + ".lock").c_str());
}

TEST(CUIDFileLockTest, BasicLockAcquireRelease) { test_file_lock_basic(); }

TEST(CUIDFileLockTest, RAIIAutoRelease) { test_file_lock_raii(); }

TEST(CUIDFileLockTest, MultipleSharedLocks) {
  test_file_lock_multiple_shared();
}

TEST(CUIDFileLockTest, ExclusiveLockBlocks) {
  test_file_lock_exclusive_blocks();
}

TEST(CUIDFileLockTest, AcquireWithTimeout) { test_file_lock_timeout(); }

TEST(CUIDFileLockTest, TimeoutSpecialCases) {
  test_file_lock_timeout_special_cases();
}

// ============================================================================
// Cuid Utilities Tests
// ============================================================================

// Test that remove_UUIDv8_bits correctly recovers the original raw bits from a
// canned UUIDv8-encoded amdcuid_id_t.
void test_remove_UUIDv8_bits_roundtrip() {
  // Canned UUIDv8 representation (computed from raw bits above)
  amdcuid_id_t id = {{0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0x8C, 0xDE, 0xBC,
                      0x48, 0xD1, 0x59, 0xE2, 0x6A, 0xF3, 0x7B}};

  const uint8_t expected[16] = {0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF,
                                0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xC0};

  uint8_t out[16] = {0};
  CuidUtilities::remove_UUIDv8_bits(&id, out);

  for (int i = 0; i < 16; ++i) {
    EXPECT_EQ(out[i], expected[i]) << "Mismatch at byte " << i;
  }
}

// Test that remove_UUIDv8_bits is a no-op when passed null pointers.
void test_remove_UUIDv8_bits_null_safety() {
  amdcuid_id_t id = {{0}};
  uint8_t out[16] = {0xFF};

  // Neither call should crash or modify out
  CuidUtilities::remove_UUIDv8_bits(nullptr, out);
  EXPECT_EQ(out[0], 0xFF); // unchanged

  CuidUtilities::remove_UUIDv8_bits(&id, nullptr); // should not crash
}

TEST(CUIDUtilitiesTest, RemoveUUIDv8BitsRoundtrip) {
  test_remove_UUIDv8_bits_roundtrip();
}

TEST(CUIDUtilitiesTest, RemoveUUIDv8BitsNullSafety) {
  test_remove_UUIDv8_bits_null_safety();
}
