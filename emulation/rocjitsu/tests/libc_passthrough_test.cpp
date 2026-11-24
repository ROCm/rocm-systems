// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/libc_passthrough.h"

#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <barrier>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fcntl.h>
#include <thread>

namespace {
std::atomic<unsigned> openat_resolutions{0};
}

extern "C" void *__real_dlsym(void *, const char *);

// Exercise the real legacy libc entry points on modern hosts too.
extern "C" void *__wrap_dlsym(void *handle, const char *name) {
  if (std::strcmp(name, "openat") == 0) {
    openat_resolutions.fetch_add(1, std::memory_order_relaxed);
  }
  if (std::getenv("RJ_TEST_LEGACY_STAT")) {
#if defined(__GLIBC__) && defined(__x86_64__)
    if (std::strcmp(name, "realpath") == 0) {
      if (auto *legacy = dlvsym(handle, name, "GLIBC_2.2.5")) {
        return legacy;
      }
    }
#endif
    for (const char *symbol : {"stat", "lstat", "fstat", "stat64", "lstat64", "fstat64"}) {
      if (std::strcmp(name, symbol) == 0) {
        return nullptr;
      }
    }
  }
  return __real_dlsym(handle, name);
}

TEST(LibcPassthroughTest, StatEntryPointsPreserveMetadataAndErrors) {
  auto &real = rocjitsu::libc_passthrough();
  real.resolve();
  ASSERT_TRUE(real.ready());
  ASSERT_NE(real.stat, nullptr);
  ASSERT_NE(real.lstat, nullptr);
  ASSERT_NE(real.fstat_fn, nullptr);
  ASSERT_NE(real.stat64_fn, nullptr);
  ASSERT_NE(real.lstat64_fn, nullptr);
  ASSERT_NE(real.fstat64_fn, nullptr);

  // /proc/self is a symlink; stat must follow it while lstat must not.
  struct stat path {
  }, descriptor{}, link{};
  ASSERT_EQ(real.stat("/proc/self", &path), 0);
  EXPECT_TRUE(S_ISDIR(path.st_mode));
  ASSERT_EQ(real.lstat("/proc/self", &link), 0);
  EXPECT_TRUE(S_ISLNK(link.st_mode));
  int fd = ::open("/proc/self", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  ASSERT_GE(fd, 0);
  rocjitsu::UniqueDriverFd owned(fd);
  ASSERT_EQ(real.fstat_fn(fd, &descriptor), 0);
  EXPECT_EQ(descriptor.st_ino, path.st_ino);
  EXPECT_EQ(descriptor.st_dev, path.st_dev);
  EXPECT_EQ(descriptor.st_mode, path.st_mode);

  struct stat64 large_path {
  }, large_descriptor{}, large_link{};
  ASSERT_EQ(real.stat64_fn("/proc/self", &large_path), 0);
  ASSERT_EQ(real.lstat64_fn("/proc/self", &large_link), 0);
  ASSERT_EQ(real.fstat64_fn(fd, &large_descriptor), 0);
  EXPECT_EQ(large_path.st_ino, path.st_ino);
  EXPECT_EQ(large_descriptor.st_ino, descriptor.st_ino);
  EXPECT_EQ(large_link.st_ino, link.st_ino);
  EXPECT_TRUE(S_ISDIR(large_path.st_mode));
  EXPECT_TRUE(S_ISLNK(large_link.st_mode));

  for (auto fn : {real.stat, real.lstat}) {
    errno = 0;
    EXPECT_EQ(fn("/proc/self/rocjitsu-nonexistent", &path), -1);
    EXPECT_EQ(errno, ENOENT);
  }
  for (auto fn : {real.stat64_fn, real.lstat64_fn}) {
    errno = 0;
    EXPECT_EQ(fn("/proc/self/rocjitsu-nonexistent", &large_path), -1);
    EXPECT_EQ(errno, ENOENT);
  }
  errno = 0;
  EXPECT_EQ(real.fstat_fn(-1, &descriptor), -1);
  EXPECT_EQ(errno, EBADF);
  errno = 0;
  EXPECT_EQ(real.fstat64_fn(-1, &large_descriptor), -1);
  EXPECT_EQ(errno, EBADF);
}

TEST(LibcPassthroughTest, RealpathAllocatesAndAcceptsCallerBuffer) {
  auto &real = rocjitsu::libc_passthrough();
  real.resolve();
  ASSERT_TRUE(real.ready());
  char buffer[PATH_MAX];
  ASSERT_EQ(real.realpath_fn("/proc/self", buffer), buffer);
  char *allocated = real.realpath_fn("/proc/self", nullptr);
  ASSERT_NE(allocated, nullptr);
  EXPECT_STREQ(allocated, buffer);
  std::free(allocated);
  EXPECT_EQ(real.realpath_fn("/proc/self/rocjitsu-nonexistent", nullptr), nullptr);
  EXPECT_EQ(errno, ENOENT);
}

TEST(LibcPassthroughTest, ConcurrentFirstUseResolvesOneCompleteTable) {
  rocjitsu::LibcPassthrough table;
  const auto before = openat_resolutions.load();
  std::array<std::jthread, 8> workers;
  std::barrier start(static_cast<std::ptrdiff_t>(workers.size()));
  for (auto &worker : workers) {
    worker = std::jthread([&] {
      start.arrive_and_wait();
      table.resolve();
      EXPECT_TRUE(table.ready());
      EXPECT_NE(table.openat, nullptr);
      EXPECT_NE(table.exit_immediately, nullptr);
    });
  }
  for (auto &worker : workers) {
    worker.join();
  }
  EXPECT_EQ(openat_resolutions.load(), before + 1);
}
