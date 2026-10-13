// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/kmd/linux/libc_passthrough.h"

#include <gtest/gtest.h>

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>

extern "C" void *__real_dlsym(void *, const char *);

// Exercise the real legacy libc entry points on modern hosts too. Only hide
// the six symbols absent from glibc 2.28; do not mock filesystem operations.
extern "C" void *__wrap_dlsym(void *handle, const char *name) {
  if (std::getenv("RJ_TEST_LEGACY_STAT")) {
    for (const char *symbol : {"stat", "lstat", "fstat", "stat64", "lstat64", "fstat64"})
      if (std::strcmp(name, symbol) == 0)
        return nullptr;
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
