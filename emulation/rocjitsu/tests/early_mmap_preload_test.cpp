// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file early_mmap_preload_test.cpp
/// @brief Exercise libc calls made by preload constructors before rocjitsu initializes.

#include <cerrno>
#include <cstdlib>
#include <dlfcn.h>
#include <fcntl.h>
#include <initializer_list>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace {

template <size_t N> [[noreturn]] void fail(const char (&message)[N]) {
  ssize_t ignored = write(STDERR_FILENO, message, N - 1);
  static_cast<void>(ignored);
  _exit(EXIT_FAILURE);
}

} // namespace

__attribute__((constructor)) static void early_mmap_constructor() {
  // rocprofiler-sdk canonicalizes its library path in a preload constructor.
  // Its legacy stat calls must reach libc even before the eager alias table
  // has been initialized. Do not let successful ordinary stat hide this path.
  const int fd = static_cast<int>(syscall(SYS_openat, AT_FDCWD, "/dev/null", O_RDONLY, 0));
  if (fd < 0)
    fail("early preload: open failed\n");
  struct stat st{};
  struct stat64 st64{};
  auto check = [](int rc, mode_t mode) {
    if (rc != 0 || !S_ISCHR(mode))
      fail("early preload: stat alias failed before interposer initialization\n");
  };
  for (const char *symbol : {"stat64", "lstat64"}) {
    auto fn = reinterpret_cast<int (*)(const char *, struct stat64 *)>(dlsym(RTLD_DEFAULT, symbol));
    if (!fn)
      fail("early preload: missing stat64 symbol\n");
    const int rc = fn("/dev/null", &st64);
    check(rc, st64.st_mode);
  }
  auto fstat64_fn = reinterpret_cast<int (*)(int, struct stat64 *)>(dlsym(RTLD_DEFAULT, "fstat64"));
  if (!fstat64_fn)
    fail("early preload: missing fstat64 symbol\n");
  const int fd_rc = fstat64_fn(fd, &st64);
  check(fd_rc, st64.st_mode);
#if defined(__GLIBC__) && defined(__x86_64__)
  // glibc's legacy x86-64 stat version used by the exported __xstat family.
  constexpr int version = 1;
  for (const char *symbol : {"__xstat", "__lxstat"}) {
    auto fn =
        reinterpret_cast<int (*)(int, const char *, struct stat *)>(dlsym(RTLD_DEFAULT, symbol));
    if (!fn)
      fail("early preload: missing xstat symbol\n");
    const int rc = fn(version, "/dev/null", &st);
    check(rc, st.st_mode);
  }
  for (const char *symbol : {"__xstat64", "__lxstat64"}) {
    auto fn =
        reinterpret_cast<int (*)(int, const char *, struct stat64 *)>(dlsym(RTLD_DEFAULT, symbol));
    if (!fn)
      fail("early preload: missing xstat64 symbol\n");
    const int rc = fn(version, "/dev/null", &st64);
    check(rc, st64.st_mode);
  }
  auto fxstat_fn =
      reinterpret_cast<int (*)(int, int, struct stat *)>(dlsym(RTLD_DEFAULT, "__fxstat"));
  auto fxstat64_fn =
      reinterpret_cast<int (*)(int, int, struct stat64 *)>(dlsym(RTLD_DEFAULT, "__fxstat64"));
  if (!fxstat_fn || !fxstat64_fn)
    fail("early preload: missing fxstat symbol\n");
  const int legacy_fd_rc = fxstat_fn(version, fd, &st);
  check(legacy_fd_rc, st.st_mode);
  const int legacy_fd64_rc = fxstat64_fn(version, fd, &st64);
  check(legacy_fd64_rc, st64.st_mode);
#endif
  syscall(SYS_close, fd);
  errno = 0;
  if (mmap(nullptr, 0, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0) != MAP_FAILED ||
      errno != EINVAL)
    fail("early mmap preload: zero-length mmap did not fail with EINVAL\n");

  void *ptr = mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (ptr == MAP_FAILED)
    fail("early mmap preload: mmap failed\n");

  static_cast<char *>(ptr)[0] = 7;

  errno = 0;
  if (munmap(ptr, 0) == 0 || errno != EINVAL)
    fail("early mmap preload: zero-length munmap did not fail with EINVAL\n");

  if (munmap(ptr, 4096) != 0)
    fail("early mmap preload: munmap failed\n");
}
