// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cerrno>
#include <cstdint>
#include <dlfcn.h>
#include <pthread.h>
#include <unistd.h>

static thread_local int notification_mode = 0;
static thread_local unsigned notification_writes = 0;

// Mode 1 leaves exported, already-completed fences awaiting notification after
// EINTR. Mode 2 requests deferred cancellation at the first retried notification.
extern "C" __attribute__((visibility("default"))) void rj_test_notification_writes(int mode) {
  notification_mode = mode;
  notification_writes = 0;
}

extern "C" __attribute__((visibility("default"))) unsigned rj_test_notification_write_count() {
  return notification_writes;
}

extern "C" __attribute__((visibility("default"))) ssize_t write(int fd, const void *data,
                                                                size_t size) {
  static auto next = reinterpret_cast<decltype(&write)>(dlsym(RTLD_NEXT, "write"));
  if (notification_mode && size == sizeof(uint64_t)) {
    ++notification_writes;
    if (notification_mode == 1) {
      errno = EINTR;
      return -1;
    }
    if (notification_writes == 1)
      pthread_cancel(pthread_self());
  }
  return next(fd, data, size);
}
