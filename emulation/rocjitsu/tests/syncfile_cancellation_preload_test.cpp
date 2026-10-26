// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file syncfile_cancellation_preload_test.cpp
/// @brief Write-interception preload for sync-file publication cancellation tests.

#include "syncfile_cancellation_test.h"

#include <cerrno>
#include <cstdint>
#include <dlfcn.h>
#include <pthread.h>
#include <unistd.h>

static thread_local NotificationWriteMode notification_mode = NotificationWriteMode::Disabled;
static thread_local unsigned notification_writes = 0;

// Interrupt leaves completed exports awaiting notification; Cancel requests
// deferred cancellation at the first notification write.
extern "C" __attribute__((visibility("default"))) void
rj_test_notification_writes(NotificationWriteMode mode) {
  notification_mode = mode;
  notification_writes = 0;
}

extern "C" __attribute__((visibility("default"))) unsigned rj_test_notification_write_count() {
  return notification_writes;
}

extern "C" __attribute__((visibility("default"))) ssize_t write(int fd, const void *data,
                                                                size_t size) {
  static auto next = reinterpret_cast<decltype(&write)>(dlsym(RTLD_NEXT, "write"));
  if (notification_mode != NotificationWriteMode::Disabled && size == sizeof(uint64_t)) {
    ++notification_writes;
    if (notification_mode == NotificationWriteMode::Interrupt) {
      errno = EINTR;
      return -1;
    }
    if (notification_writes == 1)
      pthread_cancel(pthread_self());
  }
  return next(fd, data, size);
}
