// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cerrno>
#include <dlfcn.h>
#include <pthread.h>
#include <sys/stat.h>

namespace {
thread_local const char *socket_path = nullptr;
thread_local unsigned remaining_attempts = 0;
thread_local bool injected = false;
} // namespace

extern "C" __attribute__((visibility("default"))) void
rj_test_fail_daemon_thread(const char *path, unsigned attempt) {
  socket_path = path;
  remaining_attempts = attempt;
  injected = false;
}

extern "C" __attribute__((visibility("default"))) bool rj_test_daemon_thread_failure_injected() {
  return injected;
}

extern "C" __attribute__((visibility("default"))) int
pthread_create(pthread_t *thread, const pthread_attr_t *attributes, void *(*entry)(void *),
               void *argument) {
  // Count only creations on the caller after the listener is bound. The
  // engine and any VM workers have independent fault-injection state.
  struct stat info {};
  if (remaining_attempts && lstat(socket_path, &info) == 0 && S_ISSOCK(info.st_mode) &&
      --remaining_attempts == 0) {
    injected = true;
    return EAGAIN;
  }
  static const auto create =
      reinterpret_cast<decltype(&pthread_create)>(dlsym(RTLD_NEXT, "pthread_create"));
  return create(thread, attributes, entry, argument);
}
