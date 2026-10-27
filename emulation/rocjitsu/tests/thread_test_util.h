// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <charconv>
#include <string_view>
#include <sys/types.h>
#include <system_error>
#include <unistd.h>

namespace rocjitsu::test {

// These tests inspect /proc/self/task/<tid>/syscall to observe blocked threads.
// Read the kernel TID from procfs using libc, including on older glibc versions
// without gettid(). Return -1 if procfs cannot identify the current thread.
inline pid_t current_thread_id() {
  char target[64];
  const ssize_t length = readlink("/proc/thread-self", target, sizeof(target));
  if (length <= 0 || static_cast<size_t>(length) == sizeof(target))
    return -1;
  const std::string_view path(target, static_cast<size_t>(length));
  const size_t separator = path.rfind('/');
  if (separator == std::string_view::npos)
    return -1;
  pid_t tid = -1;
  const char *end = path.data() + path.size();
  const std::from_chars_result parsed = std::from_chars(path.data() + separator + 1, end, tid);
  if (parsed.ec != std::errc{} || parsed.ptr != end || tid <= 0)
    return -1;
  return tid;
}

} // namespace rocjitsu::test
