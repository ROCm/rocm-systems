/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hrr_clock_hook.hh"

#ifndef _WIN32
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

thread_local void (*t_hrr_clock_hook)() = nullptr;

extern "C" int clock_gettime(clockid_t clk, struct timespec* ts) noexcept {
  if (auto hook = t_hrr_clock_hook) hook();
  return static_cast<int>(syscall(SYS_clock_gettime, clk, ts));
}
#endif  // !_WIN32
