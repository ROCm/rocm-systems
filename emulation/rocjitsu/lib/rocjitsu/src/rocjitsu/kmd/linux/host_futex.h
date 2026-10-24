// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file host_futex.h
/// @brief Timed host waits that bypass the syscall interposition boundary.

#pragma once

#include "rocjitsu/kmd/linux/libc_passthrough.h"

#include <chrono>
#include <cstdint>
#include <ctime>
#include <linux/futex.h>
#include <sys/syscall.h>
#include <thread>

namespace rocjitsu {

namespace detail {

struct HostFutexSyscallProvider {
  static auto get() noexcept { return libc_passthrough().syscall; }
};

} // namespace detail

/// @brief Parking policy for runtime-owned synchronization words.
/// @details The interposer resolves the native table before starting workers.
/// Standalone VM users may run before that initialization; timed sleeping keeps
/// them live without resolving symbols from a contended path or re-entering the
/// interposer. Callers retain the table's resolve-before-workers contract.
/// @tparam Syscalls Supplies the resolved host syscall function pointer.
template <typename Syscalls = detail::HostFutexSyscallProvider> struct HostFutexParking {
  static void wait(uint32_t *word, uint32_t expected, long timeout_ns) {
    auto native_syscall = Syscalls::get();
    if (!native_syscall) {
      std::this_thread::sleep_for(std::chrono::nanoseconds(timeout_ns));
      return;
    }
    const timespec timeout{0, timeout_ns};
    static_cast<void>(
        native_syscall(SYS_futex, word, FUTEX_WAIT_PRIVATE, expected, &timeout, nullptr, 0));
  }

  static void wake(uint32_t *word) {
    if (auto native_syscall = Syscalls::get())
      static_cast<void>(
          native_syscall(SYS_futex, word, FUTEX_WAKE_PRIVATE, 1, nullptr, nullptr, 0));
  }
};

} // namespace rocjitsu
