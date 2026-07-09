// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file kfd_ioctl_result.h
/// @brief Translate an emulated driver's ioctl result to the libc ioctl(2)
/// contract at the interposer's syscall boundary.

#ifndef ROCJITSU_KMD_LINUX_KFD_IOCTL_RESULT_H
#define ROCJITSU_KMD_LINUX_KFD_IOCTL_RESULT_H

#include <cerrno>

namespace rocjitsu {
namespace kmd {

/// @brief Convert a kernel-style driver ioctl result into the libc ioctl(2)
/// return/`errno` contract.
///
/// @details The emulated KFD/DRM drivers return results the way the kernel does
/// internally: `0` or a positive value on success, and a negative `-errno` on
/// failure. libc's `ioctl(2)` — which every caller, including rocm-dbgapi,
/// relies on — instead returns `-1` and sets `errno` on failure. rocm-dbgapi in
/// particular computes `ret < 0 ? -errno : ret`, so returning a raw negative
/// value with a stale `errno` makes it misread e.g. `-EAGAIN` as success. The
/// interposer stands in for libc, so it performs that translation here.
///
/// A non-negative result passes through unchanged — including the DBG_TRAP
/// `SUSPEND_QUEUES`/`RESUME_QUEUES` "number of queues handled" count, which is a
/// meaningful positive return value, not an error.
///
/// @param r Driver result: `>= 0` on success, `-errno` on failure.
/// @returns @p r unchanged when non-negative; otherwise `-1` with `errno` set to
///          `-r`.
inline int kfd_ioctl_ret(int r) {
  if (r < 0) {
    errno = -r;
    return -1;
  }
  return r;
}

} // namespace kmd
} // namespace rocjitsu

#endif // ROCJITSU_KMD_LINUX_KFD_IOCTL_RESULT_H
