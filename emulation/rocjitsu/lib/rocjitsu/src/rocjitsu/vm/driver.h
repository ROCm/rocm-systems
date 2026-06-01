// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file driver.h
/// @brief Abstract kernel-mode driver interface.
///
/// @details Defines the syscall-level interface between the host runtime
/// (e.g., ROCR's user-mode driver API) and the simulated GPU. The Linux implementation
/// emulates the AMD KFD ioctl interface. Platform-specific interposer libraries
/// route open/close/ioctl/mmap calls to a concrete Driver implementation.

#ifndef ROCJITSU_VM_DRIVER_H_
#define ROCJITSU_VM_DRIVER_H_

#include <cstddef>
#include <cstdint>

#ifdef __linux__
#include <sys/types.h>
#else
using off_t = int64_t;
#endif

namespace rocjitsu {

/// @brief Abstract kernel-mode driver interface for simulated GPU access.
class Driver {
public:
  virtual ~Driver() = default;

  /// @brief Open the device.
  /// @returns File descriptor (or fake fd for simulation).
  virtual int open() = 0;

  /// @brief Close the device.
  /// @returns 0 on success, negative errno on failure.
  virtual int close() = 0;

  /// @brief Handle an ioctl request.
  /// @param request The ioctl command number.
  /// @param arg Pointer to the ioctl argument struct.
  /// @returns 0 on success, negative errno on failure.
  virtual int ioctl(unsigned long request, void *arg) = 0;

  /// @brief Map device memory into the host address space.
  /// @returns Mapped address, or MAP_FAILED on failure.
  virtual void *mmap(void *addr, size_t length, int prot, int flags, off_t offset) = 0;

  /// @brief Unmap previously mapped device memory.
  /// @returns 0 on success, negative errno on failure.
  virtual int munmap(void *addr, size_t length) = 0;
};

} // namespace rocjitsu

#endif // ROCJITSU_VM_DRIVER_H_
