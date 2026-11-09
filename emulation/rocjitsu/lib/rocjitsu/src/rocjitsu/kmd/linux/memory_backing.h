// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file memory_backing.h
/// @brief Device identity retained across shared-memory descriptor transfers.

#pragma once

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <string_view>

#include <fcntl.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace rocjitsu::kmd {

inline constexpr std::string_view kDeviceBackingNamePrefix = "rocjitsu_alloc_renderD";

/// @brief Recognize device backing independently of a descriptor's number.
/// @details SCM_RIGHTS and DMA-BUF export create independent descriptor numbers for the
/// same backing. Its name preserves the device identity across those transfers.
/// Only allocations associated with published topology carry this marker and
/// receive device dump policy. Standalone allocations made before topology is
/// published retain ordinary host dump eligibility through every alias.
inline bool is_device_memory_backing(int fd) {
  const int saved_errno = errno;
  std::array<char, 64> path{};
  std::array<char, 256> target{};
  std::snprintf(path.data(), path.size(), "/proc/self/fd/%d", fd);
  const ssize_t length =
      syscall(SYS_readlinkat, AT_FDCWD, path.data(), target.data(), target.size());
  errno = saved_errno;
  if (length <= 0)
    return false;
  const std::string_view name(target.data(), static_cast<std::size_t>(length));
  constexpr std::string_view kMemfdPrefix = "/memfd:";
  return name.starts_with(kMemfdPrefix) &&
         name.substr(kMemfdPrefix.size()).starts_with(kDeviceBackingNamePrefix);
}

} // namespace rocjitsu::kmd
