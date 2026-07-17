// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file fd_utils.h
/// @brief Utilities for owning Linux file descriptors.

#ifndef ROCJITSU_KMD_LINUX_FD_UTILS_H_
#define ROCJITSU_KMD_LINUX_FD_UTILS_H_

#include <unistd.h>
#include <utility>

namespace rocjitsu {

/// @brief Move-only RAII owner of a file descriptor.
///
/// @details Closes the descriptor on destruction, reset, or move-assignment.
/// A descriptor of -1 owns nothing.
class UniqueFd {
public:
  UniqueFd() = default;
  explicit UniqueFd(int fd) : fd_(fd) {}
  ~UniqueFd() { reset(); }

  UniqueFd(const UniqueFd &) = delete;
  UniqueFd &operator=(const UniqueFd &) = delete;

  UniqueFd(UniqueFd &&other) noexcept : fd_(other.release()) {}
  UniqueFd &operator=(UniqueFd &&other) noexcept {
    if (this != &other)
      reset(other.release());
    return *this;
  }

  /// @brief Return the owned descriptor, or -1 if empty.
  [[nodiscard]] int get() const { return fd_; }

  /// @brief Release ownership and return the descriptor.
  [[nodiscard]] int release() { return std::exchange(fd_, -1); }

  /// @brief Close the current descriptor and take ownership of @p fd.
  void reset(int fd = -1) {
    if (fd_ >= 0 && fd_ != fd)
      ::close(fd_);
    fd_ = fd;
  }

private:
  int fd_ = -1;
};

} // namespace rocjitsu

#endif // ROCJITSU_KMD_LINUX_FD_UTILS_H_
