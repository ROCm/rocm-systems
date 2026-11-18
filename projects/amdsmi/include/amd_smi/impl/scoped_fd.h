// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMD_SMI_INCLUDE_IMPL_SCOPED_FD_H_
#define AMD_SMI_INCLUDE_IMPL_SCOPED_FD_H_

#include <unistd.h>

#include <string>

class ScopedFD {
 private:
  int fd_;
  std::string path_;

 public:
  ScopedFD(const std::string& path, int flags);
  ~ScopedFD();

  // Non-copyable
  ScopedFD(const ScopedFD&) = delete;
  ScopedFD& operator=(const ScopedFD&) = delete;

  // Movable
  ScopedFD(ScopedFD&& other) noexcept;
  ScopedFD& operator=(ScopedFD&& other) noexcept;

  int get() const;
  bool valid() const;
  operator int() const;  // Allows direct use as int
};

#endif  // AMD_SMI_INCLUDE_IMPL_SCOPED_FD_H_
