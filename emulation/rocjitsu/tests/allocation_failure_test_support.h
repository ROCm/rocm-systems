// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

#include <cstddef>

namespace rocjitsu::test_support {

// Linker wrapping keeps allocation failure injection in the test executable;
// production allocators and sanitizer interception remain unchanged.
class ScopedAllocationFailure {
public:
  explicit ScopedAllocationFailure(size_t minimum_bytes);
  ~ScopedAllocationFailure();
  ScopedAllocationFailure(const ScopedAllocationFailure &) = delete;
  ScopedAllocationFailure &operator=(const ScopedAllocationFailure &) = delete;
  size_t failures() const { return failures_; }
  bool reject(size_t bytes) {
    if (bytes < minimum_bytes_)
      return false;
    ++failures_;
    return true;
  }

private:
  size_t minimum_bytes_;
  size_t failures_ = 0;
  ScopedAllocationFailure *previous_;
};

} // namespace rocjitsu::test_support
