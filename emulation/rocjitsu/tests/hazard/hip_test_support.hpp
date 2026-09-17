// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>

#include <vector>

namespace rocjitsu::test {

/// Common HIP setup used by tests that exercise synchronization hazards through
/// rocjitsu. Detection-specific fixtures add their own report assertions.
class HipHazardTestBase : public ::testing::Test {
public:
  template <typename T> T *alloc(int count) {
    T *pointer = nullptr;
    (void)hipMalloc(&pointer, count * sizeof(T));
    return pointer;
  }

  template <typename T> T *allocWithData(int count) {
    T *pointer = alloc<T>(count);
    std::vector<T> host(count);
    for (int index = 0; index < count; ++index)
      host[index] = static_cast<T>(index);
    (void)hipMemcpy(pointer, host.data(), count * sizeof(T), hipMemcpyHostToDevice);
    return pointer;
  }

  void sync() { (void)hipDeviceSynchronize(); }
};

} // namespace rocjitsu::test
