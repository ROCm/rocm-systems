/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <vector>
#include <cstring>

#include "../common/TestChecks.hpp"

namespace RcclUnitTesting
{

// RAII typed device memory buffer with host<->device transfer helpers.
// Standalone — no RCCL or NCCL dependencies.
template<typename T>
class DeviceBuffer {
public:
  T*     ptr   = nullptr;
  size_t count = 0;

  explicit DeviceBuffer(size_t n) : count(n) {
    HIP_EXPECT(hipMalloc(&ptr, n * sizeof(T)));
  }

  ~DeviceBuffer() { if (ptr) HIP_EXPECT(hipFree(ptr)); }

  DeviceBuffer(const DeviceBuffer&)            = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  DeviceBuffer(DeviceBuffer&& o) noexcept : ptr(o.ptr), count(o.count) {
    o.ptr = nullptr;
  }

  void copyFrom(const std::vector<T>& h) {
    HIP_CHECK(hipMemcpy(ptr, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice));
  }

  void copyFrom(const T* src, size_t n) {
    HIP_CHECK(hipMemcpy(ptr, src, n * sizeof(T), hipMemcpyHostToDevice));
  }

  void upload(const T& val) {
    HIP_CHECK(hipMemcpy(ptr, &val, sizeof(T), hipMemcpyHostToDevice));
  }

  std::vector<T> copyTo() const {
    std::vector<T> h(count);
    HIP_EXPECT(hipMemcpy(h.data(), ptr, count * sizeof(T), hipMemcpyDeviceToHost));
    return h;
  }

  T download() const {
    T val;
    HIP_EXPECT(hipMemcpy(&val, ptr, sizeof(T), hipMemcpyDeviceToHost));
    return val;
  }

  void zero() { HIP_CHECK(hipMemset(ptr, 0, count * sizeof(T))); }
};

// Base test fixture: selects GPU 0 and provides launch helpers.
class DeviceTestBase : public ::testing::Test {
protected:
  void SetUp() override { ASSERT_EQ(hipSetDevice(0), hipSuccess); }

  static constexpr int kDefaultBlockSize = 256;

  static dim3 gridFor(size_t n, int blockSize = kDefaultBlockSize) {
    return dim3(static_cast<unsigned>((n + blockSize - 1) / blockSize));
  }

  void syncAndCheck() {
    ASSERT_EQ(hipGetLastError(), hipSuccess);
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);
  }
};

} // namespace RcclUnitTesting
