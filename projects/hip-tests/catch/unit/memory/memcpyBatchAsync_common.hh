/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <utility>
#include <vector>

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>
#include <hip_test_process.hh>
#include <resource_guards.hh>
#include <utils.hh>

// Copy `data` from the host into `buffer`, picking the copy kind from the buffer's allocation type
// so device and host buffers can be filled through one call.
inline void fillBuffer(void* buffer, const std::vector<unsigned char>& data,
                       const LinearAllocs allocType) {
  const hipMemcpyKind kind =
      allocType == LinearAllocs::hipMalloc ? hipMemcpyHostToDevice : hipMemcpyHostToHost;
  HIP_CHECK(hipMemcpy(buffer, data.data(), data.size(), kind));
}

// Read `buffer` back to the host, picking the copy kind from its allocation type, and require it to
// equal `expected` byte for byte. The caller must have made the buffer's device current.
inline void requireBufferEquals(const void* buffer, const std::vector<unsigned char>& expected,
                                const LinearAllocs allocType) {
  std::vector<unsigned char> host_out(expected.size());
  const hipMemcpyKind kind =
      allocType == LinearAllocs::hipMalloc ? hipMemcpyDeviceToHost : hipMemcpyHostToHost;
  HIP_CHECK(hipMemcpy(host_out.data(), buffer, expected.size(), kind));
  for (size_t j = 0; j < expected.size(); ++j) {
    REQUIRE(host_out[j] == expected[j]);
  }
}

// Enable peer access from the first device of each pair to the second. Tolerates pairs whose peer
// access is already enabled so tests can share device state without failing.
inline void EnablePeerAccess(const std::vector<std::pair<int, int>>& peer_pairs) {
  for (const auto& [src_device, dst_device] : peer_pairs) {
    HIP_CHECK(hipSetDevice(src_device));
    hipError_t peer_status = hipDeviceEnablePeerAccess(dst_device, 0);
    if (peer_status != hipSuccess && peer_status != hipErrorPeerAccessAlreadyEnabled) {
      HIP_CHECK(peer_status);
    }
  }
}

inline void DisablePeerAccess(const std::vector<std::pair<int, int>>& peer_pairs) {
  for (const auto& [src_device, dst_device] : peer_pairs) {
    HIP_CHECK(hipSetDevice(src_device));
    HIP_CHECK(hipDeviceDisablePeerAccess(dst_device));
  }
}

// A swap exchanges both endpoints, so the two sides are symmetric and named a/b rather than src/dst.
inline hipError_t getSwapExpectedReturn(const LinearAllocs allocTypeA, const LinearAllocs allocTypeB,
                                        const int deviceA = 0, const int deviceB = 0) {
  // The swap endpoints are peer-to-peer when they live on different devices.
  const bool is_p2p = deviceA != deviceB;

  // Support for H2H will be implemented later.
  if (allocTypeA == LinearAllocs::malloc || allocTypeB == LinearAllocs::malloc) {
    return hipErrorNotSupported;
  }

  if (allocTypeA == LinearAllocs::hipHostMalloc && allocTypeB == LinearAllocs::hipHostMalloc) {
    return hipErrorNotSupported;
  }

  // Support for D2D will be implemented later.
  if (allocTypeA == LinearAllocs::hipMalloc && allocTypeB == LinearAllocs::hipMalloc && !is_p2p) {
    return hipErrorNotSupported;
  }

  const auto supportsSwap = [](int device) {
    int major, minor;
    HIP_CHECK(hipDeviceComputeCapability(&major, &minor, device));
    return major == 9 && minor >= 4;
  };

  if (supportsSwap(deviceA) && supportsSwap(deviceB)) {
    return hipSuccess;
  }

  return hipErrorNotSupported;
}
