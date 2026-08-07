/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <algorithm>
#include <iterator>
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

  const auto diff = std::mismatch(host_out.begin(), host_out.end(), expected.begin());
  INFO("First mismatch at byte " << std::distance(host_out.begin(), diff.first));
  REQUIRE(diff.first == host_out.end());
}

// Allocate the pointer slot that an indirect copy dereferences to reach `target`, keep it alive in
// `slots` and return the slot address to hand to hipMemcpyBatchAsync in place of `target`.
inline void* addPointerSlot(std::vector<LinearAllocGuard<void*>>& slots, void* target,
                            const LinearAllocs alloc_type) {
  LinearAllocGuard<void*> slot(alloc_type, sizeof(void*));
  void* slot_ptr = slot.ptr();

  // The contents of a slot are the address it holds, so the bytes written are the object
  // representation of `target` rather than a payload the caller supplies.
  const auto* address = reinterpret_cast<const unsigned char*>(&target);
  fillBuffer(slot_ptr, std::vector<unsigned char>(address, address + sizeof(void*)), alloc_type);

  slots.push_back(std::move(slot));
  return slot_ptr;
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

// Every ExtOp flag rides the SDMA batch path, which only carries transfers between device memory
// and pinned host memory, plus peer device-to-device copies. Pageable memory, host-to-host and
// same-device device-to-device pairings are rejected before the architecture is consulted.
inline bool extOpPairingSupported(const LinearAllocs alloc_type_a, const LinearAllocs alloc_type_b,
                                  const bool is_p2p) {
  if (alloc_type_a == LinearAllocs::malloc || alloc_type_b == LinearAllocs::malloc) {
    return false;
  }

  if (alloc_type_a == LinearAllocs::hipHostMalloc && alloc_type_b == LinearAllocs::hipHostMalloc) {
    return false;
  }

  if (alloc_type_a == LinearAllocs::hipMalloc && alloc_type_b == LinearAllocs::hipMalloc &&
      !is_p2p) {
    return false;
  }

  return true;
}

// A swap exchanges both endpoints, so the two sides are symmetric and named a/b rather than
// src/dst.
inline hipError_t getSwapExpectedReturn(const LinearAllocs alloc_type_a,
                                        const LinearAllocs alloc_type_b, const int device_a = 0,
                                        const int device_b = 0) {
  // The swap endpoints are peer-to-peer when they live on different devices.
  const bool is_p2p = device_a != device_b;

  if (!extOpPairingSupported(alloc_type_a, alloc_type_b, is_p2p)) {
    return hipErrorNotSupported;
  }

  // Mirrors CLR's sdma_swap_supported_ check (rocclr/device/rocm/rocsettings.cpp).
  // Keep in sync if CLR adds architectures.
  const auto supportsSwap = [](int device) {
    int major, minor;
    HIP_CHECK(hipDeviceComputeCapability(&major, &minor, device));
    return (major == 9 && minor >= 4) || (major == 12 && minor >= 5);
  };

  if (supportsSwap(device_a) && supportsSwap(device_b)) {
    return hipSuccess;
  }

  return hipErrorNotSupported;
}

// An indirect copy is classified from the pointers handed to hipMemcpyBatchAsync, so the allocation
// types below are those of the pointer slot on an indirect side and of the buffer on a direct side.
// Only the device of the stream the batch is enqueued on decides indirect support, so a peer copy
// whose buffers live elsewhere still follows `stream_device`.
inline hipError_t getIndirectExpectedReturn(const LinearAllocs alloc_type_src,
                                            const LinearAllocs alloc_type_dst,
                                            const int device_src = 0, const int device_dst = 0,
                                            const int stream_device = 0) {
  const bool is_p2p = device_src != device_dst;

  if (!extOpPairingSupported(alloc_type_src, alloc_type_dst, is_p2p)) {
    return hipErrorNotSupported;
  }

  // Mirrors CLR's sdma_indirect_supported_ check (rocclr/device/rocm/rocsettings.cpp), which is
  // gfx1250 only. Keep in sync if CLR adds architectures.
  int major, minor;
  HIP_CHECK(hipDeviceComputeCapability(&major, &minor, stream_device));

  return (major == 12 && minor == 5) ? hipSuccess : hipErrorNotSupported;
}
