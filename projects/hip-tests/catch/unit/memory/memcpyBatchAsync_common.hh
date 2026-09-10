/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <algorithm>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>
#include <hip_test_process.hh>
#include <resource_guards.hh>
#include <utils.hh>

#if HT_AMD
inline void skipMemcpyBatchAsyncIfAnyGfx1250() {
  const int device_count = HipTest::getDeviceCount();
  for (int dev = 0; dev < device_count; ++dev) {
    hipDeviceProp_t props{};
    HIP_CHECK(hipGetDeviceProperties(&props, dev));
    const std::string arch(props.gcnArchName);
    if (arch.find("gfx1250") != std::string::npos) {
      HIP_SKIP_TEST("ROCM-29275: not supported on gfx1250");
    }
  }
}
#endif

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

// Allocate a buffer holding `contents`, keep it alive in `allocations` and return its address.
inline void* addBuffer(std::vector<LinearAllocGuard<unsigned char>>& allocations,
                       const std::vector<unsigned char>& contents, const LinearAllocs alloc_type) {
  LinearAllocGuard<unsigned char> alloc(alloc_type, contents.size());
  void* ptr = alloc.ptr();
  allocations.push_back(std::move(alloc));
  fillBuffer(ptr, contents, alloc_type);
  return ptr;
}

// Allocate the pointer slot that an indirect copy dereferences to reach `target`, keep it alive in
// `slots` and return the slot address to hand to hipMemcpyBatchAsync in place of `target`.
inline void* addPointerSlot(std::vector<LinearAllocGuard<void*>>& slots, void* target,
                            const LinearAllocs alloc_type) {
  LinearAllocGuard<void*> slot(alloc_type, sizeof(void*));
  void* slot_ptr = slot.ptr();

  const auto* address = reinterpret_cast<const unsigned char*>(&target);
  fillBuffer(slot_ptr, std::vector<unsigned char>(address, address + sizeof(void*)), alloc_type);

  slots.push_back(std::move(slot));
  return slot_ptr;
}

// Buffers and pointer arrays for one indirect-copy batch.
struct IndirectCopyBuffers {
  // The buffers the copies must read from and write to. Destinations start zeroed.
  std::vector<void*> src_ptrs;
  std::vector<void*> dst_ptrs;
  // The pointers hipMemcpyBatchAsync receives.
  std::vector<void*> batch_src_ptrs;
  std::vector<void*> batch_dst_ptrs;
  // The pattern written to the src.
  std::vector<std::vector<unsigned char>> initial_values;
  // Keeps the source and destination buffers alive.
  std::vector<LinearAllocGuard<unsigned char>> allocations;
  // Keeps alive the slots the caller took from addPointerSlot.
  std::vector<LinearAllocGuard<void*>> slots;
};

// Allocate and initialize `count` source and destination buffers.
inline IndirectCopyBuffers makeIndirectCopyBuffers(const size_t count, const size_t size_in_bytes,
                                                   const LinearAllocs alloc_type_src,
                                                   const LinearAllocs alloc_type_dst,
                                                   const int device_src = 0,
                                                   const int device_dst = 0) {
  const std::vector<unsigned char> zeros(size_in_bytes, 0);
  IndirectCopyBuffers buffers;

  for (size_t i = 0; i < count; ++i) {
    buffers.initial_values.emplace_back(size_in_bytes, static_cast<unsigned char>(10 + i));

    HIP_CHECK(hipSetDevice(device_src));
    buffers.src_ptrs.push_back(
        addBuffer(buffers.allocations, buffers.initial_values.back(), alloc_type_src));

    HIP_CHECK(hipSetDevice(device_dst));
    buffers.dst_ptrs.push_back(addBuffer(buffers.allocations, zeros, alloc_type_dst));
  }

  buffers.batch_src_ptrs = buffers.src_ptrs;
  buffers.batch_dst_ptrs = buffers.dst_ptrs;
  return buffers;
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
