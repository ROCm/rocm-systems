/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstdlib>
#include <iterator>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <hip_test_common.hh>
#include <hip_test_kernels.hh>
#include <hip_test_process.hh>
#include <resource_guards.hh>
#include <utils.hh>

// Back-to-back batches enqueue many batches on one stream before a single synchronization, so
// the stream's completion signals are reused several times over while earlier batches are still
// queued. Copies of 2 MiB stay on SDMA under the default thresholds (peer copies move to SDMA above
// 1 MiB), where one batch becomes several engine operations that all wait on the stream's prior
// work.
inline constexpr size_t kBackToBackCopySize = 2 * 1024 * 1024;
inline constexpr size_t kBackToBackIterations = 400;

// A child that hangs, as these tests do with the legacy tracker when it re-arms a signal under a
// queued wait, is killed after this long. A test takes about a second, and the ones that check
// their own deadlines give up after 20 s at most. The all-device allgather takes a few seconds on 8
// GPUs but about 60 s on a 64-partition MI350X CPX host, where enabling 4032 peer pairs and filling
// and checking 8064 buffers dominate both of its passes.
inline constexpr int kSignalTrackerChildDeadlineSeconds = 30;
inline constexpr int kSignalTrackerChildLongDeadlineSeconds = 120;

// The exit code of a Catch2 run in which every test case was skipped, as HIP_SKIP_TEST does
inline constexpr int kAllTestsSkippedExitCode = 4;

// Runs the child to completion and returns its exit code, or kills it after the deadline and sets
// timed_out. The code of a killed child is 128 plus the signal on Linux.
inline int RunChildWithDeadline(hip::SpawnProc& child, const std::string& args, int deadline_seconds,
                                bool& timed_out) {
  timed_out = false;
  if (child.spawn(args) != 0) {
    return -1;
  }
  std::mutex mutex;
  std::condition_variable condition;
  bool finished = false;
  std::thread watchdog([&] {
    std::unique_lock<std::mutex> lock(mutex);
    if (!condition.wait_for(lock, std::chrono::seconds(deadline_seconds),
                            [&] { return finished; })) {
      timed_out = true;
#if HT_WIN
      TerminateProcess(child.getProcess().hProcess, 1);
#else
      kill(child.getProcess(), SIGKILL);
#endif
    }
  });
  const int exit_code = child.wait();
  {
    std::lock_guard<std::mutex> lock(mutex);
    finished = true;
  }
  condition.notify_all();
  watchdog.join();
  return exit_code;
}

// The runtime picks one of two signal trackers when it loads, by DEBUG_CLR_DEV_SIGNALS: the legacy
// one (0) re-armed a stream's signal under the waits of the batches queued behind it, and the
// retiring one (1, the default since the queue signals moved to device memory) pins them. In a
// process that has no tracker chosen for it, runs the calling test case once with each tracker in a
// process of its own and returns true, so the caller returns. In those processes it returns false
// and the caller runs the test. A child that doesn't finish within deadline_seconds fails the test
// and is killed. If the test skips itself in the children, for lack of a second GPU for example, it
// is skipped here too.
inline bool RunInEachSignalTracker(const std::string& test_name,
                                   const int deadline_seconds = kSignalTrackerChildDeadlineSeconds) {
  constexpr const char* kChildMarker = "HIP_TEST_BATCH_COPY_SIGNAL_TRACKER_CHILD";
  if (std::getenv(kChildMarker) != nullptr) {
    return false;
  }
  bool skipped = false;
  bool failed = false;
  for (const char* dev_signals : {"0", "1"}) {
    INFO("DEBUG_CLR_DEV_SIGNALS=" << dev_signals);
    hip::SpawnProc proc(getSelfExePath());
    proc.setEnv("DEBUG_CLR_DEV_SIGNALS", dev_signals);
    proc.setEnv(kChildMarker, "1");
    bool timed_out = false;
    const int exit_code =
        RunChildWithDeadline(proc, test_name, deadline_seconds, timed_out);
    if (!timed_out && exit_code == kAllTestsSkippedExitCode) {
      skipped = true;
      continue;
    }
    failed = failed || timed_out || exit_code != 0;
    INFO("Exit code " << exit_code);
    INFO("The test did not finish within " << deadline_seconds << " s and was killed: " << timed_out);
    CHECK(!timed_out);
    CHECK(exit_code == 0);
  }
  if (skipped && !failed) {
    HIP_SKIP_TEST("The test skipped itself in the process it ran in.");
  }
  return true;
}

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

// Every ExtOp flag rides the SDMA batch path, which carries transfers between device memory and
// pinned host memory and device-to-device copies, same-device or peer. Pageable memory and
// host-to-host pairings are rejected before the architecture is consulted.
inline bool extOpPairingSupported(const LinearAllocs alloc_type_a,
                                  const LinearAllocs alloc_type_b) {
  if (alloc_type_a == LinearAllocs::malloc || alloc_type_b == LinearAllocs::malloc) {
    return false;
  }

  if (alloc_type_a == LinearAllocs::hipHostMalloc && alloc_type_b == LinearAllocs::hipHostMalloc) {
    return false;
  }

  return true;
}

// A swap exchanges both endpoints, so the two sides are symmetric and named a/b rather than
// src/dst.
inline hipError_t getSwapExpectedReturn(const LinearAllocs alloc_type_a,
                                        const LinearAllocs alloc_type_b, const int device_a = 0,
                                        const int device_b = 0) {
  if (!extOpPairingSupported(alloc_type_a, alloc_type_b)) {
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
                                            [[maybe_unused]] const int device_src = 0,
                                            [[maybe_unused]] const int device_dst = 0,
                                            const int stream_device = 0) {
  if (!extOpPairingSupported(alloc_type_src, alloc_type_dst)) {
    return hipErrorNotSupported;
  }

  // Mirrors CLR's sdma_indirect_supported_ check (rocclr/device/rocm/rocsettings.cpp).
  // Keep in sync if CLR adds architectures.
  int major, minor;
  HIP_CHECK(hipDeviceComputeCapability(&major, &minor, stream_device));

  return (major == 12 && minor >= 5) ? hipSuccess : hipErrorNotSupported;
}
