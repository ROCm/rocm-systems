/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @file
 * Implicit synchronization between the legacy null stream and blocking streams.
 *
 * A stream created with hipStreamCreate() is a *blocking* stream: work enqueued
 * on it is implicitly ordered after any work already outstanding on the legacy
 * null stream, and vice versa. The runtime implements that by inserting a
 * cross-stream dependency at enqueue time.
 *
 * These tests pin down the cases where the runtime may and may not skip that
 * dependency. A runtime that is too eager to skip one -- for example, by
 * remembering that a stream has already been fenced against a queue and not
 * re-checking what has been enqueued there since -- still passes a simple
 * two-kernel test but drops ordering in the cases below.
 *
 * The checked values hold regardless of timing: they are what the HIP stream
 * ordering rules require, not what a particular schedule happens to produce.
 */

#include <hip_test_common.hh>

#include <vector>

namespace {

constexpr int kSeed = 7;

/**
 * @brief Grid-stride fill. Also the kernel whose completion the blocking
 * streams below must wait for, so it is launched with a small grid to keep it
 * outstanding long enough for the implicit-sync path to be taken rather than
 * short-circuited by an already-complete command. Correctness of the checks
 * does not depend on that -- it only decides whether the path is exercised.
 */
__global__ void fillKernel(int* buf, int value, size_t n) {
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    buf[i] = value;
  }
}

__global__ void incKernel(int* buf, size_t n) {
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    buf[i] += 1;
  }
}

/** @brief Increment used when several unordered streams touch the same buffer. */
__global__ void atomicIncKernel(int* buf, size_t n) {
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    atomicAdd(&buf[i], 1);
  }
}

__global__ void doubleKernel(int* buf, size_t n) {
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    buf[i] *= 2;
  }
}

size_t elementCount() { return isQuickLevel() ? (1 << 12) : (1 << 20); }

/** @brief Fails on the first mismatching element rather than once per element. */
void requireAllEqual(const std::vector<int>& host, int expected) {
  for (size_t i = 0; i < host.size(); ++i) {
    if (host[i] != expected) {
      INFO("first mismatch at element " << i);
      REQUIRE(host[i] == expected);
    }
  }
  SUCCEED();
}

}  // namespace

/**
 * Repeated launches on one blocking stream, behind one outstanding null-stream
 * kernel.
 *
 * Every launch on `stream` re-enters the implicit-sync path while the fill is
 * still outstanding. Only the first launch needs a cross-stream dependency --
 * the rest are ordered behind it by stream order -- so a runtime is free to
 * skip the later ones. What it must not do is let any of them start before the
 * fill, which would make the increments land on uninitialised memory.
 */
HIP_TEST_CASE(Unit_hipStreamImplicitSync_RepeatedLaunchesAfterNullStream) {
  const size_t n = elementCount();
  const int launches = 6;

  int* buf = nullptr;
  HIP_CHECK(hipMalloc(&buf, n * sizeof(int)));

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  hipLaunchKernelGGL(fillKernel, dim3(1), dim3(64), 0, 0, buf, kSeed, n);
  HIP_CHECK(hipGetLastError());

  for (int i = 0; i < launches; ++i) {
    hipLaunchKernelGGL(incKernel, dim3(1), dim3(64), 0, stream, buf, n);
    HIP_CHECK(hipGetLastError());
  }

  std::vector<int> host(n, 0);
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipMemcpy(host.data(), buf, n * sizeof(int), hipMemcpyDeviceToHost));

  requireAllEqual(host, kSeed + launches);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(buf));
}

/**
 * Null-stream work interleaved between launches on the same blocking stream.
 *
 * The dependency the blocking stream needs is a different command each time
 * round, so a runtime that caches "already fenced against the null stream" and
 * skips on that basis alone will miss `doubleKernel` and produce
 * (kSeed + 1 + 1) * 2 instead of (kSeed + 1) * 2 + 1.
 */
HIP_TEST_CASE(Unit_hipStreamImplicitSync_InterleavedNullStreamWork) {
  const size_t n = elementCount();

  int* buf = nullptr;
  HIP_CHECK(hipMalloc(&buf, n * sizeof(int)));

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  hipLaunchKernelGGL(fillKernel, dim3(1), dim3(64), 0, 0, buf, kSeed, n);
  HIP_CHECK(hipGetLastError());

  hipLaunchKernelGGL(incKernel, dim3(1), dim3(64), 0, stream, buf, n);
  HIP_CHECK(hipGetLastError());

  // Ordered after the increment above, because the null stream also blocks on
  // outstanding work in blocking streams.
  hipLaunchKernelGGL(doubleKernel, dim3(1), dim3(64), 0, 0, buf, n);
  HIP_CHECK(hipGetLastError());

  hipLaunchKernelGGL(incKernel, dim3(1), dim3(64), 0, stream, buf, n);
  HIP_CHECK(hipGetLastError());

  std::vector<int> host(n, 0);
  HIP_CHECK(hipStreamSynchronize(stream));
  HIP_CHECK(hipMemcpy(host.data(), buf, n * sizeof(int), hipMemcpyDeviceToHost));

  requireAllEqual(host, (kSeed + 1) * 2 + 1);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(buf));
}

/**
 * Several blocking streams behind one outstanding null-stream kernel.
 *
 * Each stream needs its own dependency on the fill. The streams are not ordered
 * with respect to each other, so the increments are atomic and the total is the
 * only thing checked. A runtime that tracks "has been fenced" per waited queue
 * instead of per waiting stream would fence the first stream and skip the rest.
 */
HIP_TEST_CASE(Unit_hipStreamImplicitSync_MultipleBlockingStreams) {
  const size_t n = elementCount();
  const int num_streams = 4;

  int* buf = nullptr;
  HIP_CHECK(hipMalloc(&buf, n * sizeof(int)));

  std::vector<hipStream_t> streams(num_streams, nullptr);
  for (int i = 0; i < num_streams; ++i) {
    HIP_CHECK(hipStreamCreate(&streams[i]));
  }

  hipLaunchKernelGGL(fillKernel, dim3(1), dim3(64), 0, 0, buf, kSeed, n);
  HIP_CHECK(hipGetLastError());

  for (int i = 0; i < num_streams; ++i) {
    hipLaunchKernelGGL(atomicIncKernel, dim3(1), dim3(64), 0, streams[i], buf, n);
    HIP_CHECK(hipGetLastError());
  }

  std::vector<int> host(n, 0);
  for (int i = 0; i < num_streams; ++i) {
    HIP_CHECK(hipStreamSynchronize(streams[i]));
  }
  HIP_CHECK(hipMemcpy(host.data(), buf, n * sizeof(int), hipMemcpyDeviceToHost));

  requireAllEqual(host, kSeed + num_streams);

  for (int i = 0; i < num_streams; ++i) {
    HIP_CHECK(hipStreamDestroy(streams[i]));
  }
  HIP_CHECK(hipFree(buf));
}
