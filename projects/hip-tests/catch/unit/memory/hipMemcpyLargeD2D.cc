/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
This testcase verifies large same-device copies with hipMemcpy and hipMemcpyAsync.
Above half of the last-level cache (MALL) the device-to-device blit switches to a full grid with
nontemporal stores, and above the whole MALL to nontemporal loads too. The sizes cross both
thresholds for MALL sizes from 64 MiB to 256 MiB, and include sizes that are not a multiple of
16 bytes and offsets that are not 16-byte aligned. Bytes around the destination range are checked
to be untouched.
*/

#include <hip_test_common.hh>

namespace {
constexpr size_t kMiB = 1024 * 1024;
constexpr size_t kGuard = 4096;
constexpr unsigned char kSentinel = 0xA5;

__global__ void FillPattern(unsigned char* buf, size_t n, unsigned int salt) {
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    buf[i] = static_cast<unsigned char>((i * 2654435761u) >> 13) ^ static_cast<unsigned char>(salt);
  }
}

// Counts bytes of dst that differ from src inside [dst_off, dst_off + len) or from the
// sentinel outside of it.
__global__ void CountMismatches(const unsigned char* src, size_t src_off, const unsigned char* dst,
                                size_t dst_off, size_t len, size_t total, unsigned long long* bad) {
  for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < total;
       i += static_cast<size_t>(gridDim.x) * blockDim.x) {
    const bool in_range = i >= dst_off && i < dst_off + len;
    const unsigned char expected = in_range ? src[src_off + (i - dst_off)] : kSentinel;
    if (dst[i] != expected) {
      atomicAdd(bad, 1ull);
    }
  }
}
}  // namespace

/**
 * Test Description
 * ------------------------
 *  - Copies 48 MiB to 300 MiB within one device, with odd sizes and aligned and unaligned
 *    offsets, using hipMemcpy and hipMemcpyAsync. Verifies the copied range and the guard
 *    bytes on both sides on the device.
 * Test source
 * ------------------------
 *  - unit/memory/hipMemcpyLargeD2D.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.2
 */
HIP_TEST_CASE(Unit_hipMemcpy_LargeDeviceToDevice) {
  const size_t size = GENERATE(48 * kMiB + 5, 100 * kMiB + 3, 129 * kMiB + 1, 200 * kMiB + 7,
                               257 * kMiB + 15, 300 * kMiB);
  const auto offsets =
      GENERATE(std::make_pair<size_t, size_t>(0, 0), std::make_pair<size_t, size_t>(16, 48),
               std::make_pair<size_t, size_t>(4, 8));
  const bool use_async = GENERATE(false, true);
  const size_t src_off = offsets.first;
  const size_t dst_off = offsets.second;
  const size_t total = size + 2 * kGuard;

  size_t free_mem = 0, total_mem = 0;
  HIP_CHECK(hipMemGetInfo(&free_mem, &total_mem));
  if (free_mem < 2 * total + kMiB) {
    HIP_SKIP_TEST("Not enough free device memory");
  }

  unsigned char *src = nullptr, *dst = nullptr;
  unsigned long long* bad = nullptr;
  hipStream_t stream = nullptr;
  HIP_CHECK(hipMalloc(&src, total));
  HIP_CHECK(hipMalloc(&dst, total));
  HIP_CHECK(hipMalloc(&bad, sizeof(*bad)));
  HIP_CHECK(hipStreamCreate(&stream));

  FillPattern<<<1024, 256>>>(src, total, static_cast<unsigned int>(size + dst_off));
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemset(dst, kSentinel, total));
  HIP_CHECK(hipMemset(bad, 0, sizeof(*bad)));
  HIP_CHECK(hipDeviceSynchronize());

  if (use_async) {
    HIP_CHECK(hipMemcpyAsync(dst + dst_off, src + src_off, size, hipMemcpyDeviceToDevice, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
  } else {
    HIP_CHECK(hipMemcpy(dst + dst_off, src + src_off, size, hipMemcpyDeviceToDevice));
  }

  CountMismatches<<<4096, 256>>>(src, src_off, dst, dst_off, size, total, bad);
  HIP_CHECK(hipGetLastError());
  unsigned long long bad_h = 0;
  HIP_CHECK(hipMemcpy(&bad_h, bad, sizeof(bad_h), hipMemcpyDeviceToHost));
  INFO("size " << size << " src_off " << src_off << " dst_off " << dst_off << " async "
               << use_async);
  REQUIRE(bad_h == 0);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipFree(bad));
  HIP_CHECK(hipFree(dst));
  HIP_CHECK(hipFree(src));
}
