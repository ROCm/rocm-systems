/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
Testcase Scenarios:
 1) hipMemset on a plain malloc() buffer succeeds when XNACK is enabled
    (hipDeviceAttributePageableMemoryAccess == 1) and every byte has the
    expected value, without any hipMemcpy.
 2) hipMemsetAsync on a plain malloc() buffer succeeds when XNACK is enabled;
    verify bytes after hipStreamSynchronize.
 3) hipMemset on an mmap(MAP_ANONYMOUS) buffer at a non-page-aligned offset
    succeeds when XNACK is enabled; bytes in [offset, N) have the expected value.
 4) When XNACK is NOT enabled, hipMemset on a plain malloc() buffer returns
    hipErrorInvalidValue (regression guard).
*/

#include <hip_test_common.hh>
#include <sys/mman.h>

// Returns true when the current device supports pageable memory access,
// i.e. HSA_XNACK=1 and the offload target was compiled with xnack+.
static bool pageableAccessEnabled() {
  int dev = 0;
  HIP_CHECK(hipGetDevice(&dev));
  int val = 0;
  HIP_CHECK(hipDeviceGetAttribute(&val, hipDeviceAttributePageableMemoryAccess, dev));
  return val != 0;
}

/**
 * Test Description
 * ------------------------
 *    - Allocate 4096 bytes with malloc, call hipMemset(p, 0xAB, 4096), then
 *      verify every byte equals 0xAB directly on the host (no hipMemcpy
 *      required — the buffer is host memory).
 * Test source
 * ------------------------
 *    - unit/memory/hipMemsetPageable.cc
 * Test requirements
 * ------------------------
 *    - Device must report hipDeviceAttributePageableMemoryAccess == 1
 *      (XNACK-enabled path).
 */
HIP_TEST_CASE(Unit_hipMemset_Pageable_Malloc) {
  if (!pageableAccessEnabled()) {
    HIP_SKIP_TEST(HipTest::SkipReason::kPageableMemoryAccessUnsupported);
  }

  constexpr size_t N = 4096;
  constexpr int kPattern = 0xAB;

  char* p = reinterpret_cast<char*>(malloc(N));
  REQUIRE(p != nullptr);

  HIP_CHECK(hipMemset(p, kPattern, N));

  for (size_t i = 0; i < N; ++i) {
    if (static_cast<unsigned char>(p[i]) != kPattern) {
      CAPTURE(i, static_cast<unsigned>(static_cast<unsigned char>(p[i])), kPattern);
      REQUIRE(false);
    }
  }

  free(p);
}

/**
 * Test Description
 * ------------------------
 *    - Allocate 8192 bytes with malloc, call hipMemsetAsync(p, 0x5A, 8192, s),
 *      synchronize the stream, then verify every byte equals 0x5A on the host.
 * Test source
 * ------------------------
 *    - unit/memory/hipMemsetPageable.cc
 * Test requirements
 * ------------------------
 *    - Device must report hipDeviceAttributePageableMemoryAccess == 1
 *      (XNACK-enabled path).
 */
HIP_TEST_CASE(Unit_hipMemset_Pageable_MallocAsync) {
  if (!pageableAccessEnabled()) {
    HIP_SKIP_TEST(HipTest::SkipReason::kPageableMemoryAccessUnsupported);
  }

  constexpr size_t N = 8192;
  constexpr int kPattern = 0x5A;

  char* p = reinterpret_cast<char*>(malloc(N));
  REQUIRE(p != nullptr);

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  HIP_CHECK(hipMemsetAsync(p, kPattern, N, stream));
  HIP_CHECK(hipStreamSynchronize(stream));

  for (size_t i = 0; i < N; ++i) {
    if (static_cast<unsigned char>(p[i]) != kPattern) {
      CAPTURE(i, static_cast<unsigned>(static_cast<unsigned char>(p[i])), kPattern);
      REQUIRE(false);
    }
  }

  HIP_CHECK(hipStreamDestroy(stream));
  free(p);
}

/**
 * Test Description
 * ------------------------
 *    - mmap an anonymous PRIVATE buffer of 16384 bytes, apply hipMemset
 *      starting at offset 3 (deliberately unaligned), then verify bytes
 *      [3, 16384) all equal 0xC4 on the host.
 * Test source
 * ------------------------
 *    - unit/memory/hipMemsetPageable.cc
 * Test requirements
 * ------------------------
 *    - Device must report hipDeviceAttributePageableMemoryAccess == 1
 *      (XNACK-enabled path).
 */
HIP_TEST_CASE(Unit_hipMemset_Pageable_MmapUnaligned) {
  if (!pageableAccessEnabled()) {
    HIP_SKIP_TEST(HipTest::SkipReason::kPageableMemoryAccessUnsupported);
  }

  constexpr size_t N = 16384;
  constexpr size_t kOffset = 3;
  constexpr int kPattern = 0xC4;

  void* base = mmap(nullptr, N, PROT_READ | PROT_WRITE,
                    MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  REQUIRE(base != MAP_FAILED);

  char* buf = reinterpret_cast<char*>(base);

  HIP_CHECK(hipMemset(buf + kOffset, kPattern, N - kOffset));

  for (size_t i = kOffset; i < N; ++i) {
    if (static_cast<unsigned char>(buf[i]) != kPattern) {
      CAPTURE(i, static_cast<unsigned>(static_cast<unsigned char>(buf[i])), kPattern);
      munmap(base, N);
      REQUIRE(false);
    }
  }

  munmap(base, N);
}

/**
 * Test Description
 * ------------------------
 *    - Regression guard for the XNACK-off path: when the device does NOT
 *      support pageable memory access, hipMemset on a plain malloc() buffer
 *      must return hipErrorInvalidValue.
 * Test source
 * ------------------------
 *    - unit/memory/hipMemsetPageable.cc
 * Test requirements
 * ------------------------
 *    - Device must report hipDeviceAttributePageableMemoryAccess == 0
 *      (XNACK disabled / not available).
 */
HIP_TEST_CASE(Unit_hipMemset_Pageable_RejectsWhenDisabled) {
  if (pageableAccessEnabled()) {
    HIP_SKIP_TEST(HipTest::SkipReason::kGpuXnackNotEnabled);
  }

  constexpr size_t N = 256;

  char* p = reinterpret_cast<char*>(malloc(N));
  REQUIRE(p != nullptr);

  HIP_CHECK_ERROR(hipMemset(p, 0x1, N), hipErrorInvalidValue);

  free(p);
}
