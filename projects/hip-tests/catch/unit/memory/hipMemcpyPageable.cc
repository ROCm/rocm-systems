/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/*
 * Test cases for hipMemcpy with plain pageable (malloc/mmap) host memory used
 * in a device role when the device runs with XNACK enabled
 * (HSA_XNACK=1 + an xnack+ offload target).
 *
 * These tests are gated on hipDeviceAttributePageableMemoryAccess and are
 * skipped on any configuration that does not report pageable access support.
 */

#include <hip_test_common.hh>

// ---------------------------------------------------------------------------
// Helper: returns true when hipDeviceAttributePageableMemoryAccess is non-zero
// on the current device, i.e. the driver will handle page-faults for ordinary
// malloc/mmap buffers accessed from the GPU.
// ---------------------------------------------------------------------------
static bool pageableAccessEnabled() {
  int dev = 0;
  HIP_CHECK(hipGetDevice(&dev));
  int val = 0;
  HIP_CHECK(hipDeviceGetAttribute(&val, hipDeviceAttributePageableMemoryAccess, dev));
  return val != 0;
}

// ---------------------------------------------------------------------------
// Simple fill kernel used by the coherence test.
// NOTE: the TU must be compiled with an xnack+ offload target (e.g.
//       --offload-arch=gfx90a:xnack+) for the GPU page-fault handler to be
//       active at runtime.
// ---------------------------------------------------------------------------
__global__ void fill_kernel(int* p, int v, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = v;
}

// ---------------------------------------------------------------------------
// Test 1: plain HtoD — regression guard (should already work wherever the
//         runtime supports pageable memory access).
// ---------------------------------------------------------------------------
HIP_TEST_CASE(Unit_hipMemcpy_Pageable_HtoD) {
  if (!pageableAccessEnabled()) {
    HIP_SKIP_TEST(HipTest::SkipReason::kGpuXnackNotEnabled);
  }

  static constexpr int N = 1024;
  const size_t Nbytes = N * sizeof(int);

  // Allocate and fill plain pageable host source buffer.
  int* h_src = static_cast<int*>(malloc(Nbytes));
  REQUIRE(h_src != nullptr);
  for (int i = 0; i < N; ++i) h_src[i] = i;

  // Device buffer.
  int* d_buf = nullptr;
  HIP_CHECK(hipMalloc(&d_buf, Nbytes));

  // HtoD copy using pageable host pointer.
  HIP_CHECK(hipMemcpy(d_buf, h_src, Nbytes, hipMemcpyHostToDevice));

  // Copy back to a second malloc buffer and verify.
  int* h_dst = static_cast<int*>(malloc(Nbytes));
  REQUIRE(h_dst != nullptr);
  HIP_CHECK(hipMemcpy(h_dst, d_buf, Nbytes, hipMemcpyDeviceToHost));

  for (int i = 0; i < N; ++i) {
    REQUIRE(h_dst[i] == i);
  }

  HIP_CHECK(hipFree(d_buf));
  free(h_src);
  free(h_dst);
}

// ---------------------------------------------------------------------------
// Test 2: malloc'd buffer used AS the device (source) pointer in a DtoD copy.
//         This is the genuinely new XNACK-enabled capability.
// ---------------------------------------------------------------------------
HIP_TEST_CASE(Unit_hipMemcpy_Pageable_AsDevice_DtoD) {
  if (!pageableAccessEnabled()) {
    HIP_SKIP_TEST(HipTest::SkipReason::kGpuXnackNotEnabled);
  }

  static constexpr int N = 512;
  const size_t Nbytes = N * sizeof(int);

  // Plain pageable source — used as the "device" source pointer.
  int* h_src = static_cast<int*>(malloc(Nbytes));
  REQUIRE(h_src != nullptr);
  for (int i = 0; i < N; ++i) h_src[i] = i * 7 + 1;

  // hipMalloc'd destination.
  int* d_dst = nullptr;
  HIP_CHECK(hipMalloc(&d_dst, Nbytes));

  // DtoD copy where the source is a plain pageable host pointer.
  HIP_CHECK(hipMemcpy(d_dst, h_src, Nbytes, hipMemcpyDeviceToDevice));

  // Copy dst back to host and verify.
  int* h_out = static_cast<int*>(malloc(Nbytes));
  REQUIRE(h_out != nullptr);
  HIP_CHECK(hipMemcpy(h_out, d_dst, Nbytes, hipMemcpyDeviceToHost));

  for (int i = 0; i < N; ++i) {
    REQUIRE(h_out[i] == i * 7 + 1);
  }

  HIP_CHECK(hipFree(d_dst));
  free(h_src);
  free(h_out);
}

// ---------------------------------------------------------------------------
// Test 3: coherence — a kernel writes into a pageable buffer; hipMemcpy
//         (DtoD) then reads from it.  Requires XNACK-enabled offload target.
//
// NOTE: this TU must be compiled with an xnack+ offload target
//       (e.g. --offload-arch=gfx90a:xnack+) for GPU page-fault handling to
//       be active at kernel execution time.
// ---------------------------------------------------------------------------
HIP_TEST_CASE(Unit_hipMemcpy_Pageable_CoherenceAfterKernel) {
  if (!pageableAccessEnabled()) {
    HIP_SKIP_TEST(HipTest::SkipReason::kGpuXnackNotEnabled);
  }

  static constexpr int N = 1024;
  static constexpr int kFillValue = 0x2468;
  const size_t Nbytes = N * sizeof(int);

  // Plain pageable buffer that the kernel writes into via page-fault.
  int* p = static_cast<int*>(malloc(Nbytes));
  REQUIRE(p != nullptr);

  // Launch fill_kernel to write kFillValue into every element of p.
  static constexpr int kBlockDim = 256;
  const int kGridDim = (N + kBlockDim - 1) / kBlockDim;
  fill_kernel<<<kGridDim, kBlockDim>>>(p, kFillValue, N);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  // DtoD copy from the pageable buffer into a separate malloc buffer.
  int* out = static_cast<int*>(malloc(Nbytes));
  REQUIRE(out != nullptr);
  HIP_CHECK(hipMemcpy(out, p, Nbytes, hipMemcpyDeviceToDevice));

  for (int i = 0; i < N; ++i) {
    REQUIRE(out[i] == kFillValue);
  }

  free(p);
  free(out);
}

// ---------------------------------------------------------------------------
// Test 4: on non-XNACK hardware the runtime must reject a DtoD copy that
//         supplies a plain pageable host pointer (XNACK-off regression guard).
// ---------------------------------------------------------------------------
HIP_TEST_CASE(Unit_hipMemcpy_Pageable_RejectsWhenDisabled) {
  if (pageableAccessEnabled()) {
    HIP_SKIP_TEST("pageable memory access is enabled; skipping rejection test.");
  }

  static constexpr size_t kBytes = 256;

  int* h_src = static_cast<int*>(malloc(kBytes));
  REQUIRE(h_src != nullptr);

  int* d_dst = nullptr;
  HIP_CHECK(hipMalloc(&d_dst, kBytes));

  // Passing a pageable host pointer as the source of a DtoD copy must fail
  // with hipErrorInvalidValue when pageable access is not enabled.
  HIP_CHECK_ERROR(hipMemcpy(d_dst, h_src, kBytes, hipMemcpyDeviceToDevice),
                  hipErrorInvalidValue);

  HIP_CHECK(hipFree(d_dst));
  free(h_src);
}
