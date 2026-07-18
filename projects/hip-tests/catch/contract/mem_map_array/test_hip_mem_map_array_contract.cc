/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>
#include <contract_cleanup.hh>

// hipMemMapArrayAsync maps or unmaps subregions of sparse HIP arrays and sparse
// HIP mipmapped arrays. Exercising a real sparse map requires a sparse-array
// fixture built from CUDA-driver-only descriptor flags (CUDA_ARRAY3D_SPARSE and
// the tile-pool allocation usage), which are not defined in the HIP public
// headers, and sparse-array support that the AMD runtime does not implement (the
// map path returns hipErrorNotSupported by design). These contracts therefore
// exercise the portable, externally observable invariants that need no sparse
// fixture: the documented invalid-input rejections. The exact error code is
// backend-specific, so only a non-success status is required.
namespace {
void RequireDevice() {
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count <= 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kNoGpuDevice);
  }
}
}  // namespace

HIP_TEST_CASE(Contract_MemMapArray_NullMapList_IsRejected) {
  RequireDevice();
  hip::contract::ContractCleanup cleanup;

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });

  // A null map-info list is invalid input and must be rejected rather than
  // silently succeeding. Any pre-existing sticky error is cleared first, and the
  // sticky error left by the rejection is cleared afterward, so neither leaks
  // into later tests.
  HIP_CHECK(hipGetLastError());
  const hipError_t status = hipMemMapArrayAsync(nullptr, 1, stream);
  REQUIRE(status != hipSuccess);
  (void)hipGetLastError();
}

HIP_TEST_CASE(Contract_MemMapArray_ZeroCount_IsRejected) {
  RequireDevice();
  hip::contract::ContractCleanup cleanup;

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));
  cleanup.Add([stream] { (void)hipStreamDestroy(stream); });

  // A zero operation count is invalid input and must be rejected. A
  // zero-initialized map-info entry is supplied so the rejection is attributable
  // to the count rather than a null list.
  HIP_CHECK(hipGetLastError());
  hipArrayMapInfo map_info{};
  const hipError_t status = hipMemMapArrayAsync(&map_info, 0, stream);
  REQUIRE(status != hipSuccess);
  (void)hipGetLastError();
}
