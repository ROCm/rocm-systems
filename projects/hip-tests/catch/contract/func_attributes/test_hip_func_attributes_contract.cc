/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>

#include <hip/hip_runtime_api.h>
#include <hip_test_common.hh>

namespace {
// Tiny in-source kernel used as the function symbol under test. It performs a
// single trivial store so it stays launchable while carrying real function
// attributes, avoiding any HIPRTC or external fixture dependency.
__global__ void FuncAttributesKernel(int* output) {
  if (threadIdx.x == 0 && blockIdx.x == 0 && output != nullptr) {
    *output = 1;
  }
}

// Skips the test when no device is visible so that the function attribute
// contracts are only exercised against a provisioned runtime.
void RequireDevice() {
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count <= 0) {
    HIP_SKIP_TEST(HipTest::SkipReason::kNoGpuDevice);
  }
}
}  // namespace

HIP_TEST_CASE(Contract_FuncAttributes_GetAttributes_ReturnsSaneStruct) {
  RequireDevice();

  hipFuncAttributes attributes{};
  HIP_CHECK(hipFuncGetAttributes(&attributes, reinterpret_cast<const void*>(FuncAttributesKernel)));

  // A launchable function must accept at least one thread per block; the exact
  // ceiling is device- and kernel-dependent and therefore not asserted.
  REQUIRE(attributes.maxThreadsPerBlock > 0);

  // The remaining int-valued fields are non-negative resource or version
  // figures; their specific magnitudes are backend-dependent.
  REQUIRE(attributes.numRegs >= 0);
  REQUIRE(attributes.maxDynamicSharedSizeBytes >= 0);
  REQUIRE(attributes.binaryVersion >= 0);
  REQUIRE(attributes.ptxVersion >= 0);
  REQUIRE(attributes.cacheModeCA >= 0);

  // The size_t byte-count fields are well-defined values populated by the
  // query; only that they are readable is part of the structural contract, so
  // their magnitudes are not asserted against any specific bound.
  (void)attributes.constSizeBytes;
  (void)attributes.localSizeBytes;
  (void)attributes.sharedSizeBytes;
}

HIP_TEST_CASE(Contract_FuncAttributes_GetAttributes_NullAttr_IsRejected) {
  RequireDevice();

  // Passing a null output struct must not silently succeed. Backends may report
  // the specific hipErrorInvalidValue or another non-success error, so the
  // contract only requires that the query is rejected.
  const hipError_t status =
      hipFuncGetAttributes(nullptr, reinterpret_cast<const void*>(FuncAttributesKernel));
  REQUIRE(status != hipSuccess);
}

HIP_TEST_CASE(Contract_FuncAttributes_GetAttributes_MatchesScalarGetAttribute) {
  RequireDevice();

  hipFuncAttributes attributes{};
  HIP_CHECK(hipFuncGetAttributes(&attributes, reinterpret_cast<const void*>(FuncAttributesKernel)));

  hipFunction_t function = nullptr;
  HIP_CHECK(hipGetFuncBySymbol(&function, reinterpret_cast<const void*>(FuncAttributesKernel)));
  REQUIRE(function != nullptr);

  int max_threads_per_block = 0;
  HIP_CHECK(hipFuncGetAttribute(&max_threads_per_block,
                                HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function));

  // The struct query and the scalar query describe the same function and must
  // therefore agree on the maximum threads-per-block for that function.
  REQUIRE(max_threads_per_block == attributes.maxThreadsPerBlock);
}

HIP_TEST_CASE(Contract_FuncAttributes_GetFuncBySymbol_ResolvesInSourceKernel) {
  RequireDevice();

  hipFunction_t function = nullptr;
  HIP_CHECK(hipGetFuncBySymbol(&function, reinterpret_cast<const void*>(FuncAttributesKernel)));
  REQUIRE(function != nullptr);

  // The resolved handle must be usable with the driver-style attribute query,
  // confirming that symbol resolution produced a real function object.
  int max_threads_per_block = 0;
  HIP_CHECK(hipFuncGetAttribute(&max_threads_per_block,
                                HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, function));
  REQUIRE(max_threads_per_block > 0);
}

HIP_TEST_CASE(Contract_FuncAttributes_SetAttribute_MaxDynamicSharedMemory_IsAccepted) {
  RequireDevice();

  int current_device = 0;
  hipDeviceProp_t properties{};
  HIP_CHECK(hipGetDevice(&current_device));
  HIP_CHECK(hipGetDeviceProperties(&properties, current_device));

  // Choose a small request that stays within the per-block shared-memory limit
  // so the hint is representable. When the device reports no per-block shared
  // memory, request zero, which remains a valid in-range value.
  int requested_bytes = 0;
  if (properties.sharedMemPerBlock >= 1024) {
    requested_bytes = 1024;
  }

  // Setting a per-function dynamic shared-memory hint must be accepted. The
  // post-state is not part of the contract, so no attribute read-back is
  // asserted afterwards.
  HIP_CHECK(hipFuncSetAttribute(reinterpret_cast<const void*>(FuncAttributesKernel),
                                hipFuncAttributeMaxDynamicSharedMemorySize, requested_bytes));
}

HIP_TEST_CASE(Contract_FuncAttributes_SetAttribute_PreferredCarveout_IsAccepted) {
  RequireDevice();

  // The preferred shared-memory carveout is an optional per-function hint.
  const hipError_t status =
      hipFuncSetAttribute(reinterpret_cast<const void*>(FuncAttributesKernel),
                          hipFuncAttributePreferredSharedMemoryCarveout, 50);
  if (status == hipErrorNotSupported) {
    // Some runtimes do not honor the carveout hint; an unsupported report is a
    // contract-compliant outcome, not a failure.
    return;
  }
  HIP_CHECK(status);
}

HIP_TEST_CASE(Contract_FuncAttributes_SetCacheConfig_PreferNone_Succeeds) {
  RequireDevice();

  // Requesting the neutral cache preference for the function must be accepted.
  // This is a per-function hint on the in-source kernel and performs no global
  // device configuration change.
  HIP_CHECK(hipFuncSetCacheConfig(reinterpret_cast<const void*>(FuncAttributesKernel),
                                  hipFuncCachePreferNone));
}

HIP_TEST_CASE(Contract_FuncAttributes_SetSharedMemConfig_Default_Succeeds) {
  RequireDevice();

  // Requesting the default shared-memory bank size for the function must be
  // accepted. This is a per-function hint and applies no global device change.
  HIP_CHECK(hipFuncSetSharedMemConfig(reinterpret_cast<const void*>(FuncAttributesKernel),
                                      hipSharedMemBankSizeDefault));
}
