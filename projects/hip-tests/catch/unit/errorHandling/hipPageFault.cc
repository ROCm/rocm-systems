/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include "hip/hip_runtime_api.h"
#include "hip_test_context.hh"

__global__ void PageFaultKernel(int* ptr) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  ptr[tid] = tid;
}

static bool isAbortOnErrorEnabled() {
  std::string abort_env = TestContext::getEnvVar("HIP_SKIP_ABORT_ON_GPU_ERROR");
  if (!abort_env.empty()) {
    try {
      return !std::stoi(abort_env);
    } catch (...) {
      return true;
    }
  }
  return false;
}

/**
 * Test Description
 * ------------------------
 *  - Triggers a page fault by accessing nullptr and verifies error is sticky across multiple HIP API calls.
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipPageFault.cc
 */
HIP_TEST_CASE(Unit_PageFault_NullPtrAccess) {
#if HT_AMD
  if (isAbortOnErrorEnabled()) {
    HIP_SKIP_TEST("Test incompatible with aborts enabled through HIP_SKIP_ABORT_ON_GPU_ERROR.");
  }
#endif

  int* d_temp;
  PageFaultKernel<<<1, 16>>>(nullptr);

  // Verify error is sticky across multiple HIP API calls
  HIP_CHECK_ERROR(hipDeviceSynchronize(), hipErrorIllegalAddress);
  HIP_CHECK_ERROR(hipStreamSynchronize(nullptr), hipErrorIllegalAddress);
  HIP_CHECK_ERROR(hipMalloc(&d_temp, 64), hipErrorIllegalAddress);
  HIP_CHECK_ERROR(hipFree(d_temp), hipErrorIllegalAddress);
  HIP_CHECK_ERROR(hipMemset(d_temp, 0, 64), hipErrorIllegalAddress);
  HIP_CHECK_ERROR(hipPeekAtLastError(), hipErrorIllegalAddress);
  HIP_CHECK_ERROR(hipDeviceReset(), hipErrorIllegalAddress);
}

/**
 * Test Description
 * ------------------------
 *  - Triggers a page fault by accessing invalid address and verifies error is sticky across multiple HIP API calls.
 * Test source
 * ------------------------
 *  - unit/errorHandling/hipPageFault.cc
 */
HIP_TEST_CASE(Unit_PageFault_InvalidAddress) {
#if HT_AMD
  if (isAbortOnErrorEnabled()) {
    HIP_SKIP_TEST("Test incompatible with aborts enabled through HIP_SKIP_ABORT_ON_GPU_ERROR.");
  }
#endif

  int* invalid_ptr = reinterpret_cast<int*>(0xDEADBEEF);
  int* d_temp;
  hipStream_t stream;
  int device_count;

  PageFaultKernel<<<1, 16>>>(invalid_ptr);

  // Verify error is sticky across different HIP API categories
  HIP_CHECK_ERROR(hipDeviceSynchronize(), hipErrorIllegalAddress);
  HIP_CHECK_ERROR(hipStreamCreate(&stream), hipErrorIllegalAddress);
  HIP_CHECK_ERROR(hipMalloc(&d_temp, 128), hipErrorIllegalAddress);
  HIP_CHECK_ERROR(hipMemcpy(d_temp, &invalid_ptr, sizeof(int), hipMemcpyHostToDevice), hipErrorIllegalAddress);
  HIP_CHECK_ERROR(hipStreamDestroy(stream), hipErrorIllegalAddress);
  HIP_CHECK_ERROR(hipGetDeviceCount(&device_count), hipErrorIllegalAddress);
  HIP_CHECK_ERROR(hipPeekAtLastError(), hipErrorIllegalAddress);
  HIP_CHECK_ERROR(hipDeviceReset(), hipErrorIllegalAddress);
}
