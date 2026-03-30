/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "execution_control_common.hh"

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <resource_guards.hh>
#include <utils.hh>

HIP_TEST_CASE(Unit_hipLaunchKernel_Positive_Basic) {
  SECTION("Kernel with no arguments") {
    HIP_CHECK(hipLaunchKernel(reinterpret_cast<void*>(kernel), dim3{1, 1, 1}, dim3{1, 1, 1},
                              nullptr, 0, nullptr));
    HIP_CHECK(hipDeviceSynchronize());
  }

  SECTION("Kernel with arguments using kernelParams") {
    LinearAllocGuard<int> result_dev(LinearAllocs::hipMalloc, sizeof(int));
    HIP_CHECK(hipMemset(result_dev.ptr(), 0, sizeof(*result_dev.ptr())));
    int* result_ptr = result_dev.ptr();
    void* kernel_args[1] = {&result_ptr};
    HIP_CHECK(hipLaunchKernel(reinterpret_cast<void*>(kernel_42), dim3{1, 1, 1}, dim3{1, 1, 1},
                              kernel_args, 0, nullptr));
    int result = 0;
    HIP_CHECK(hipMemcpy(&result, result_dev.ptr(), sizeof(result), hipMemcpyDefault));
    REQUIRE(result == 42);
  }
}

HIP_TEST_CASE(Unit_hipLaunchKernel_Positive_Parameters) {
  SECTION("blockDim.x == maxBlockDimX") {
    const unsigned int x = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimX, 0);
    HIP_CHECK(hipLaunchKernel(reinterpret_cast<void*>(kernel), dim3{1, 1, 1}, dim3{x, 1, 1},
                              nullptr, 0, nullptr));
  }

  SECTION("blockDim.y == maxBlockDimY") {
    const unsigned int y = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimY, 0);
    HIP_CHECK(hipLaunchKernel(reinterpret_cast<void*>(kernel), dim3{1, 1, 1}, dim3{y, 1, 1},
                              nullptr, 0, nullptr));
  }

  SECTION("blockDim.z == maxBlockDimZ") {
    const unsigned int z = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimZ, 0);
    HIP_CHECK(hipLaunchKernel(reinterpret_cast<void*>(kernel), dim3{1, 1, 1}, dim3{z, 1, 1},
                              nullptr, 0, nullptr));
  }
}

HIP_TEST_CASE(Unit_hipLaunchKernel_Negative_Parameters) {
  SECTION("f == nullptr") {
    HIP_CHECK_ERROR(hipLaunchKernel(nullptr, dim3{1, 1, 1}, dim3{1, 1, 1}, nullptr, 0, nullptr),
                    hipErrorInvalidDeviceFunction);
  }

  SECTION("gridDim.x == 0") {
    HIP_CHECK_ERROR(hipLaunchKernel(reinterpret_cast<void*>(kernel), dim3{0, 1, 1}, dim3{1, 1, 1},
                                    nullptr, 0, nullptr),
                    hipErrorInvalidConfiguration);
  }

  SECTION("gridDim.y == 0") {
    HIP_CHECK_ERROR(hipLaunchKernel(reinterpret_cast<void*>(kernel), dim3{1, 0, 1}, dim3{1, 1, 1},
                                    nullptr, 0, nullptr),
                    hipErrorInvalidConfiguration);
  }

  SECTION("gridDim.z == 0") {
    HIP_CHECK_ERROR(hipLaunchKernel(reinterpret_cast<void*>(kernel), dim3{1, 1, 0}, dim3{1, 1, 1},
                                    nullptr, 0, nullptr),
                    hipErrorInvalidConfiguration);
  }

  SECTION("blockDim.x == 0") {
    HIP_CHECK_ERROR(hipLaunchKernel(reinterpret_cast<void*>(kernel), dim3{1, 1, 1}, dim3{0, 1, 1},
                                    nullptr, 0, nullptr),
                    hipErrorInvalidConfiguration);
  }

  SECTION("blockDim.y == 0") {
    HIP_CHECK_ERROR(hipLaunchKernel(reinterpret_cast<void*>(kernel), dim3{1, 1, 1}, dim3{1, 0, 1},
                                    nullptr, 0, nullptr),
                    hipErrorInvalidConfiguration);
  }

  SECTION("blockDim.z == 0") {
    HIP_CHECK_ERROR(hipLaunchKernel(reinterpret_cast<void*>(kernel), dim3{1, 1, 1}, dim3{1, 1, 0},
                                    nullptr, 0, nullptr),
                    hipErrorInvalidConfiguration);
  }

  SECTION("blockDim.x > maxBlockDimX") {
    const unsigned int x = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimX, 0) + 1u;
    HIP_CHECK_ERROR(hipLaunchKernel(reinterpret_cast<void*>(kernel), dim3{1, 1, 1}, dim3{x, 1, 1},
                                    nullptr, 0, nullptr),
                    hipErrorInvalidConfiguration);
  }

  SECTION("blockDim.y > maxBlockDimY") {
    const unsigned int y = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimY, 0) + 1u;
    HIP_CHECK_ERROR(hipLaunchKernel(reinterpret_cast<void*>(kernel), dim3{1, 1, 1}, dim3{1, y, 1},
                                    nullptr, 0, nullptr),
                    hipErrorInvalidConfiguration);
  }

  SECTION("blockDim.z > maxBlockDimZ") {
    const unsigned int z = GetDeviceAttribute(hipDeviceAttributeMaxBlockDimZ, 0) + 1u;
    HIP_CHECK_ERROR(hipLaunchKernel(reinterpret_cast<void*>(kernel), dim3{1, 1, 1}, dim3{1, 1, z},
                                    nullptr, 0, nullptr),
                    hipErrorInvalidConfiguration);
  }

  SECTION("blockDim.x * blockDim.y * blockDim.z > maxThreadsPerBlock") {
    const unsigned int max = GetDeviceAttribute(hipDeviceAttributeMaxThreadsPerBlock, 0);
    const unsigned int dim = std::ceil(std::cbrt(max));
    HIP_CHECK_ERROR(hipLaunchKernel(reinterpret_cast<void*>(kernel), dim3{1, 1, 1},
                                    dim3{dim, dim, dim}, nullptr, 0, nullptr),
                    hipErrorInvalidConfiguration);
  }

  SECTION("sharedMemBytes > maxSharedMemoryPerBlock") {
    const unsigned int max = GetDeviceAttribute(hipDeviceAttributeMaxSharedMemoryPerBlock, 0) + 1u;
    HIP_CHECK_ERROR(hipLaunchKernel(reinterpret_cast<void*>(kernel), dim3{1, 1, 1}, dim3{1, 1, 1},
                                    nullptr, max, nullptr),
                    hipErrorInvalidValue);
  }

#if HT_AMD
  SECTION("Invalid stream") {
    hipStream_t stream = reinterpret_cast<hipStream_t>(0xDEADBEEF);
    HIP_CHECK_ERROR(hipLaunchKernel(reinterpret_cast<void*>(kernel), dim3{1, 1, 1}, dim3{1, 1, 1},
                                    nullptr, 0, stream),
                    hipErrorInvalidValue);
  }
#endif
}

HIP_TEST_CASE(Unit_hipLaunchKernel_Verify_Capture) {
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  GENERATE_CAPTURE();
  BEGIN_CAPTURE(stream);
  HIP_CHECK(hipLaunchKernel(reinterpret_cast<void*>(kernel), dim3{1, 1, 1}, dim3{1, 1, 1}, nullptr,
                            0, stream));
  END_CAPTURE(stream);

  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipStreamDestroy(stream));
}
