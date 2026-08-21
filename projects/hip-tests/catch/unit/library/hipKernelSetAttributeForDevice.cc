/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#include <string>

namespace {
void LoadLibraryKernel(hipLibrary_t* library, hipKernel_t* kernel) {
  std::string code_object = "library_code_load.code";
  HIP_CHECK(hipLibraryLoadFromFile(library, code_object.c_str(), nullptr, nullptr, 0, nullptr,
                                   nullptr, 0));
  HIP_CHECK(hipLibraryGetKernel(kernel, *library, "add_kernel"));
}
}  // namespace

HIP_TEST_CASE(Unit_hipKernelSetAttributeForDevice_Positive_Values) {
  HIP_CHECK(hipSetDevice(0));
  hipLibrary_t library = nullptr;
  hipKernel_t kernel = nullptr;
  LoadLibraryKernel(&library, &kernel);

  int observed = -1;
  HIP_CHECK(hipKernelGetAttribute(&observed, HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                  kernel, 0));
  REQUIRE(observed >= 0);
  HIP_CHECK(hipKernelGetAttribute(&observed, HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                                  kernel, 0));
  REQUIRE(observed == -1);

  HIP_CHECK(hipKernelSetAttributeForDevice(
      kernel, hipFuncAttributeMaxDynamicSharedMemorySize, 0, 0));
  HIP_CHECK(hipKernelGetAttribute(&observed, HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                  kernel, 0));
  REQUIRE(observed == 0);

  for (const int value : {-1, 0, 50, 100}) {
    HIP_CHECK(hipKernelSetAttributeForDevice(
        kernel, hipFuncAttributePreferredSharedMemoryCarveout, value, 0));
    HIP_CHECK(hipKernelGetAttribute(&observed, HIP_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                                    kernel, 0));
    REQUIRE(observed == value);
  }

  HIP_CHECK(hipKernelSetAttributeForDevice(
      kernel, hipFuncAttributeRequiredClusterWidth, 2, 0));
  HIP_CHECK(hipKernelSetAttributeForDevice(
      kernel, hipFuncAttributeRequiredClusterHeight, 1, 0));
  HIP_CHECK(hipKernelSetAttributeForDevice(
      kernel, hipFuncAttributeRequiredClusterDepth, 1, 0));
  HIP_CHECK(hipKernelGetAttribute(&observed, HIP_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH, kernel, 0));
  REQUIRE(observed == 2);

  for (const int value : {-1, 0, 1, 2}) {
    HIP_CHECK(hipKernelSetAttributeForDevice(
        kernel, hipFuncAttributeNonPortableClusterSizeAllowed, value, 0));
    HIP_CHECK(hipKernelGetAttribute(
        &observed, HIP_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, kernel, 0));
    REQUIRE(observed == (value != 0));
  }

  for (const int value : {0, 1, 2}) {
    HIP_CHECK(hipKernelSetAttributeForDevice(
        kernel, hipFuncAttributeClusterSchedulingPolicyPreference, value, 0));
    HIP_CHECK(hipKernelGetAttribute(
        &observed, HIP_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE, kernel, 0));
    REQUIRE(observed == value);
  }
  HIP_CHECK(hipLibraryUnload(library));
}

HIP_TEST_CASE(Unit_hipKernelSetAttributeForDevice_Positive_CrossDevice) {
  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  if (device_count < 2) {
    HIP_SKIP_TEST(HipTest::SkipReason::kFewerThanTwoGpus);
  }

  HIP_CHECK(hipSetDevice(0));
  hipLibrary_t library = nullptr;
  hipKernel_t kernel = nullptr;
  LoadLibraryKernel(&library, &kernel);

  HIP_CHECK(hipKernelSetAttributeForDevice(
      kernel, hipFuncAttributeMaxDynamicSharedMemorySize, 0, 0));
  HIP_CHECK(hipKernelSetAttributeForDevice(
      kernel, hipFuncAttributeMaxDynamicSharedMemorySize, 1024, 1));

  int current_device = -1;
  HIP_CHECK(hipGetDevice(&current_device));
  REQUIRE(current_device == 0);

  int device_0_value = -1;
  int device_1_value = -1;
  HIP_CHECK(hipKernelGetAttribute(&device_0_value,
                                  HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, kernel, 0));
  HIP_CHECK(hipKernelGetAttribute(&device_1_value,
                                  HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, kernel, 1));
  REQUIRE(device_0_value == 0);
  REQUIRE(device_1_value == 1024);
  HIP_CHECK(hipLibraryUnload(library));
}

HIP_TEST_CASE(Unit_hipKernelSetAttributeForDevice_Positive_FunctionOverride) {
  HIP_CHECK(hipSetDevice(0));
  hipLibrary_t library = nullptr;
  hipKernel_t kernel = nullptr;
  LoadLibraryKernel(&library, &kernel);

  HIP_CHECK(hipKernelSetAttributeForDevice(
      kernel, hipFuncAttributeMaxDynamicSharedMemorySize, 1024, 0));

  hipFunction_t function = nullptr;
  HIP_CHECK(hipKernelGetFunction(&function, kernel));
  HIP_CHECK(hipFuncSetAttribute(function, hipFuncAttributeMaxDynamicSharedMemorySize, 0));

  HIP_CHECK(hipKernelSetAttributeForDevice(
      kernel, hipFuncAttributeMaxDynamicSharedMemorySize, 2048, 0));

  int kernel_value = -1;
  int function_value = -1;
  HIP_CHECK(hipKernelGetAttribute(&kernel_value,
                                  HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, kernel, 0));
  HIP_CHECK(hipFuncGetAttribute(
      &function_value, HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, function));
  REQUIRE(kernel_value == 2048);
  REQUIRE(function_value == 0);
  HIP_CHECK(hipLibraryUnload(library));
}

HIP_TEST_CASE(Unit_hipKernelSetAttributeForDevice_Negative_Parameters) {
  HIP_CHECK(hipSetDevice(0));
  hipLibrary_t library = nullptr;
  hipKernel_t kernel = nullptr;
  LoadLibraryKernel(&library, &kernel);

  HIP_CHECK_ERROR(hipKernelSetAttributeForDevice(
                      nullptr, hipFuncAttributeMaxDynamicSharedMemorySize, 0, 0),
                  hipErrorInvalidResourceHandle);

  int device_count = 0;
  HIP_CHECK(hipGetDeviceCount(&device_count));
  HIP_CHECK_ERROR(hipKernelSetAttributeForDevice(
                      kernel, hipFuncAttributeMaxDynamicSharedMemorySize, 0, -1),
                  hipErrorInvalidDevice);
  HIP_CHECK_ERROR(hipKernelSetAttributeForDevice(
                      kernel, hipFuncAttributeMaxDynamicSharedMemorySize, 0, device_count),
                  hipErrorInvalidDevice);

  for (const auto attribute : {static_cast<hipFuncAttribute>(-1),
                               hipFuncAttributeClusterDimMustBeSet, hipFuncAttributeMax}) {
    HIP_CHECK_ERROR(hipKernelSetAttributeForDevice(kernel, attribute, 0, 0),
                    hipErrorInvalidValue);
  }

  HIP_CHECK_ERROR(hipKernelSetAttributeForDevice(
                      kernel, hipFuncAttributeMaxDynamicSharedMemorySize, -1, 0),
                  hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipKernelSetAttributeForDevice(
                      kernel, hipFuncAttributePreferredSharedMemoryCarveout, -2, 0),
                  hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipKernelSetAttributeForDevice(
                      kernel, hipFuncAttributePreferredSharedMemoryCarveout, 101, 0),
                  hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipKernelSetAttributeForDevice(
                      kernel, hipFuncAttributeRequiredClusterWidth, -1, 0),
                  hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipKernelSetAttributeForDevice(
                      kernel, hipFuncAttributeClusterSchedulingPolicyPreference, -1, 0),
                  hipErrorInvalidValue);
  HIP_CHECK_ERROR(hipKernelSetAttributeForDevice(
                      kernel, hipFuncAttributeClusterSchedulingPolicyPreference, 3, 0),
                  hipErrorInvalidValue);
  HIP_CHECK(hipLibraryUnload(library));
}
