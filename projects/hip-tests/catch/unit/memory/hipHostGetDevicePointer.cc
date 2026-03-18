/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <utils.hh>

TEST_CASE(Unit_hipHostGetDevicePointer_Negative) {
  int* hPtr{nullptr};
  int* dPtr{nullptr};
  HIP_CHECK(hipHostMalloc(&hPtr, sizeof(int)));

  if (!DeviceAttributesSupport(0, hipDeviceAttributeCanMapHostMemory)) {
    HIP_CHECK_ERROR(hipHostGetDevicePointer(reinterpret_cast<void**>(&dPtr), hPtr, 0),
                    hipErrorNotSupported);
    return;
  }

  SECTION("Nullptr as device") {
    HIP_CHECK_ERROR(hipHostGetDevicePointer(nullptr, hPtr, 0), hipErrorInvalidValue);
  }

  SECTION("Nullptr as host") {
    int* dPtr{nullptr};
    HIP_CHECK_ERROR(hipHostGetDevicePointer(reinterpret_cast<void**>(&dPtr), nullptr, 0),
                    hipErrorInvalidValue);
  }

  SECTION("Non pinned memory as host") {
    int* hPtr = reinterpret_cast<int*>(malloc(sizeof(*hPtr)));
    HIP_CHECK_ERROR(hipHostGetDevicePointer(reinterpret_cast<void**>(&dPtr), hPtr, 0),
                    hipErrorInvalidValue);
    free(hPtr);
  }

  SECTION("Flags non-zero") {
    HIP_CHECK_ERROR(hipHostGetDevicePointer(reinterpret_cast<void**>(&dPtr), hPtr, 1),
                    hipErrorInvalidValue);
  }

  HIP_CHECK(hipHostFree(hPtr));
}

template <typename T> __global__ void set(T* ptr, T val) { *ptr = val; }

TEST_CASE(Unit_hipHostGetDevicePointer_UseCase) {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeCanMapHostMemory)) {
    HipTest::HIP_SKIP_TEST("Device does not support mapping host memory");
    return;
  }

  int* hPtr{nullptr};
  HIP_CHECK(hipHostMalloc(&hPtr, sizeof(int)));

  auto kernel = set<int>;
  constexpr int value = 10;

  SECTION("Set the value on device - Get device ptr") {
    int* dPtr{nullptr};
    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&dPtr), hPtr, 0));
    REQUIRE(dPtr != nullptr);

    kernel<<<1, 1>>>(dPtr, value);
    HIP_CHECK(hipDeviceSynchronize());

    REQUIRE(*hPtr == value);
  }

  SECTION("Set the value on device - by hipHostRegister") {
    int res{0};                                        // Stuff on stack
    HIP_CHECK(hipHostRegister(&res, sizeof(int), 0));  // Lets map stack memory :)

    int* dPtr{nullptr};
    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&dPtr), &res, 0))

    kernel<<<1, 1>>>(dPtr, value);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipHostUnregister(&res));

    REQUIRE(res == value);
  }

  HIP_CHECK(hipHostFree(hPtr));
}

TEST_CASE(Unit_hipHostGetDevicePointer_Capture) {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeCanMapHostMemory)) {
    HipTest::HIP_SKIP_TEST("Device does not support mapping host memory");
    return;
  }

  int* host_ptr = nullptr;
  int* device_ptr = nullptr;
  HIP_CHECK(hipHostMalloc(&host_ptr, sizeof(int)));

  hipStream_t stream = nullptr;
  HIP_CHECK(hipStreamCreate(&stream));

  GENERATE_CAPTURE();
  BEGIN_CAPTURE(stream);
  HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&device_ptr), host_ptr, 0));
  END_CAPTURE(stream);

  HIP_CHECK(hipStreamDestroy(stream));
  HIP_CHECK(hipHostFree(host_ptr));
}
