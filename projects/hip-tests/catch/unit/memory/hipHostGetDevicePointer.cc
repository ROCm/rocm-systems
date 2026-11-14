/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <utils.hh>

HIP_TEST_CASE(Unit_hipHostGetDevicePointer_Negative) {
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

HIP_TEST_CASE(Unit_hipHostGetDevicePointer_UseCase) {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeCanMapHostMemory)) {
    HIP_SKIP_TEST(HipTest::SkipReason::kHostPinnedMemoryUnsupported);
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

// An interior host pointer must map to a device pointer with the same offset.
HIP_TEST_CASE(Unit_hipHostGetDevicePointer_InteriorPointer) {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeCanMapHostMemory)) {
    HIP_SKIP_TEST(HipTest::SkipReason::kHostPinnedMemoryUnsupported);
  }

  constexpr size_t kSize = 1u << 16;
  constexpr size_t kOffsets[] = {sizeof(int), 4096, kSize - sizeof(int)};
  constexpr int kSentinel = 0x5eed;

  auto kernel = set<int>;

  auto checkInterior = [&](char* base, void* devBase, size_t offset) {
    int* dPtr{nullptr};
    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&dPtr), base + offset, 0));
    REQUIRE(dPtr != nullptr);
    REQUIRE(reinterpret_cast<char*>(dPtr) - reinterpret_cast<char*>(devBase) ==
            static_cast<ptrdiff_t>(offset));

    const int before = *reinterpret_cast<int*>(base);

    kernel<<<1, 1>>>(dPtr, kSentinel);
    HIP_CHECK(hipDeviceSynchronize());

    REQUIRE(*reinterpret_cast<int*>(base + offset) == kSentinel);
    // A runtime that dropped the offset would have written to the base instead.
    REQUIRE(*reinterpret_cast<int*>(base) == before);
  };

  SECTION("hipHostMalloc") {
    char* base{nullptr};
    HIP_CHECK(hipHostMalloc(reinterpret_cast<void**>(&base), kSize, hipHostMallocMapped));
    memset(base, 0, kSize);

    void* devBase{nullptr};
    HIP_CHECK(hipHostGetDevicePointer(&devBase, base, 0));

    for (size_t offset : kOffsets) {
      checkInterior(base, devBase, offset);
    }

    HIP_CHECK(hipHostFree(base));
  }

  SECTION("hipHostRegister") {
    char* base = reinterpret_cast<char*>(malloc(kSize));
    REQUIRE(base != nullptr);
    memset(base, 0, kSize);
    HIP_CHECK(hipHostRegister(base, kSize, hipHostRegisterMapped));

    // Query an interior pointer before the base, as an application laying out fields does.
    int* dInterior{nullptr};
    HIP_CHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&dInterior), base + kOffsets[0], 0));
    REQUIRE(dInterior != nullptr);

    void* devBase{nullptr};
    HIP_CHECK(hipHostGetDevicePointer(&devBase, base, 0));
    REQUIRE(reinterpret_cast<char*>(dInterior) - reinterpret_cast<char*>(devBase) ==
            static_cast<ptrdiff_t>(kOffsets[0]));

    for (size_t offset : kOffsets) {
      checkInterior(base, devBase, offset);
    }

    HIP_CHECK(hipHostUnregister(base));
    free(base);
  }
}

HIP_TEST_CASE(Unit_hipHostGetDevicePointer_Capture) {
  if (!DeviceAttributesSupport(0, hipDeviceAttributeCanMapHostMemory)) {
    HIP_SKIP_TEST(HipTest::SkipReason::kHostPinnedMemoryUnsupported);
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
