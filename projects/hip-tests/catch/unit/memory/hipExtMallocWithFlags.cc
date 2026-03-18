/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <hip/hip_runtime_api.h>
#include <utils.hh>

TEST_CASE(Unit_hipExtMallocWithFlags_Positive_Basic) {
  void* ptr = nullptr;

  SECTION("hipDeviceMallocDefault") {
    const auto alloc_size =
        GENERATE_COPY(10, kPageSize / 2, kPageSize, kPageSize * 3 / 2, kPageSize * 2);
    HIP_CHECK(hipExtMallocWithFlags(&ptr, alloc_size, hipDeviceMallocDefault));
    CHECK(ptr != nullptr);
    CHECK(reinterpret_cast<intptr_t>(ptr) % 256 == 0);
    HIP_CHECK(hipFree(ptr));
  }

  SECTION("hipDeviceMallocFinegrained") {
    if (!DeviceAttributesSupport(0, hipDeviceAttributeFineGrainSupport)) {
      HipTest::HIP_SKIP_TEST("Device does not support fine-grained memory allocations");
      return;
    }
    const auto alloc_size =
        GENERATE_COPY(10, kPageSize / 2, kPageSize, kPageSize * 3 / 2, kPageSize * 2);
    HIP_CHECK(hipExtMallocWithFlags(&ptr, alloc_size, hipDeviceMallocFinegrained));
    CHECK(ptr != nullptr);
    CHECK(reinterpret_cast<intptr_t>(ptr) % 256 == 0);
    HIP_CHECK(hipFree(ptr));
  }

  SECTION("hipMallocSignalMemory") {
    HIP_CHECK(hipExtMallocWithFlags(&ptr, 8, hipMallocSignalMemory));
    CHECK(ptr != nullptr);
    HIP_CHECK(hipFree(ptr));
  }
}

TEST_CASE(Unit_hipExtMallocWithFlags_Positive_Zero_Size) {
  void* ptr = reinterpret_cast<void*>(0x1);
  const auto flag = GENERATE(hipDeviceMallocDefault, hipDeviceMallocFinegrained);
  HIP_CHECK(hipExtMallocWithFlags(&ptr, 0, flag));
  REQUIRE(ptr == nullptr);
}

TEST_CASE(Unit_hipExtMallocWithFlags_Positive_Alignment) {
  void *ptr1 = nullptr, *ptr2 = nullptr;
  const auto flag = GENERATE(hipDeviceMallocDefault, hipDeviceMallocFinegrained);
  if (flag == hipDeviceMallocFinegrained &&
      !DeviceAttributesSupport(0, hipDeviceAttributeFineGrainSupport)) {
    HipTest::HIP_SKIP_TEST("Device does not support fine-grained memory allocations");
    return;
  }
  HIP_CHECK(hipExtMallocWithFlags(&ptr1, 1, flag));
  HIP_CHECK(hipExtMallocWithFlags(&ptr2, 10, flag));
  CHECK(reinterpret_cast<intptr_t>(ptr1) % 256 == 0);
  CHECK(reinterpret_cast<intptr_t>(ptr2) % 256 == 0);
  HIP_CHECK(hipFree(ptr1));
  HIP_CHECK(hipFree(ptr2));
}

TEST_CASE(Unit_hipExtMallocWithFlags_Negative_Parameters) {
  SECTION("Invalid flags") {
    void* ptr = nullptr;
    HIP_CHECK_ERROR(
        hipExtMallocWithFlags(&ptr, 4096, hipDeviceMallocDefault | hipMallocSignalMemory),
        hipErrorInvalidValue);
  }

  SECTION("hipDeviceMallocDefault") {
    SECTION("ptr == nullptr") {
      HIP_CHECK_ERROR(hipExtMallocWithFlags(nullptr, 4096, hipDeviceMallocDefault),
                      hipErrorInvalidValue);
    }

    SECTION("size == max size_t") {
      void* ptr;
      HIP_CHECK_ERROR(
          hipExtMallocWithFlags(&ptr, std::numeric_limits<size_t>::max(), hipDeviceMallocDefault),
          hipErrorOutOfMemory);
    }
  }

  SECTION("hipDeviceMallocFinegrained") {
    SECTION("ptr == nullptr") {
      HIP_CHECK_ERROR(hipExtMallocWithFlags(nullptr, 4096, hipDeviceMallocFinegrained),
                      hipErrorInvalidValue);
    }

    SECTION("size == max size_t") {
      void* ptr;
      HIP_CHECK_ERROR(hipExtMallocWithFlags(&ptr, std::numeric_limits<size_t>::max(),
                                            hipDeviceMallocFinegrained),
                      hipErrorOutOfMemory);
    }
  }

  SECTION("hipMallocSignalMemory") {
    SECTION("ptr == nullptr") {
      HIP_CHECK_ERROR(hipExtMallocWithFlags(nullptr, 4096, hipMallocSignalMemory),
                      hipErrorInvalidValue);
    }

    SECTION("size == 0") {
      void* ptr;
      HIP_CHECK_ERROR(hipExtMallocWithFlags(&ptr, 0, hipMallocSignalMemory), hipErrorInvalidValue);
    }

    SECTION("size != 8") {
      void* ptr;
      HIP_CHECK_ERROR(hipExtMallocWithFlags(&ptr, 16, hipMallocSignalMemory), hipErrorInvalidValue);
    }
  }
}