/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipArrayGetMemoryRequirements hipArrayGetMemoryRequirements
 * @{
 * @ingroup MemoryTest
 * `hipArrayGetMemoryRequirements(hipArrayMemoryRequirements* memoryRequirements, hipArray_t array,
 * hipDevice_t device)` - Returns the memory requirements of a HIP array.
 */

#include <hip_test_common.hh>
#include <resource_guards.hh>

#include <algorithm>

namespace {
size_t LogicalSizeInBytes(const hipExtent& extent, const size_t element_size) {
  return extent.width * std::max<size_t>(extent.height, 1) * std::max<size_t>(extent.depth, 1) *
         element_size;
}

// Queries the memory requirements of the array and checks the properties that hold for any array:
// the size covers at least the logical contents, the alignment is a non-zero power of two, and
// repeated queries return the same result.
hipArrayMemoryRequirements QueryMemoryRequirements(const hipArray_t array, const int device,
                                                   const size_t min_size) {
  hipArrayMemoryRequirements requirements{};
  HIP_CHECK(hipArrayGetMemoryRequirements(&requirements, array, device));
  REQUIRE(requirements.size >= min_size);
  REQUIRE(requirements.alignment > 0);
  REQUIRE((requirements.alignment & (requirements.alignment - 1)) == 0);

  hipArrayMemoryRequirements requirements_again{};
  HIP_CHECK(hipArrayGetMemoryRequirements(&requirements_again, array, device));
  REQUIRE(requirements_again.size == requirements.size);
  REQUIRE(requirements_again.alignment == requirements.alignment);

  return requirements;
}
}  // namespace

/**
 * Test Description
 * ------------------------
 *  - Queries the memory requirements of 1D, 2D, 3D and layered arrays created through both the
 *    runtime (hipMalloc3DArray) and driver (hipArray3DCreate) APIs, for several element types.
 * Test source
 * ------------------------
 *  - unit/memory/hipArrayGetMemoryRequirements.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.18
 */
HIP_TEMPLATE_TEST_CASE(Unit_hipArrayGetMemoryRequirements_Positive_Basic, char, uchar2, short4,
                       float, float4) {
  CHECK_IMAGE_SUPPORT

  int device = 0;
  HIP_CHECK(hipGetDevice(&device));

  const bool use_driver_api = GENERATE(false, true);

  constexpr size_t kSize = 64;
  constexpr size_t kLayers = 4;
  hipExtent extent{};
  unsigned int flags = hipArrayDefault;

  SECTION("1D") { extent = make_hipExtent(kSize, 0, 0); }
  SECTION("2D") { extent = make_hipExtent(kSize, kSize, 0); }
  SECTION("3D") { extent = make_hipExtent(kSize, kSize, kSize); }
  SECTION("1D layered") {
    extent = make_hipExtent(kSize, 0, kLayers);
    flags = hipArrayLayered;
  }
  SECTION("2D layered") {
    extent = make_hipExtent(kSize, kSize, kLayers);
    flags = hipArrayLayered;
  }

  CAPTURE(use_driver_api, extent.width, extent.height, extent.depth, flags);

  const size_t min_size = LogicalSizeInBytes(extent, sizeof(TestType));
  if (use_driver_api) {
    // DrvArrayAllocGuard takes the width in bytes.
    DrvArrayAllocGuard<TestType> array(
        make_hipExtent(extent.width * sizeof(TestType), extent.height, extent.depth), flags);
    QueryMemoryRequirements(array.ptr(), device, min_size);
  } else {
    ArrayAllocGuard<TestType> array(extent, flags);
    QueryMemoryRequirements(array.ptr(), device, min_size);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Checks that a larger array reports a larger size than a smaller array of the same format.
 * Test source
 * ------------------------
 *  - unit/memory/hipArrayGetMemoryRequirements.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipArrayGetMemoryRequirements_Positive_SizeScalesWithExtent) {
  CHECK_IMAGE_SUPPORT

  int device = 0;
  HIP_CHECK(hipGetDevice(&device));

  const hipExtent small_extent = make_hipExtent(64, 64, 0);
  const hipExtent large_extent = make_hipExtent(128, 128, 0);
  ArrayAllocGuard<float> small_array(small_extent);
  ArrayAllocGuard<float> large_array(large_extent);

  const auto small_requirements = QueryMemoryRequirements(
      small_array.ptr(), device, LogicalSizeInBytes(small_extent, sizeof(float)));
  const auto large_requirements = QueryMemoryRequirements(
      large_array.ptr(), device, LogicalSizeInBytes(large_extent, sizeof(float)));

  REQUIRE(large_requirements.size > small_requirements.size);
}

/**
 * Test Description
 * ------------------------
 *  - Negative parameters test for `hipArrayGetMemoryRequirements`.
 * Test source
 * ------------------------
 *  - unit/memory/hipArrayGetMemoryRequirements.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipArrayGetMemoryRequirements_Negative_Parameters) {
  CHECK_IMAGE_SUPPORT

  int device = 0;
  HIP_CHECK(hipGetDevice(&device));

  ArrayAllocGuard<float> array(make_hipExtent(64, 64, 0));
  hipArrayMemoryRequirements requirements{};

  SECTION("memoryRequirements is nullptr") {
    HIP_CHECK_ERROR(hipArrayGetMemoryRequirements(nullptr, array.ptr(), device),
                    hipErrorInvalidValue);
  }

  SECTION("array is nullptr") {
    HIP_CHECK_ERROR(hipArrayGetMemoryRequirements(&requirements, nullptr, device),
                    hipErrorInvalidHandle);
  }

  SECTION("array is freed") {
    hipArray_t freed_array = nullptr;
    hipChannelFormatDesc desc = hipCreateChannelDesc<float>();
    HIP_CHECK(hipMallocArray(&freed_array, &desc, 64, 64));
    HIP_CHECK(hipFreeArray(freed_array));
    HIP_CHECK_ERROR(hipArrayGetMemoryRequirements(&requirements, freed_array, device),
                    hipErrorInvalidHandle);
  }

  SECTION("array is a mipmap level") {
    // A level array owns no memory; its mipmapped array must be queried instead.
    MipmappedArrayAllocGuard<float> mipmap(make_hipExtent(64, 64, 0), 3, hipArrayDefault);
    HIP_CHECK_ERROR(hipArrayGetMemoryRequirements(&requirements, mipmap.GetLevel(0), device),
                    hipErrorInvalidValue);
  }

  SECTION("device ordinal is negative") {
    HIP_CHECK_ERROR(hipArrayGetMemoryRequirements(&requirements, array.ptr(), -1),
                    hipErrorInvalidDevice);
  }

  SECTION("device ordinal is out of range") {
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    HIP_CHECK_ERROR(hipArrayGetMemoryRequirements(&requirements, array.ptr(), device_count),
                    hipErrorInvalidDevice);
  }
}

/**
 * Test Description
 * ------------------------
 *  - Gets `hipArrayGetMemoryRequirements` through hipGetProcAddress and checks that calling it
 *    through the function pointer gives the same result as calling it directly.
 * Test source
 * ------------------------
 *  - unit/memory/hipArrayGetMemoryRequirements.cc
 * Test requirements
 * ------------------------
 *  - HIP_VERSION >= 7.18
 */
HIP_TEST_CASE(Unit_hipArrayGetMemoryRequirements_Positive_GetProcAddress) {
  CHECK_IMAGE_SUPPORT

  int device = 0;
  HIP_CHECK(hipGetDevice(&device));

  int current_hip_version = 0;
  HIP_CHECK(hipRuntimeGetVersion(&current_hip_version));

  void* fn_ptr = nullptr;
  HIP_CHECK(hipGetProcAddress("hipArrayGetMemoryRequirements", &fn_ptr, current_hip_version, 0,
                              nullptr));
  REQUIRE(fn_ptr != nullptr);

  auto dyn_hipArrayGetMemoryRequirements =
      reinterpret_cast<hipError_t (*)(hipArrayMemoryRequirements*, hipArray_t, hipDevice_t)>(
          fn_ptr);

  ArrayAllocGuard<float> array(make_hipExtent(64, 64, 0));

  hipArrayMemoryRequirements requirements{};
  hipArrayMemoryRequirements dyn_requirements{};
  HIP_CHECK(hipArrayGetMemoryRequirements(&requirements, array.ptr(), device));
  HIP_CHECK(dyn_hipArrayGetMemoryRequirements(&dyn_requirements, array.ptr(), device));

  REQUIRE(dyn_requirements.size == requirements.size);
  REQUIRE(dyn_requirements.alignment == requirements.alignment);
}

/**
 * End doxygen group MemoryTest.
 * @}
 */
