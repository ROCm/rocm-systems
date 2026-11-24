/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_array_common.hh>
#include <hip_test_common.hh>

#include <algorithm>

#if HT_AMD
namespace {
// Creates a mipmapped array and frees it on scope exit. Skips the test if the device does not
// support mipmapped arrays.
class ScopedMipmappedArray {
 public:
  ScopedMipmappedArray(const hipChannelFormatDesc& desc, const hipExtent extent,
                       const unsigned int levels) {
    const hipError_t status =
        hipMallocMipmappedArray(&array_, &desc, extent, levels, hipArrayDefault);
    if (status == hipErrorNotSupported) {
      (void)hipGetLastError();
      HIP_SKIP_TEST("Mipmapped arrays are not supported by this device.");
    }
    HIP_CHECK(status);
  }

  ~ScopedMipmappedArray() { static_cast<void>(hipFreeMipmappedArray(array_)); }

  ScopedMipmappedArray(const ScopedMipmappedArray&) = delete;
  ScopedMipmappedArray& operator=(const ScopedMipmappedArray&) = delete;

  hipMipmappedArray_t get() const { return array_; }

 private:
  hipMipmappedArray_t array_ = nullptr;
};

// Size in bytes of all mip levels, without any padding the device adds.
size_t LogicalChainBytes(const hipExtent& extent, const unsigned int levels,
                         const size_t element_size) {
  size_t total = 0;
  for (unsigned int level = 0; level < levels; ++level) {
    const size_t width = std::max<size_t>(extent.width >> level, 1);
    const size_t height = std::max<size_t>(extent.height >> level, 1);
    const size_t depth = std::max<size_t>(extent.depth >> level, 1);
    total += width * height * depth * element_size;
  }
  return total;
}
}  // namespace
#endif

/**
 * Queries the memory requirements of 2D and 3D mipmapped arrays with one or more levels. The size
 * must cover all mip levels, and the alignment must be a non-zero power of two. This test is
 * AMD-only: CUDA only accepts mipmapped arrays created with deferred mapping, which HIP does not
 * have.
 */
HIP_TEMPLATE_TEST_CASE(Unit_hipMipmappedArrayGetMemoryRequirements_Positive_Basic, char, uchar2,
                       float, float4) {
  CHECK_IMAGE_SUPPORT
#if HT_AMD
  int device = 0;
  HIP_CHECK(hipGetDevice(&device));

  hipExtent extent{};
  unsigned int levels = 1;

  SECTION("2D, one level") {
    extent = make_hipExtent(64, 64, 0);
    levels = 1;
  }
  SECTION("2D, three levels") {
    extent = make_hipExtent(64, 64, 0);
    levels = 3;
  }
  SECTION("3D, three levels") {
    extent = make_hipExtent(16, 16, 16);
    levels = 3;
  }

  CAPTURE(extent.width, extent.height, extent.depth, levels);

  ScopedMipmappedArray mipmap(hipCreateChannelDesc<TestType>(), extent, levels);

  hipArrayMemoryRequirements requirements{};
  HIP_CHECK(hipMipmappedArrayGetMemoryRequirements(&requirements, mipmap.get(), device));
  REQUIRE(requirements.size >= LogicalChainBytes(extent, levels, sizeof(TestType)));
  REQUIRE(requirements.alignment > 0);
  REQUIRE((requirements.alignment & (requirements.alignment - 1)) == 0);

  hipArrayMemoryRequirements requirements_again{};
  HIP_CHECK(hipMipmappedArrayGetMemoryRequirements(&requirements_again, mipmap.get(), device));
  REQUIRE(requirements_again.size == requirements.size);
  REQUIRE(requirements_again.alignment == requirements.alignment);
#else
  HIP_SKIP_TEST("cuMipmappedArrayGetMemoryRequirements only accepts deferred-mapping arrays");
#endif
}

HIP_TEST_CASE(Unit_hipMipmappedArrayGetMemoryRequirements_Negative_Parameters) {
  CHECK_IMAGE_SUPPORT;

  const int device_id = 0;
  hipArrayMemoryRequirements memoryRequirements{};
  hipMipmappedArray_t array;
  unsigned int levels = 1 + std::log2(6);

  HIP_CHECK(hipFree(0));
#if HT_AMD
  HIP_ARRAY3D_DESCRIPTOR desc = {};
  using vec_info = vector_info<float>;
  desc.Format = vec_info::format;
  desc.NumChannels = vec_info::size;
  desc.Width = 4;
  desc.Height = 4;
  desc.Depth = 6;
  desc.Flags = 0;
  HIP_CHECK(hipMipmappedArrayCreate(&array, &desc, levels));
#elif HT_NVIDIA
  hipChannelFormatDesc desc = hipCreateChannelDesc<float>();
  hipExtent extent = make_hipExtent(4, 4, 6);
  HIP_CHECK(hipMallocMipmappedArray(&array, &desc, extent, levels));
#endif

  SECTION("memoryRequirements is nullptr") {
    HIP_CHECK_ERROR(hipMipmappedArrayGetMemoryRequirements(nullptr, array, device_id), hipErrorInvalidValue);
  }

  SECTION("mipmap is nullptr") {
    HIP_CHECK_ERROR(hipMipmappedArrayGetMemoryRequirements(&memoryRequirements, nullptr, device_id), hipErrorInvalidHandle);
  }

#if HT_AMD
  SECTION("device ordinal is negative") {
    HIP_CHECK_ERROR(hipMipmappedArrayGetMemoryRequirements(&memoryRequirements, array, -1),
                    hipErrorInvalidDevice);
  }

  SECTION("device ordinal is out of range") {
    int device_count = 0;
    HIP_CHECK(hipGetDeviceCount(&device_count));
    HIP_CHECK_ERROR(
        hipMipmappedArrayGetMemoryRequirements(&memoryRequirements, array, device_count),
        hipErrorInvalidDevice);
  }
#endif

#if HT_AMD
  HIP_CHECK(hipMipmappedArrayDestroy(array));
#elif HT_NVIDIA
  HIP_CHECK(hipFreeMipmappedArray(array));
#endif
}
