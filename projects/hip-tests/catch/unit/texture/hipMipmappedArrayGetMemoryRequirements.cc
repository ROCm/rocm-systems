/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_array_common.hh>
#include <hip_test_common.hh>

HIP_TEST_CASE(Unit_hipMipmappedArrayGetMemoryRequirements_Negative_Parameters) {
  CHECK_IMAGE_SUPPORT;

  const int device_id = 0;
  hipArrayMemoryRequirements memoryRequirements{};
  hipmipmappedArray array;
  HIP_ARRAY3D_DESCRIPTOR desc = {};
  using vec_info = vector_info<float>;
  desc.Format = vec_info::format;
  desc.NumChannels = vec_info::size;
  desc.Width = 4;
  desc.Height = 4;
  desc.Depth = 6;
  desc.Flags = 0;

  unsigned int levels = 1 + std::log2(desc.Depth);

  HIP_CHECK(hipFree(0));
  HIP_CHECK(hipMipmappedArrayCreate(&array, &desc, levels));

  SECTION("memoryRequirements is nullptr") {
    HIP_CHECK_ERROR(hipMipmappedArrayGetMemoryRequirements(nullptr, array, device_id), hipErrorInvalidValue);
  }

  SECTION("mipmap is nullptr") {
    HIP_CHECK_ERROR(hipMipmappedArrayGetMemoryRequirements(&memoryRequirements, nullptr, device_id), hipErrorInvalidHandle);
  }
}
