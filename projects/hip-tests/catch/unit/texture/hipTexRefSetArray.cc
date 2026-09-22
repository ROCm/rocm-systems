/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#if defined(__HIP_PLATFORM_AMD__) || CUDA_VERSION < CUDA_12000

texture<float, 1, hipReadModeElementType> tex;

HIP_TEST_CASE(Unit_hipTexRefSetArray_Positive) {
  CHECK_IMAGE_SUPPORT
  hipArray_t array_set = nullptr;
  hipArray_t array_get = nullptr;
  hipModule_t module = nullptr;
  hipTexRef tex_ref = nullptr;
  HIP_ARRAY_DESCRIPTOR array_desc;

  array_desc.Format = HIP_AD_FORMAT_FLOAT;
  array_desc.NumChannels = 1;
  array_desc.Width = 16;
  array_desc.Height = 16;

  HIP_CHECK(hipFree(nullptr));
  HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
  HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));
  HIP_CHECK(hipArrayCreate(&array_set, &array_desc));
  HIP_CHECK(hipTexRefSetArray(tex_ref, array_set, HIP_TRSA_OVERRIDE_FORMAT));
  HIP_CHECK(hipTexRefGetArray(&array_get, tex_ref));
  REQUIRE(array_get == array_set);
  HIP_CHECK(hipArrayDestroy(array_set));
  HIP_CHECK(hipModuleUnload(module));
}

HIP_TEST_CASE(Unit_hipTexRefSetArray_CheckData) {
  CHECK_IMAGE_SUPPORT
  hipModule_t module = nullptr;
  hipTexRef tex_ref = nullptr;
  int num_channels = 0;
  hipArray_Format format;
  hipArray_t array = nullptr;
  HIP_ARRAY_DESCRIPTOR array_desc;

  array_desc.Format = GENERATE(HIP_AD_FORMAT_UNSIGNED_INT32, HIP_AD_FORMAT_SIGNED_INT32);
  array_desc.NumChannels = GENERATE(1, 2, 4);
  array_desc.Width = 16;
  array_desc.Height = 16;

  HIP_CHECK(hipFree(nullptr));
  HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
  HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));

  HIP_CHECK(hipArrayCreate(&array, &array_desc));
  HIP_CHECK(hipTexRefSetArray(tex_ref, array, HIP_TRSA_OVERRIDE_FORMAT));

  HIP_CHECK(hipTexRefGetFormat(&format, &num_channels, tex_ref));
  REQUIRE(format == array_desc.Format);
  REQUIRE(num_channels == array_desc.NumChannels);

  HIP_CHECK(hipFreeArray(array));
  HIP_CHECK(hipModuleUnload(module));
}

HIP_TEST_CASE(Unit_hipTexRefSetArray_Negative) {
  CHECK_IMAGE_SUPPORT
  hipArray_t array_set = nullptr;
  hipModule_t module = nullptr;
  hipTexRef tex_ref = nullptr;
  HIP_ARRAY_DESCRIPTOR array_desc;

  array_desc.Format = HIP_AD_FORMAT_FLOAT;
  array_desc.NumChannels = 1;
  array_desc.Width = 16;
  array_desc.Height = 16;

  HIP_CHECK(hipFree(nullptr));
  HIP_CHECK(hipModuleLoad(&module, "tex_ref_get_module.code"));
  HIP_CHECK(hipModuleGetTexRef(&tex_ref, module, "tex"));
  HIP_CHECK(hipArrayCreate(&array_set, &array_desc));

  SECTION("array is null") {
#if HT_AMD
    HIP_CHECK_ERROR(hipTexRefSetArray(tex_ref, nullptr, HIP_TRSA_OVERRIDE_FORMAT),
                    hipErrorInvalidValue);
#else
    HIP_CHECK_ERROR(hipTexRefSetArray(tex_ref, nullptr, HIP_TRSA_OVERRIDE_FORMAT),
                    hipErrorInvalidResourceHandle);
#endif
  }

  SECTION("texture reference is null") {
#if HT_AMD
    HIP_CHECK_ERROR(hipTexRefSetArray(nullptr, array_set, HIP_TRSA_OVERRIDE_FORMAT),
                    hipErrorInvalidValue);
#else
    HIP_CHECK_ERROR(hipTexRefSetArray(nullptr, array_set, HIP_TRSA_OVERRIDE_FORMAT),
                    hipErrorInvalidResourceHandle);
#endif
  }

  HIP_CHECK(hipArrayDestroy(array_set));
  HIP_CHECK(hipModuleUnload(module));
}

#endif
