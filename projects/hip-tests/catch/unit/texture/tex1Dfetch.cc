/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <vector>

#include <hip_test_common.hh>
#include <resource_guards.hh>
#include "test_fixture.hh"
#include "kernels.hh"
#include "utils.hh"
#include "vec4.hh"

/**
 * @addtogroup tex1D tex1D
 * @{
 * @ingroup TextureTest
 */

/**
 * Test Description
 * ------------------------
 *    - Test texture fetching with `tex1Dfetch` and read mode set to `hipReadModeElementType`.
 * Test source
 * ------------------------
 *    - unit/texture/tex1Dfetch.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE(Unit_tex1Dfetch_Positive_ReadModeElementType, char, unsigned char, short,
                   unsigned short, int, unsigned int, float) {
  CHECK_IMAGE_SUPPORT;

  std::vector<vec4<TestType>> tex_h(1024);
  for (auto i = 0u; i < tex_h.size(); ++i) {
    tex_h[i].x = i + 7;
    tex_h[i].y = i + 7;
    tex_h[i].z = i + 7;
    tex_h[i].w = i + 7;
  }

  const auto alloc_size = tex_h.size() * sizeof(vec4<TestType>);
  LinearAllocGuard<vec4<TestType>> tex_alloc_d(LinearAllocs::hipMalloc, alloc_size);
  HIP_CHECK(hipMemcpy(tex_alloc_d.ptr(), tex_h.data(), alloc_size, hipMemcpyHostToDevice));

  hipResourceDesc res_desc;
  memset(&res_desc, 0, sizeof(res_desc));
  res_desc.resType = hipResourceTypeLinear;
  res_desc.res.linear.devPtr = tex_alloc_d.ptr();
  res_desc.res.linear.desc = hipCreateChannelDesc<vec4<TestType>>();
  res_desc.res.linear.sizeInBytes = alloc_size;

  hipTextureDesc tex_desc;
  memset(&tex_desc, 0, sizeof(tex_desc));
  tex_desc.filterMode = hipFilterModePoint;
  tex_desc.readMode = hipReadModeElementType;
  tex_desc.normalizedCoords = false;
  tex_desc.addressMode[0] = hipAddressModeClamp;

  LinearAllocGuard<vec4<TestType>> out_alloc_d(LinearAllocs::hipMalloc, alloc_size);
  TextureGuard tex(&res_desc, &tex_desc);

  const auto num_threads = std::min<size_t>(1024, tex_h.size());
  const auto num_blocks = (tex_h.size() + num_threads - 1) / num_threads;
  tex1DfetchKernel<vec4<TestType>>
      <<<num_blocks, num_threads>>>(out_alloc_d.ptr(), tex_h.size(), tex.object());

  std::vector<vec4<TestType>> out_alloc_h(tex_h.size());
  HIP_CHECK(hipMemcpy(out_alloc_h.data(), out_alloc_d.ptr(), alloc_size, hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  for (auto i = 0u; i < out_alloc_h.size(); ++i) {
    const auto ref_val = tex_h[i];
    if (!(out_alloc_h[i] == ref_val)) {
      INFO("Index: " << i);
      REQUIRE(false);
    }
  }
}

/**
 * Test Description
 * ------------------------
 *    - Test texture fetching with `tex1Dfetch` and read mode set to `hipReadModeNormalizedFloat`.
 * Test source
 * ------------------------
 *    - unit/texture/tex1Dfetch.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
TEMPLATE_TEST_CASE(Unit_tex1Dfetch_Positive_ReadModeNormalizedFloat, char, unsigned char,
                   short, unsigned short) {
  CHECK_IMAGE_SUPPORT;

  std::vector<vec4<TestType>> tex_h(1024);
  for (auto i = 0u; i < tex_h.size(); ++i) {
    tex_h[i].x = i + 7;
    tex_h[i].y = i + 7;
    tex_h[i].z = i + 7;
    tex_h[i].w = i + 7;
  }

  const auto alloc_size = tex_h.size() * sizeof(vec4<TestType>);
  LinearAllocGuard<vec4<TestType>> tex_alloc_d(LinearAllocs::hipMalloc, alloc_size);
  HIP_CHECK(hipMemcpy(tex_alloc_d.ptr(), tex_h.data(), alloc_size, hipMemcpyHostToDevice));

  hipResourceDesc res_desc;
  memset(&res_desc, 0, sizeof(res_desc));
  res_desc.resType = hipResourceTypeLinear;
  res_desc.res.linear.devPtr = tex_alloc_d.ptr();
  res_desc.res.linear.desc = hipCreateChannelDesc<vec4<TestType>>();
  res_desc.res.linear.sizeInBytes = alloc_size;

  hipTextureDesc tex_desc;
  memset(&tex_desc, 0, sizeof(tex_desc));
  tex_desc.filterMode = hipFilterModePoint;
  tex_desc.readMode = hipReadModeNormalizedFloat;
  tex_desc.normalizedCoords = false;
  tex_desc.addressMode[0] = hipAddressModeClamp;

  LinearAllocGuard<vec4<float>> out_alloc_d(LinearAllocs::hipMalloc,
                                            tex_h.size() * sizeof(vec4<float>));
  TextureGuard tex(&res_desc, &tex_desc);

  const auto num_threads = std::min<size_t>(1024, tex_h.size());
  const auto num_blocks = (tex_h.size() + num_threads - 1) / num_threads;
  tex1DfetchKernel<vec4<float>>
      <<<num_blocks, num_threads>>>(out_alloc_d.ptr(), tex_h.size(), tex.object());

  std::vector<vec4<float>> out_alloc_h(tex_h.size());
  HIP_CHECK(hipMemcpy(out_alloc_h.data(), out_alloc_d.ptr(), tex_h.size() * sizeof(vec4<float>),
                      hipMemcpyDeviceToHost));
  HIP_CHECK(hipDeviceSynchronize());

  for (auto i = 0u; i < out_alloc_h.size(); ++i) {
    const auto ref_val = Vec4Map(tex_h[i]);
    if (!(out_alloc_h[i] == ref_val)) {
      INFO("Index: " << i);
      REQUIRE(false);
    }
  }
}

/**
 * End doxygen group TextureTest.
 * @}
 */
