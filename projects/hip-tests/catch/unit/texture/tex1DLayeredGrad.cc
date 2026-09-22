/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#include "kernels.hh"
#include "test_fixture.hh"

/**
 * @addtogroup tex1DLayeredGrad tex1DLayeredGrad
 * @{
 * @ingroup TextureTest
 */

/**
 * Test Description
 * ------------------------
 *    - Test texture fetching with `tex1DLayeredGrad` and read mode set to `hipReadModeElementType`.
 * The test is performed with:
 *      - normalized coordinates
 *      - non-normalized coordinates
 *      - Nearest-point sampling
 *      - Linear filtering
 *      - All combinations of different addressing modes.
 * Test source
 * ------------------------
 *    - unit/texture/tex1DLayeredGrad.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.7
 */
HIP_TEMPLATE_TEST_CASE(Unit_tex1DLayeredGrad_Positive_ReadModeElementType, char, unsigned char,
                   short, unsigned short, int, unsigned int, float) {
  CHECK_IMAGE_SUPPORT;

  TextureTestParams<TestType> params = {};
  params.extent = make_hipExtent(1024, 0, 0);
  params.layers = 2;
  params.num_subdivisions = 4;
  params.GenerateTextureDesc();

  TextureTestFixture<TestType, false, false> fixture{params};

  const auto [num_threads, num_blocks] = GetLaunchConfig(1024, params.NumItersX());

  for (auto layer = 0u; layer < params.layers; ++layer) {
    tex1DLayeredGradKernel<vec4<TestType>><<<num_blocks, num_threads>>>(
        fixture.out_alloc_d.ptr(), params.NumItersX(), fixture.tex.object(), params.Width(),
        params.num_subdivisions, params.tex_desc.normalizedCoords, layer, 0.5f, 0.5f);

    fixture.LoadOutput();

    for (auto i = 0u; i < params.NumItersX(); ++i) {
      float x = GetCoordinate(i, params.NumItersX(), params.Width(), params.num_subdivisions,
                              params.tex_desc.normalizedCoords);
      auto ref_val = fixture.tex_h.Tex1DLayered(x, layer, params.tex_desc);
      if (!fixture.Verify(fixture.out_alloc_h[i], ref_val)) {
        INFO("Layer: " << layer);
        INFO("Index: " << i);
        INFO("Filtering mode: " << FilteringModeToString(params.tex_desc.filterMode));
        INFO("Normalized coordinates: " << std::boolalpha << params.tex_desc.normalizedCoords);
        INFO("Address mode: " << AddressModeToString(params.tex_desc.addressMode[0]));
        INFO("x: " << std::fixed << std::setprecision(16) << x);
        REQUIRE(false);
      }
    }
  }
}

/**
 * Test Description
 * ------------------------
 *    - Test texture fetching with `tex1DLayeredGrad` and read mode set to
 * `hipReadModeNormalizedFloat`. The test is performed with:
 *      - normalized coordinates
 *      - non-normalized coordinates
 *      - Nearest-point sampling
 *      - Linear filtering
 *      - All combinations of different addressing modes.
 * Test source
 * ------------------------
 *    - unit/texture/tex1DLayeredGrad.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.7
 */
HIP_TEMPLATE_TEST_CASE(Unit_tex1DLayeredGrad_Positive_ReadModeNormalizedFloat, char,
                   unsigned char, short, unsigned short) {
  CHECK_IMAGE_SUPPORT;

  TextureTestParams<TestType> params = {};
  params.extent = make_hipExtent(1024, 0, 0);
  params.layers = 2;
  params.num_subdivisions = 4;
  params.GenerateTextureDesc(hipReadModeNormalizedFloat);

  TextureTestFixture<TestType, true, false> fixture{params};

  const auto [num_threads, num_blocks] = GetLaunchConfig(1024, params.NumItersX());

  for (auto layer = 0u; layer < params.layers; ++layer) {
    tex1DLayeredGradKernel<vec4<float>><<<num_blocks, num_threads>>>(
        fixture.out_alloc_d.ptr(), params.NumItersX(), fixture.tex.object(), params.Width(),
        params.num_subdivisions, params.tex_desc.normalizedCoords, layer, 0.5f, 0.5f);

    fixture.LoadOutput();

    for (auto i = 0u; i < params.NumItersX(); ++i) {
      float x = GetCoordinate(i, params.NumItersX(), params.Width(), params.num_subdivisions,
                              params.tex_desc.normalizedCoords);
      auto ref_val = fixture.tex_h.Tex1DLayered(x, layer, params.tex_desc);
      if (!fixture.Verify(fixture.out_alloc_h[i], ref_val)) {
        INFO("Layer: " << layer);
        INFO("i: " << i);
        INFO("Filtering mode: " << FilteringModeToString(params.tex_desc.filterMode));
        INFO("Normalized coordinates: " << std::boolalpha << params.tex_desc.normalizedCoords);
        INFO("Address mode: " << AddressModeToString(params.tex_desc.addressMode[0]));
        INFO("Filter mode: " << FilteringModeToString(params.tex_desc.filterMode));
        INFO("x: " << std::fixed << std::setprecision(16) << x);
        REQUIRE(false);
      }
    }
  }
}

/**
 * End doxygen group TextureTest.
 * @}
 */
