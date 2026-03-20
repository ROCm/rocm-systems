/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>

#include "kernels.hh"
#include "test_fixture.hh"

/**
 * @addtogroup tex2DLayeredLod tex2DLayeredLod
 * @{
 * @ingroup TextureTest
 */

/**
 * Test Description
 * ------------------------
 *    - Test texture fetching with `tex2DLayeredLod` and read mode set to `hipReadModeElementType`.
 * The test is performed with:
 *      - normalized coordinates
 *      - non-normalized coordinates
 *      - Nearest-point sampling
 *      - Linear filtering
 *      - All combinations of different addressing modes.
 * Test source
 * ------------------------
 *    - unit/texture/tex2DLayeredLod.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.7
 */
HIP_TEMPLATE_TEST_CASE(Unit_tex2DLayeredLod_Positive_ReadModeElementType, char, unsigned char,
                   short, unsigned short, int, unsigned int, float) {
  CHECK_IMAGE_SUPPORT;

  TextureTestParams<TestType> params = {};
  params.extent = make_hipExtent(16, 4, 0);
  params.layers = 2;
  params.num_subdivisions = 4;
  params.GenerateTextureDesc(hipReadModeElementType, true);

  TextureTestFixture<TestType, false, true> fixture{params};

  const auto [num_threads_x, num_blocks_x] = GetLaunchConfig(32, params.NumItersX());
  const auto [num_threads_y, num_blocks_y] = GetLaunchConfig(32, params.NumItersY());

  dim3 dim_grid;
  dim_grid.x = num_blocks_x;
  dim_grid.y = num_blocks_y;

  dim3 dim_block;
  dim_block.x = num_threads_x;
  dim_block.y = num_threads_y;

  for (auto layer = 0u; layer < params.layers; ++layer) {
    tex2DLayeredLodKernel<vec4<TestType>><<<dim_grid, dim_block>>>(
        fixture.out_alloc_d.ptr(), params.NumItersX(), params.NumItersY(), fixture.tex.object(),
        params.Width(), params.Height(), params.num_subdivisions, params.tex_desc.normalizedCoords,
        layer, 0);
    HIP_CHECK(hipGetLastError());

    fixture.LoadOutput();

    for (auto i = 0u; i < params.NumItersX() * params.NumItersY(); ++i) {
      float x = i % params.NumItersX();
      float y = i / params.NumItersX();

      x = GetCoordinate(x, params.NumItersX(), params.Width(), params.num_subdivisions,
                        params.tex_desc.normalizedCoords);
      y = GetCoordinate(y, params.NumItersY(), params.Height(), params.num_subdivisions,
                        params.tex_desc.normalizedCoords);
      const auto ref_val = fixture.tex_h.Tex2DLayered(x, y, layer, params.tex_desc);
      if (!fixture.Verify(fixture.out_alloc_h[i], ref_val)) {
        INFO("Layer: " << layer);
        INFO("Filtering mode: " << FilteringModeToString(params.tex_desc.filterMode));
        INFO("Normalized coordinates: " << std::boolalpha << params.tex_desc.normalizedCoords);
        INFO("Address mode X: " << AddressModeToString(params.tex_desc.addressMode[0]));
        INFO("Address mode Y: " << AddressModeToString(params.tex_desc.addressMode[1]));
        INFO("x: " << std::fixed << std::setprecision(16) << x);
        INFO("y: " << std::fixed << std::setprecision(16) << y);
        REQUIRE(false);
      }
    }
  }
}

/**
 * Test Description
 * ------------------------
 *    - Test texture fetching with `tex2DLayeredLod` and read mode set to
 * `hipReadModeNormalizedFloat`. The test is performed with:
 *      - normalized coordinates
 *      - non-normalized coordinates
 *      - Nearest-point sampling
 *      - Linear filtering
 *      - All combinations of different addressing modes.
 * Test source
 * ------------------------
 *    - unit/texture/tex2DLayeredLod.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.7
 */
HIP_TEMPLATE_TEST_CASE(Unit_tex2DLayeredLod_Positive_ReadModeNormalizedFloat, char, unsigned char,
                   short, unsigned short) {
  CHECK_IMAGE_SUPPORT;

  TextureTestParams<TestType> params = {};
  params.extent = make_hipExtent(16, 4, 0);
  params.layers = 2;
  params.num_subdivisions = 4;
  params.GenerateTextureDesc(hipReadModeNormalizedFloat, true);

  TextureTestFixture<TestType, true, true> fixture{params};

  const auto [num_threads_x, num_blocks_x] = GetLaunchConfig(32, params.NumItersX());
  const auto [num_threads_y, num_blocks_y] = GetLaunchConfig(32, params.NumItersY());

  dim3 dim_grid;
  dim_grid.x = num_blocks_x;
  dim_grid.y = num_blocks_y;

  dim3 dim_block;
  dim_block.x = num_threads_x;
  dim_block.y = num_threads_y;

  for (auto layer = 0u; layer < params.layers; ++layer) {
    tex2DLayeredLodKernel<vec4<float>><<<dim_grid, dim_block>>>(
        fixture.out_alloc_d.ptr(), params.NumItersX(), params.NumItersY(), fixture.tex.object(),
        params.Width(), params.Height(), params.num_subdivisions, params.tex_desc.normalizedCoords,
        layer, 0);
    HIP_CHECK(hipGetLastError());

    fixture.LoadOutput();

    for (auto i = 0u; i < params.NumItersX() * params.NumItersY(); ++i) {
      float x = i % params.NumItersX();
      float y = i / params.NumItersX();

      x = GetCoordinate(x, params.NumItersX(), params.Width(), params.num_subdivisions,
                        params.tex_desc.normalizedCoords);
      y = GetCoordinate(y, params.NumItersY(), params.Height(), params.num_subdivisions,
                        params.tex_desc.normalizedCoords);
      auto ref_val = fixture.tex_h.Tex2DLayered(x, y, layer, params.tex_desc);
      if (!fixture.Verify(fixture.out_alloc_h[i], ref_val)) {
        INFO("Layer: " << layer);
        INFO("Filtering mode: " << FilteringModeToString(params.tex_desc.filterMode));
        INFO("Normalized coordinates: " << std::boolalpha << params.tex_desc.normalizedCoords);
        INFO("Address mode X: " << AddressModeToString(params.tex_desc.addressMode[0]));
        INFO("Address mode Y: " << AddressModeToString(params.tex_desc.addressMode[1]));
        INFO("x: " << std::fixed << std::setprecision(16) << x);
        INFO("y: " << std::fixed << std::setprecision(16) << y);
        REQUIRE(false);
      }
    }
  }
}

/**
 * End doxygen group TextureTest.
 * @}
 */
