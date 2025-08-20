/*
Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip_test_common.hh>

#include "kernels.hh"
#include "test_fixture.hh"

/**
 * @addtogroup tex2DGrad tex2DGrad
 * @{
 * @ingroup TextureTest
 */

/**
 * Test Description
 * ------------------------
 *    - Test texture fetching with `tex2DGrad` and read mode set to `hipReadModeElementType`. The
 * test is performed with:
 *      - normalized coordinates
 *      - non-normalized coordinates
 *      - Nearest-point sampling
 *      - Linear filtering
 *      - All combinations of different addressing modes.
 * Test source
 * ------------------------
 *    - unit/texture/tex2DGrad.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.7
 */
TEMPLATE_TEST_CASE("Unit_tex2DGrad_Positive_ReadModeElementType", "", char, unsigned char, short,
                   unsigned short, int, unsigned int, float) {
  CHECK_IMAGE_SUPPORT;

  TextureTestParams<TestType> params = {};
  params.extent = make_hipExtent(16, 4, 0);
  params.num_subdivisions = 4;
  params.GenerateTextureDesc();

  TextureTestFixture<TestType, false, false> fixture{params};

  const auto [num_threads_x, num_blocks_x] = GetLaunchConfig(32, params.NumItersX());
  const auto [num_threads_y, num_blocks_y] = GetLaunchConfig(32, params.NumItersY());

  dim3 dim_grid;
  dim_grid.x = num_blocks_x;
  dim_grid.y = num_blocks_y;

  dim3 dim_block;
  dim_block.x = num_threads_x;
  dim_block.y = num_threads_y;

  tex2DGradKernel<vec4<TestType>><<<dim_grid, dim_block>>>(
      fixture.out_alloc_d.ptr(), params.NumItersX(), params.NumItersY(), fixture.tex.object(),
      params.Width(), params.Height(), params.num_subdivisions, params.tex_desc.normalizedCoords,
      float2{0.5f, 0.5f}, float2{0.5f, 0.5f});
  HIP_CHECK(hipGetLastError());

  fixture.LoadOutput();

  for (auto i = 0u; i < params.NumItersX() * params.NumItersY(); ++i) {
    float x = i % params.NumItersX();
    float y = i / params.NumItersX();

    x = GetCoordinate(x, params.NumItersX(), params.Width(), params.num_subdivisions,
                      params.tex_desc.normalizedCoords);
    y = GetCoordinate(y, params.NumItersY(), params.Height(), params.num_subdivisions,
                      params.tex_desc.normalizedCoords);

    INFO("Filtering mode: " << FilteringModeToString(params.tex_desc.filterMode));
    INFO("Normalized coordinates: " << std::boolalpha << params.tex_desc.normalizedCoords);
    INFO("Address mode X: " << AddressModeToString(params.tex_desc.addressMode[0]));
    INFO("Address mode Y: " << AddressModeToString(params.tex_desc.addressMode[1]));
    INFO("x: " << std::fixed << std::setprecision(16) << x);
    INFO("y: " << std::fixed << std::setprecision(16) << y);

    const auto ref_val = fixture.tex_h.Tex2D(x, y, params.tex_desc);
    REQUIRE(fixture.Verify(fixture.out_alloc_h[i], ref_val));
  }
}

/**
 * Test Description
 * ------------------------
 *    - Test texture fetching with `tex2DGrad` and read mode set to `hipReadModeNormalizedFloat`.
 * The test is performed with:
 *      - normalized coordinates
 *      - non-normalized coordinates
 *      - Nearest-point sampling
 *      - Linear filtering
 *      - All combinations of different addressing modes.
 * Test source
 * ------------------------
 *    - unit/texture/tex2DGrad.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.7
 */
TEMPLATE_TEST_CASE("Unit_tex2DGrad_Positive_ReadModeNormalizedFloat", "", char, unsigned char,
                   short, unsigned short) {
  CHECK_IMAGE_SUPPORT;

  TextureTestParams<TestType> params = {};
  params.extent = make_hipExtent(16, 4, 0);
  params.num_subdivisions = 4;
  params.GenerateTextureDesc(hipReadModeNormalizedFloat);

  TextureTestFixture<TestType, true, false> fixture{params};

  const auto [num_threads_x, num_blocks_x] = GetLaunchConfig(32, params.NumItersX());
  const auto [num_threads_y, num_blocks_y] = GetLaunchConfig(32, params.NumItersY());

  dim3 dim_grid;
  dim_grid.x = num_blocks_x;
  dim_grid.y = num_blocks_y;

  dim3 dim_block;
  dim_block.x = num_threads_x;
  dim_block.y = num_threads_y;

  tex2DGradKernel<vec4<float>><<<dim_grid, dim_block>>>(
      fixture.out_alloc_d.ptr(), params.NumItersX(), params.NumItersY(), fixture.tex.object(),
      params.Width(), params.Height(), params.num_subdivisions, params.tex_desc.normalizedCoords,
      float2{0.5f, 0.5f}, float2{0.5f, 0.5f});
  HIP_CHECK(hipGetLastError());

  fixture.LoadOutput();

  for (auto i = 0u; i < params.NumItersX() * params.NumItersY(); ++i) {
    float x = i % params.NumItersX();
    float y = i / params.NumItersX();

    x = GetCoordinate(x, params.NumItersX(), params.Width(), params.num_subdivisions,
                      params.tex_desc.normalizedCoords);
    y = GetCoordinate(y, params.NumItersY(), params.Height(), params.num_subdivisions,
                      params.tex_desc.normalizedCoords);

    INFO("Filtering mode: " << FilteringModeToString(params.tex_desc.filterMode));
    INFO("Normalized coordinates: " << std::boolalpha << params.tex_desc.normalizedCoords);
    INFO("Address mode X: " << AddressModeToString(params.tex_desc.addressMode[0]));
    INFO("Address mode Y: " << AddressModeToString(params.tex_desc.addressMode[1]));
    INFO("x: " << std::fixed << std::setprecision(16) << x);
    INFO("y: " << std::fixed << std::setprecision(16) << y);

    auto ref_val = fixture.tex_h.Tex2D(x, y, params.tex_desc);
    REQUIRE(fixture.Verify(fixture.out_alloc_h[i], ref_val));
  }
}

/**
 * End doxygen group TextureTest.
 * @}
 */
