/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <hip_test_common.hh>
#include <resource_guards.hh>

#include "utils.hh"

#include "kernels.hh"
#include "test_fixture.hh"

/**
 * @addtogroup tex2Dgather tex2Dgather
 * @{
 * @ingroup TextureTest
 */

/**
 * Test Description
 * ------------------------
 *    - Test texture fetching with `tex2Dgather` and read mode set to `hipReadModeElementType`. The
 * test is performed with:
 *      - normalized coordinates
 *      - non-normalized coordinates
 *      - Nearest-point sampling
 *      - Linear filtering
 *      - All combinations of different addressing modes.
 * Test source
 * ------------------------
 *    - unit/texture/tex2Dgather.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEMPLATE_TEST_CASE(Unit_tex2Dgather_Positive_ReadModeElementType, char, unsigned char, short,
                   unsigned short, int, unsigned int, float) {
  CHECK_IMAGE_SUPPORT;

  TextureTestParams<TestType> params = {};
  params.extent = make_hipExtent(16, 4, 0);
  params.num_subdivisions = 4;
  params.GenerateTextureDesc();

  if (params.tex_desc.filterMode == hipFilterModeLinear) return;
  TextureTestFixture<TestType> fixture{params};

  const auto [num_threads_x, num_blocks_x] = GetLaunchConfig(32, params.NumItersX());
  const auto [num_threads_y, num_blocks_y] = GetLaunchConfig(32, params.NumItersY());

  dim3 dim_grid;
  dim_grid.x = num_blocks_x;
  dim_grid.y = num_blocks_y;

  dim3 dim_block;
  dim_block.x = num_threads_x;
  dim_block.y = num_threads_y;

  const int comp = GENERATE(0, 1, 2, 3);

  tex2DgatherKernel<vec4<TestType>><<<dim_grid, dim_block>>>(
      fixture.out_alloc_d.ptr(), comp, params.NumItersX(), params.NumItersY(), fixture.tex.object(),
      params.Width(), params.Height(), params.num_subdivisions, params.tex_desc.normalizedCoords);
  HIP_CHECK(hipGetLastError());

  fixture.LoadOutput();

  for (auto i = 0u; i < params.NumItersX() * params.NumItersY(); ++i) {
    float x = i % params.NumItersX();
    float y = i / params.NumItersX();

    x = GetCoordinate(x, params.NumItersX(), params.Width(), params.num_subdivisions,
                      params.tex_desc.normalizedCoords);
    y = GetCoordinate(y, params.NumItersY(), params.Height(), params.num_subdivisions,
                      params.tex_desc.normalizedCoords);
    const auto ref_val = fixture.tex_h.Tex2DGather(x, y, comp, params.tex_desc);
    if (!fixture.Verify(fixture.out_alloc_h[i], ref_val)) {
      INFO("Filtering mode: " << FilteringModeToString(params.tex_desc.filterMode));
      INFO("Normalized coordinates: " << std::boolalpha << params.tex_desc.normalizedCoords);
      INFO("Address mode X: " << AddressModeToString(params.tex_desc.addressMode[0]));
      INFO("Address mode Y: " << AddressModeToString(params.tex_desc.addressMode[1]));
      INFO("x: " << std::fixed << std::setprecision(16) << x);
      INFO("y: " << std::fixed << std::setprecision(16) << y);
      INFO("comp: " << comp);
      REQUIRE(false);
    }
  }
}

static __global__ void gatherDistinctChannels(hipTextureObject_t texture, float4* output) {
#if !__HIP_NO_IMAGE_SUPPORT
  const unsigned component = threadIdx.x;
  if (component == 4) {
    output[component] = tex2Dgather<float4>(texture, 0.5f, 0.5f);
    tex2Dgather<float4>(&output[5 + component], texture, 0.5f, 0.5f);
  } else {
    output[component] = tex2Dgather<float4>(texture, 0.5f, 0.5f, component);
    tex2Dgather<float4>(&output[5 + component], texture, 0.5f, 0.5f, component);
  }
#endif
}

/**
 * Test Description
 * ------------------------
 *    - Verify zero-based gather channels and the default red channel through both overloads.
 * Test source
 * ------------------------
 *    - unit/texture/tex2Dgather.cc
 * Test requirements
 * ------------------------
 *    - HIP_VERSION >= 5.2
 */
HIP_TEST_CASE(Unit_tex2Dgather_DistinctChannelsAndPointerOverload) {
  CHECK_IMAGE_SUPPORT;
  float4 pixels[16];
  for (auto& pixel : pixels) pixel = make_float4(10, 20, 30, 40);
  ArrayAllocGuard<float4> array(make_hipExtent(4, 4, 0), hipArrayTextureGather);
  HIP_CHECK(hipMemcpy2DToArray(array.ptr(), 0, 0, pixels, 4 * sizeof(float4), 4 * sizeof(float4), 4,
                               hipMemcpyHostToDevice));
  hipResourceDesc res_desc{};
  res_desc.resType = hipResourceTypeArray;
  res_desc.res.array.array = array.ptr();
  hipTextureDesc tex_desc{};
  tex_desc.normalizedCoords = 1;
  tex_desc.readMode = hipReadModeElementType;
  tex_desc.addressMode[0] = tex_desc.addressMode[1] = hipAddressModeClamp;
  TextureGuard texture(&res_desc, &tex_desc);
  LinearAllocGuard<float4> device(LinearAllocs::hipMalloc, 10 * sizeof(float4));
  gatherDistinctChannels<<<1, 5>>>(texture.object(), device.ptr());
  HIP_CHECK(hipGetLastError());
  float4 output[10];
  HIP_CHECK(hipMemcpy(output, device.ptr(), sizeof(output), hipMemcpyDeviceToHost));
  for (unsigned i = 0; i < 10; ++i) {
    const unsigned component = i % 5;
    const float expected = 10.0f * (component == 4 ? 1 : component + 1);
    INFO("component=" << component << " pointer_overload=" << (i >= 5));
    REQUIRE(output[i].x == expected);
    REQUIRE(output[i].y == expected);
    REQUIRE(output[i].z == expected);
    REQUIRE(output[i].w == expected);
  }
}

/**
 * End doxygen group TextureTest.
 * @}
 */
