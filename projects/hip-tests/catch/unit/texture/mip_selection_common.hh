/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <vector>

#include <hip_test_common.hh>
#include <resource_guards.hh>
#include "utils.hh"

// Constant-valued mip levels isolate the public accessor's LOD and layer inputs.
template <unsigned Dimensions, bool Layered>
static __global__ void sampleMipSelectionKernel(hipTextureObject_t texture, float* output,
                                                unsigned level, unsigned layer, bool pointer,
                                                unsigned gradient) {
#if !__HIP_NO_IMAGE_SUPPORT
  if (threadIdx.x != 0) return;
  if constexpr (Layered && Dimensions == 1) {
    if (pointer)
      tex1DLayeredLod(output, texture, 0.5f, layer, static_cast<float>(level));
    else
      *output = tex1DLayeredLod<float>(texture, 0.5f, layer, static_cast<float>(level));
  } else if constexpr (Layered && Dimensions == 2) {
    if (pointer)
      tex2DLayeredLod(output, texture, 0.5f, 0.5f, layer, static_cast<float>(level));
    else
      *output = tex2DLayeredLod<float>(texture, 0.5f, 0.5f, layer, static_cast<float>(level));
  } else {
    static_assert(Dimensions == 3 && !Layered);
    // Exercise each component of each derivative independently.
    float4 footprint{};
    const float magnitude = static_cast<float>(1u << level) / 16;
    if (gradient % 3 == 0) footprint.x = magnitude;
    if (gradient % 3 == 1) footprint.y = magnitude;
    if (gradient % 3 == 2) footprint.z = magnitude;
    const float4 dx = gradient < 3 ? footprint : float4{};
    const float4 dy = gradient < 3 ? float4{} : footprint;
    if (pointer)
      tex3DGrad(output, texture, 0.5f, 0.5f, 0.5f, dx, dy);
    else
      *output = tex3DGrad<float>(texture, 0.5f, 0.5f, 0.5f, dx, dy);
  }
#endif
}

// Verify nonzero mip selection and both accessor overloads using distinct level values.
template <unsigned Dimensions, bool Layered> static void CheckMipSelection() {
  constexpr unsigned levels = 4;
  const hipExtent extent = make_hipExtent(16, Dimensions == 1 ? 0 : 16, Layered ? 3 : 16);
  MipmappedArrayAllocGuard<float> allocation(extent, levels, Layered ? hipArrayLayered : 0);
  for (unsigned level = 0; level < levels; ++level) {
    const unsigned width = 16 >> level;
    const unsigned height = Dimensions == 1 ? 1 : 16 >> level;
    const unsigned depth = Layered ? 3 : 16 >> level;
    std::vector<float> host(width * height * depth);
    for (unsigned z = 0; z < depth; ++z)
      for (unsigned i = 0; i < width * height; ++i)
        host[z * width * height + i] = static_cast<float>((Layered ? 100 * z : 0) + 10 * level + 7);
    hipMemcpy3DParms copy{};
    copy.srcPtr = make_hipPitchedPtr(host.data(), width * sizeof(float), width, height);
    copy.dstArray = allocation.GetLevel(level);
    copy.extent = make_hipExtent(width, height, depth);
    copy.kind = hipMemcpyHostToDevice;
    HIP_CHECK(hipMemcpy3D(&copy));
  }
  hipResourceDesc res_desc{};
  res_desc.resType = hipResourceTypeMipmappedArray;
  res_desc.res.mipmap.mipmap = allocation.ptr();
  hipTextureDesc tex_desc{};
  tex_desc.normalizedCoords = 1;
  tex_desc.filterMode = hipFilterModePoint;
  tex_desc.mipmapFilterMode = hipFilterModePoint;
  tex_desc.maxMipmapLevelClamp = levels - 1;
  tex_desc.addressMode[0] = tex_desc.addressMode[1] = tex_desc.addressMode[2] = hipAddressModeClamp;
  TextureGuard texture(&res_desc, &tex_desc);
  LinearAllocGuard<float> output(LinearAllocs::hipMalloc, sizeof(float));
  for (unsigned level = 0; level < levels; ++level)
    for (unsigned layer = 0; layer < (Layered ? 3u : 1u); ++layer)
      for (bool pointer : {false, true})
        for (unsigned gradient = 0; gradient < (Layered ? 1u : 6u); ++gradient) {
          INFO("Level " << level << ", layer " << layer << ", pointer overload " << pointer
                        << ", derivative/component " << gradient);
          sampleMipSelectionKernel<Dimensions, Layered>
              <<<1, 32>>>(texture.object(), output.ptr(), level, layer, pointer, gradient);
          HIP_CHECK(hipGetLastError());
          HIP_CHECK(hipDeviceSynchronize());
          float actual;
          HIP_CHECK(hipMemcpy(&actual, output.ptr(), sizeof(actual), hipMemcpyDeviceToHost));
          REQUIRE(actual == static_cast<float>(100 * layer + 10 * level + 7));
        }
}
