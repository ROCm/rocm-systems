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

#pragma once

#include <hip_test_common.hh>
#include <resource_guards.hh>
#include "utils.hh"
#include "vec4.hh"
#include "texture_reference.hh"
#include "hip_texture_helper.hh"

template <typename TestType> struct TextureTestParams {
  hipExtent extent;
  size_t layers;
  size_t num_subdivisions;
  hipTextureDesc tex_desc;
  bool cubemap;

  size_t Size() const {
    return extent.width * (extent.height ?: 1) * (extent.depth ?: 1) * (layers ?: 1);
  }

  size_t NumItersX() const { return 3 * extent.width * num_subdivisions * 2 + 1; }

  size_t NumItersY() const { return 3 * extent.height * num_subdivisions * 2 + 1; }

  size_t NumItersZ() const { return 3 * extent.depth * num_subdivisions * 2 + 1; }

  size_t NumIters() const { return NumItersX() * NumItersY() * NumItersZ(); }

  size_t Width() const { return extent.width; }

  size_t Height() const { return extent.height; }

  size_t Depth() const { return extent.depth; }

  unsigned int Flags() const {
    return (Layered() ? hipArrayLayered : 0u) | (cubemap ? hipArrayCubemap : 0u);
  }

  hipExtent LayeredExtent() const {
    return Layered() ? make_hipExtent(Width(), Height(), layers) : extent;
  }

  bool Layered() const { return layers > 1; }

  void GenerateTextureDesc(decltype(hipReadModeElementType) read_mode = hipReadModeElementType,
                           bool mipmap = false) {
    constexpr bool is_floating_point = std::is_floating_point_v<TestType>;

    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.readMode = read_mode;

    tex_desc.filterMode = hipFilterModePoint;
    if (is_floating_point || tex_desc.readMode == hipReadModeNormalizedFloat) {
      tex_desc.filterMode = GENERATE(hipFilterModePoint, hipFilterModeLinear);
    }

    tex_desc.normalizedCoords = true;
    if (!mipmap) {  // mipMap requires normalizedCoords = true
      tex_desc.normalizedCoords = GENERATE(false, true);
    }

    auto address_mode_x = hipAddressModeClamp;
    auto address_mode_y = address_mode_x;
    auto address_mode_z = address_mode_y;

    if (tex_desc.normalizedCoords) {
      address_mode_x = GENERATE(hipAddressModeClamp, hipAddressModeBorder, hipAddressModeWrap,
                                hipAddressModeMirror);
      if (extent.height)
        address_mode_y = GENERATE(hipAddressModeClamp, hipAddressModeBorder, hipAddressModeWrap,
                                  hipAddressModeMirror);
      if (extent.depth)
        address_mode_z = GENERATE(hipAddressModeClamp, hipAddressModeBorder, hipAddressModeWrap,
                                  hipAddressModeMirror);
    } else {
      address_mode_x = GENERATE(hipAddressModeClamp, hipAddressModeBorder);
      if (extent.height) address_mode_y = GENERATE(hipAddressModeClamp, hipAddressModeBorder);
      if (extent.depth) address_mode_z = GENERATE(hipAddressModeClamp, hipAddressModeBorder);
    }

    tex_desc.addressMode[0] = address_mode_x;
    if (extent.height) tex_desc.addressMode[1] = address_mode_y;
    if (extent.depth) tex_desc.addressMode[2] = address_mode_z;

    tex_desc.mipmapFilterMode = tex_desc.filterMode;
  }
};

template <typename TestType, bool normalized_read = false, bool mipmap = false>
struct TextureTestFixture {
  using VecType = vec4<TestType>;
  using OutType = std::conditional_t<normalized_read, vec4<float>, VecType>;
  template <typename T>
  using ArrayAllocGuardType =
      std::conditional_t<mipmap, MipmappedArrayAllocGuard<T>, ArrayAllocGuard<T>>;

  TextureTestParams<TestType> params;
  hipResourceDesc res_desc;

  LinearAllocGuard<VecType> host_alloc;
  TextureReference<VecType, normalized_read> tex_h;
  ArrayAllocGuardType<VecType> tex_alloc_d;
  TextureGuard tex;
  LinearAllocGuard<OutType> out_alloc_d;
  std::vector<OutType> out_alloc_h;

  TextureTestFixture(const TextureTestParams<TestType>& p)
      : params{p},
        host_alloc{LinearAllocs::hipHostMalloc, sizeof(VecType) * params.Size()},
        tex_h{host_alloc.ptr(), params.extent, params.layers},
        tex_alloc_d{params.LayeredExtent(), params.Flags()},
        tex{ResDesc(), &params.tex_desc},
        out_alloc_d{LinearAllocs::hipMalloc, sizeof(OutType) * params.NumIters()},
        out_alloc_h(params.NumIters()) {}

  hipResourceDesc* ResDesc() {
    constexpr int test_value_offset = 7;
    for (auto i = 0u; i < params.Size(); ++i) {
      SetVec4<TestType>(host_alloc.ptr()[i], i + test_value_offset);
    }

    hipMemcpy3DParms memcpy_params = {};
    memset(&memcpy_params, 0, sizeof(hipMemcpy3DParms));
    if constexpr (mipmap) {
      memcpy_params.dstArray = tex_alloc_d.GetLevel(0);
    } else {
      memcpy_params.dstArray = tex_alloc_d.ptr();
    }
    memcpy_params.extent = params.LayeredExtent();
    memcpy_params.extent.height = memcpy_params.extent.height ?: 1;
    memcpy_params.extent.depth = memcpy_params.extent.depth ?: 1;
    memcpy_params.srcPtr = make_hipPitchedPtr(tex_h.ptr(0), sizeof(VecType) * params.Width(),
                                              params.Width(), params.Height() ?: 1);
    memcpy_params.kind = hipMemcpyHostToDevice;
    HIP_CHECK(hipMemcpy3D(&memcpy_params));

    memset(&res_desc, 0, sizeof(res_desc));
    if constexpr (mipmap) {
      res_desc.resType = hipResourceTypeMipmappedArray;
      res_desc.res.mipmap.mipmap = tex_alloc_d.ptr();
    } else {
      res_desc.resType = hipResourceTypeArray;
      res_desc.res.array.array = tex_alloc_d.ptr();
    }
    return &res_desc;
  }

  void LoadOutput() {
    HIP_CHECK(hipMemcpy(out_alloc_h.data(), out_alloc_d.ptr(), sizeof(OutType) * params.NumIters(),
                        hipMemcpyDeviceToHost));
    HIP_CHECK(hipDeviceSynchronize());
  }

  template <typename ValType> bool Verify(const ValType& devValue, const ValType& hostValue) {
    bool match = false;
    if (params.tex_desc.filterMode == hipFilterModeLinear)
      match = hipTextureSamplingVerify<ValType, hipFilterModeLinear>(devValue, hostValue);
    else
      match = hipTextureSamplingVerify<ValType, hipFilterModePoint>(devValue, hostValue);
    if (!match) {
      WARN((match ? "Matched: " : "Mismatched: ")
           << " GPU output : " << getString(devValue) << " CPU expected: " << getString(hostValue)
           << "\n");
    }
    return match;
  }
};
