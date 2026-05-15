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

#include <cmath>
#include "fixed_point.hh"

#if defined(_WIN64)
typedef __int64 ssize_t;
#endif // _WIN64

template <typename TexelType, bool normalized_read>
class TextureReference {
  using valType = decltype(TexelType::x);
  static constexpr bool supportFilter = normalized_read || std::is_floating_point<valType>::value;
 public:
  TextureReference(TexelType* alloc, hipExtent extent, size_t layers)
      : alloc_(alloc), extent_{extent}, layers_{layers} {}

  auto Tex1D(float x, const hipTextureDesc& tex_desc) const {
    return Tex1DLayered(x, 0, tex_desc);
  }

  auto Tex2DGather(float x, float y, int comp, const hipTextureDesc& tex_desc) const {
    x = tex_desc.normalizedCoords ? x * extent_.width : x;
    y = tex_desc.normalizedCoords ? y * extent_.height : y;
#if HT_AMD
    const auto [i, alpha] = GetLinearFilteringParams(x - 0.5f);
    const auto [j, beta] = GetLinearFilteringParams(y - 0.5f);
#else
    const auto [i, alpha] = GetLinearFilteringParams(x);
    const auto [j, beta] = GetLinearFilteringParams(y);
#endif
    const auto T_i0j0 = Sample(i, j, 0, tex_desc.addressMode);
    const auto T_i1j0 = Sample(i + 1.0f, j, 0, tex_desc.addressMode);
    const auto T_i0j1 = Sample(i, j + 1.0f, 0, tex_desc.addressMode);
    const auto T_i1j1 = Sample(i + 1.0f, j + 1.0f, 0, tex_desc.addressMode);

    const auto IndexVec4 = [](auto& vec, int comp) {
      switch (comp) {
        case 0:
          return vec.x;
        case 1:
          return vec.y;
        case 2:
          return vec.z;
        case 3:
          return vec.w;
        default:
          throw std::invalid_argument("Invalid gather comp");
      }
    };

    decltype(T_i0j0) texel {
      IndexVec4(T_i0j1, comp),
      IndexVec4(T_i1j1, comp),
      IndexVec4(T_i1j0, comp),
      IndexVec4(T_i0j0, comp)};
    return texel;
  }

  auto Tex2D(float x, float y, const hipTextureDesc& tex_desc) const {
    return Tex2DLayered(x, y, 0, tex_desc);
  }

  auto Tex3D(float x, float y, float z, const hipTextureDesc& tex_desc) const {
    x = tex_desc.normalizedCoords ? x * extent_.width : x;
    y = tex_desc.normalizedCoords ? y * extent_.height : y;
    z = tex_desc.normalizedCoords ? z * extent_.depth : z;
    if (tex_desc.filterMode == hipFilterModePoint) {
      return Sample(floorf(x), floorf(y), floorf(z), tex_desc.addressMode);
    } else if (tex_desc.filterMode == hipFilterModeLinear) {
      if constexpr (supportFilter) {
        return LinearFiltering(x, y, z, tex_desc.addressMode);
      } else {
        throw std::invalid_argument("hipFilterModeLinear not supported");
      }
    } else {
      throw std::invalid_argument("Invalid hipFilterMode value");
    }
  }

  auto TexCubemap(float x, float y, float z, const hipTextureDesc& tex_desc) const {
    x = tex_desc.normalizedCoords ? x * extent_.width : x;
    y = tex_desc.normalizedCoords ? y * extent_.height : y;
    z = tex_desc.normalizedCoords ? z * extent_.depth : z;

    int face;
    float m, s, t;

    if (std::abs(x) > std::abs(y) && std::abs(x) > std::abs(z)) {
      if (x >= 0) {
        face = 0;
        m = x;
        s = -z;
        t = -y;
      } else {
        face = 1;
        m = -x;
        s = z;
        t = -y;
      }
    } else if (std::abs(y) >= std::abs(x) && std::abs(y) > std::abs(z)) {
      if (y >= 0) {
        face = 2;
        m = y;
        s = x;
        t = z;
      } else {
        face = 3;
        m = -y;
        s = x;
        t = -z;
      }
    } else {
      if (z >= 0) {
        face = 4;
        m = z;
        s = x;
        t = -y;
      } else {
        face = 5;
        m = -z;
        s = -x;
        t = -y;
      }
    }

    float coord1 = (s / m + 1) / 2;
    float coord2 = (t / m + 1) / 2;

    if (tex_desc.filterMode == hipFilterModePoint) {
      return Sample(roundf(coord1), roundf(coord2), face, tex_desc.addressMode);
    } else if (tex_desc.filterMode == hipFilterModeLinear) {
      if constexpr (supportFilter) {
        return LinearFiltering(coord1, coord2, face, tex_desc.addressMode);
      } else {
        throw std::invalid_argument("hipFilterModeLinear not supported");
      }
    } else {
      throw std::invalid_argument("Invalid hipFilterMode value");
    }
  }

  auto Tex1DLayered(float x, int layer, const hipTextureDesc& tex_desc) const {
    x = tex_desc.normalizedCoords ? x * extent_.width : x;
    if (tex_desc.filterMode == hipFilterModePoint) {
      return Sample(floorf(x), layer, tex_desc.addressMode);
    } else if (tex_desc.filterMode == hipFilterModeLinear) {
      if constexpr (supportFilter) {
        return LinearFiltering(x, layer, tex_desc.addressMode);
      } else {
        throw std::invalid_argument("hipFilterModeLinear not supported");
      }
    } else {
      throw std::invalid_argument("Invalid hipFilterMode value");
    }
  }

  auto Tex2DLayered(float x, float y, int layer, const hipTextureDesc& tex_desc) const {
    x = tex_desc.normalizedCoords ? x * extent_.width : x;
    y = tex_desc.normalizedCoords ? y * extent_.height : y;
    if (tex_desc.filterMode == hipFilterModePoint) {
      return Sample(floorf(x), floorf(y), layer, tex_desc.addressMode);
    } else if (tex_desc.filterMode == hipFilterModeLinear) {
      if constexpr (supportFilter) {
        return LinearFiltering(x, y, layer, tex_desc.addressMode);
      } else {
        throw std::invalid_argument("hipFilterModeLinear not supported");
      }
    } else {
      throw std::invalid_argument("Invalid hipFilterMode value");
    }
  }

  TexelType* ptr(size_t layer) const {
    return alloc_ + layer * extent_.width * (extent_.height ?: 1);
  }

  size_t width() const { return extent_.width; }

  size_t height() const { return extent_.height; }

  size_t depth() const { return extent_.depth; }

 private:
  TexelType* const alloc_;
  const hipExtent extent_;
  const size_t layers_;

  TexelType Zero() const {
    TexelType ret {0, 0, 0, 0};
    return ret;
  }

  vec4<float> Zerof() const {
    vec4<float> ret {0., 0., 0., 0.};
    return ret;
  }

  float ApplyAddressMode(float coord, size_t dim, hipTextureAddressMode address_mode) const {
    switch (address_mode) {
      case hipAddressModeClamp:
        return ApplyClamp(coord, dim);
      case hipAddressModeBorder:
        if (CheckBorder(coord, dim)) {
          return std::numeric_limits<float>::quiet_NaN();
        }
      case hipAddressModeWrap:
        return ApplyWrap(coord, dim);
      case hipAddressModeMirror:
        return ApplyMirror(coord, dim);
      default:
        throw std::invalid_argument("Invalid hipAddressMode value");
    }
  }

  auto Sample(float x, int layer, const hipTextureAddressMode* address_mode) const {
    x = ApplyAddressMode(x, extent_.width, address_mode[0]);

    if constexpr (normalized_read) {
      if (std::isnan(x)) {
        return Zerof();
      }
      return Vec4Map(ptr(layer)[static_cast<size_t>(x)]);
    } else {
      if (std::isnan(x)) {
        return Zero();
      }
      return ptr(layer)[static_cast<size_t>(x)];
    }
  }

  auto Sample(float x, float y, int layer, const hipTextureAddressMode* address_mode) const {
    x = ApplyAddressMode(x, extent_.width, address_mode[0]);
    y = ApplyAddressMode(y, extent_.height, address_mode[1]);

    if constexpr (normalized_read) {
      if (std::isnan(x) || std::isnan(y)) {
        return Zerof();
      }
      return Vec4Map(
          ptr(layer)[static_cast<size_t>(y) * extent_.width + static_cast<size_t>(x)]);
    } else {
      if (std::isnan(x) || std::isnan(y)) {
        return Zero();
      }
      return ptr(layer)[static_cast<size_t>(y) * extent_.width + static_cast<size_t>(x)];
    }
  }

  auto Sample(float x, float y, float z, const hipTextureAddressMode* address_mode) const {
    x = ApplyAddressMode(x, extent_.width, address_mode[0]);
    y = ApplyAddressMode(y, extent_.height, address_mode[1]);
    z = ApplyAddressMode(z, extent_.depth, address_mode[2]);

    if constexpr (normalized_read) {
      if (std::isnan(x) || std::isnan(y) || std::isnan(z)) {
        return Zerof();
      }
      return Vec4Map(
          ptr(0)[static_cast<size_t>(z) * extent_.width * extent_.height +
                            static_cast<size_t>(y) * extent_.width + static_cast<size_t>(x)]);
    } else {
      if (std::isnan(x) || std::isnan(y) || std::isnan(z)) {
        return Zero();
      }
      return ptr(0)[static_cast<size_t>(z) * extent_.width * extent_.height +
                    static_cast<size_t>(y) * extent_.width + static_cast<size_t>(x)];
    }
  }

  // LinearFiltering won't be called when valType isn't float or normalized_read is false
  auto LinearFiltering(float x, int layer, const hipTextureAddressMode* address_mode) const {
    const auto [i, alpha] = GetLinearFilteringParams(x);

    const auto T_i0 = Sample(i, layer, address_mode);
    const auto T_i1 = Sample(i + 1.0f, layer, address_mode);

    const auto term_i0 = Vec4Scale((1.0f - alpha), T_i0);
    const auto term_i1 = Vec4Scale(alpha, T_i1);
    return term_i0 + term_i1;
  }

  auto LinearFiltering(float x, float y, int layer,
                            const hipTextureAddressMode* address_mode) const {
    const auto [i, alpha] = GetLinearFilteringParams(x);
    const auto [j, beta] = GetLinearFilteringParams(y);

    const auto T_i0j0 = Sample(i, j, layer, address_mode);
    const auto T_i1j0 = Sample(i + 1.0f, j, layer, address_mode);
    const auto T_i0j1 = Sample(i, j + 1.0f, layer, address_mode);
    const auto T_i1j1 = Sample(i + 1.0f, j + 1.0f, layer, address_mode);

    const auto term_i0j0 = Vec4Scale((1.0f - alpha) * (1.0f - beta), T_i0j0);
    const auto term_i1j0 = Vec4Scale(alpha * (1.0f - beta), T_i1j0);
    const auto term_i0j1 = Vec4Scale((1.0f - alpha) * beta, T_i0j1);
    const auto term_i1j1 = Vec4Scale(alpha * beta, T_i1j1);

    return term_i0j0 + term_i1j0 + term_i0j1 + term_i1j1;
  }

  auto LinearFiltering(float x, float y, float z,
                            const hipTextureAddressMode* address_mode) const {
    const auto [i, alpha] = GetLinearFilteringParams(x);
    const auto [j, beta] = GetLinearFilteringParams(y);
    const auto [k, gamma] = GetLinearFilteringParams(z);

    const auto T_i0j0k0 = Sample(i, j, k, address_mode);
    const auto T_i1j0k0 = Sample(i + 1.0f, j, k, address_mode);
    const auto T_i0j1k0 = Sample(i, j + 1.0f, k, address_mode);
    const auto T_i1j1k0 = Sample(i + 1.0f, j + 1.0f, k, address_mode);
    const auto T_i0j0k1 = Sample(i, j, k + 1.0f, address_mode);
    const auto T_i1j0k1 = Sample(i + 1.0f, j, k + 1.0f, address_mode);
    const auto T_i0j1k1 = Sample(i, j + 1.0f, k + 1.0f, address_mode);
    const auto T_i1j1k1 = Sample(i + 1.0f, j + 1.0f, k + 1.0f, address_mode);

    const auto term_i0j0k0 = Vec4Scale((1.0f - alpha) * (1.0f - beta) * (1.0f - gamma), T_i0j0k0);
    const auto term_i1j0k0 = Vec4Scale(alpha * (1.0f - beta) * (1.0f - gamma), T_i1j0k0);
    const auto term_i0j1k0 = Vec4Scale((1.0f - alpha) * beta * (1.0f - gamma), T_i0j1k0);
    const auto term_i1j1k0 = Vec4Scale(alpha * beta * (1.0f - gamma), T_i1j1k0);
    const auto term_i0j0k1 = Vec4Scale((1.0f - alpha) * (1.0f - beta) * gamma, T_i0j0k1);
    const auto term_i1j0k1 = Vec4Scale(alpha * (1.0f - beta) * gamma, T_i1j0k1);
    const auto term_i0j1k1 = Vec4Scale((1.0f - alpha) * beta * gamma, T_i0j1k1);
    const auto term_i1j1k1 = Vec4Scale(alpha * beta * gamma, T_i1j1k1);

    return term_i0j0k0 + term_i1j0k0 + term_i0j1k0 + term_i1j1k0 + term_i0j0k1 + term_i1j0k1 +
                   term_i0j1k1 + term_i1j1k1;
  }

  float ApplyClamp(float coord, size_t dim) const {
    return max(min(coord, static_cast<float>(dim - 1)), 0.0f);
  }

  bool CheckBorder(float coord, size_t dim) const { return coord > dim - 1 || coord < 0.0f; }

  float ApplyWrap(float coord, size_t dim) const {
    coord /= dim;
    coord = coord - floorf(coord);
    coord *= dim;

    return coord;
  }

  float ApplyMirror(float coord, size_t dim) const {
    coord /= dim;
    const float frac = coord - floor(coord);
    const bool is_reversing = static_cast<ssize_t>(floorf(coord)) % 2;
    coord = is_reversing ? 1.0f - frac : frac;
    coord *= dim;
    coord -= (coord == truncf(coord)) * is_reversing;

    return coord;
  }

  template <size_t N> float FloatToNBitFractional(float x) const {
    constexpr size_t mult = 1 << N;
    const auto x_trunc = std::trunc(x);
    const auto x_frac = std::round((x - x_trunc) * mult) / mult;
    return x_trunc + x_frac;
  }

  std::tuple<float, float> GetLinearFilteringParams(float coord) const {
    const auto coordB = FloatToNBitFractional<8>(coord - 0.5f);
    const auto index = floorf(coordB);
    const auto coeff = coordB - index;

    return {index, coeff};
  }
};
