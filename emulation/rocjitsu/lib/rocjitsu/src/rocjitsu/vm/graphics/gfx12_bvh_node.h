// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>

namespace rocjitsu::amdgpu::gfx12_bvh {

using Node = std::array<uint32_t, 32>;
using Vector = std::array<float, 3>;

// RTIP3.1 primitive structures and HP64 boxes both pack fields across DWORD boundaries.
inline uint32_t bits(const Node &node, unsigned start, unsigned count) {
  if (!count || count > 32 || start > 1024 - count)
    return 0;
  const unsigned word = start / 32;
  uint64_t pair = node[word];
  if (word + 1 < node.size())
    pair |= uint64_t{node[word + 1]} << 32;
  return (pair >> (start % 32)) & ((uint64_t{1} << count) - 1);
}

inline uint32_t high_precision_plane(unsigned block, unsigned exponent, bool maximum) {
  const bool negative = block >> 17;
  const unsigned mantissa = block & 0x1ffff;
  const bool round_out = maximum != negative;
  uint32_t significand = 0, biased = 0;
  if (mantissa) {
    const unsigned shift = std::min(exponent, 17u - std::bit_width(mantissa));
    biased = exponent - shift;
    significand = mantissa << (7 + shift);
    if (round_out && biased != 255)
      significand |= (1u << (7 + shift)) - 1;
  } else if (round_out) {
    biased = exponent > 17 ? exponent - 17 : 0;
    significand = 0x7fffff;
  }
  return (uint32_t{negative} << 31) | (biased << 23) | (significand & 0x7fffff);
}

inline Node expand_hp64(const Node &packed, bool paired_types) {
  Node node{};
  unsigned base = bits(packed, 0, 29);
  const auto flags = bits(packed, 41, 15);
  for (unsigned child = 0; child < 4; ++child) {
    const unsigned type = bits(packed, 29 + 3 * child, 3);
    node[child] = (base << 3) | type;
    base += paired_types && type == 2 ? 4 : (type == 6 || (paired_types && type == 3)) ? 2 : 1;
    node[28] |= ((flags >> (4 * child)) & 15) << (8 * child);
    for (unsigned plane = 0; plane < 6; ++plane) {
      const unsigned i = 6 * child + plane;
      node[4 + i] = high_precision_plane(bits(packed, 80 + 18 * i, 18),
                                         bits(packed, 56 + 8 * (plane % 3), 8), plane >= 3);
    }
  }
  return node;
}

struct Primitive {
  const Node &node;
  unsigned pair;
  unsigned descriptor() const { return bits(node, 1024 - (pair + 1) * 29, 29); }
  unsigned pairs() const { return 1 + bits(node, 28, 3); }
  unsigned navigation() const { return descriptor() & 1 ? 3 : pair + 1 == pairs() ? 1 : 0; }

  unsigned id(unsigned slot, bool geometry) const {
    const unsigned mid = bits(node, 42, 10);
    const unsigned width = geometry ? 2 * bits(node, 20, 4) : bits(node, 32, 5);
    const unsigned delta = geometry ? 2 * bits(node, 24, 4) : bits(node, 37, 5);
    const unsigned anchor = bits(node, geometry ? mid - width : mid, width);
    const unsigned index = 2 * pair + slot;
    if (!index)
      return anchor;
    const unsigned offset =
        geometry ? mid - width - index * delta : mid + width + (index - 1) * delta;
    return (anchor & ~((1u << delta) - 1)) | bits(node, offset, delta);
  }

  Vector vertex(unsigned index) const {
    const unsigned zeros = bits(node, 15, 5);
    unsigned prefix_start = 52, stride = 0;
    std::array<unsigned, 3> lengths{}, prefixes{};
    for (unsigned axis = 0; axis < 3; ++axis) {
      lengths[axis] = 1 + bits(node, 5 * axis, 5);
      if (lengths[axis] + zeros > 32)
        return {};
      const unsigned prefix = 32 - zeros - lengths[axis];
      if (prefix)
        prefixes[axis] = bits(node, prefix_start, prefix) << (32 - prefix);
      prefix_start += prefix;
      stride += lengths[axis];
    }
    unsigned offset = prefix_start + index * stride;
    Vector result{};
    for (unsigned axis = 0; axis < 3; ++axis) {
      result[axis] =
          std::bit_cast<float>(prefixes[axis] | (bits(node, offset, lengths[axis]) << zeros));
      offset += lengths[axis];
    }
    return result;
  }
};

inline Vector transform(const std::array<float, 12> &matrix, const Vector &value, bool point) {
  Vector result{};
  for (unsigned row = 0; row < 3; ++row) {
    const auto *m = &matrix[4 * row];
    const float z = point ? std::fma(m[2], value[2], m[3]) : m[2] * value[2];
    result[row] = std::fma(m[0], value[0], std::fma(m[1], value[1], z));
  }
  return result;
}

// The two hardware OBB lookup tables are from AMD GPURT ObbCommon.hlsli (MIT,
// copyright 2021-2025 Advanced Micro Devices). Each rotation has nine signed 6-bit indices.
inline std::array<float, 12> orientation(unsigned index) {
  static constexpr uint64_t rotations[] = {
      0x162c0ad6000019, 0x16ac02d6000019, 0x11440c51000019, 0x11c40451000019, 0xb580d8b000019,
      0xbd8058b000019,  0x640e40000019,   0xe40640000019,   0x1602b01900b016, 0x1600b01902b016,
      0x11031019011011, 0x11011019031011, 0xb03601901600b,  0xb01601903600b,  0x39019019000,
      0x19019039000,    0x190000162c0ad6, 0x19000016ac02d6, 0x19000011440c51, 0x19000011c40451,
      0x1900000b580d8b, 0x1900000bd8058b, 0x19000000640e40, 0x19000000e40640, 0x18066058186996,
      0x180460589a6196, 0x140ac09430cb11, 0x1408c094b2c311, 0x101ef1d03cfbcb, 0x101cf1d0bef3cb,
      0xc33130c451c40,  0xc31130cc71440,  0x188468581a6996, 0x18866858986196, 0x1488c89432cb11,
      0x148ac894b0c311, 0x109cf9d03efbcb, 0x109ef9d0bcf3cb, 0xcb11b0c471c40,  0xcb31b0cc51440,
      0x18181996181998, 0x18981196981198, 0x14302b11302b14, 0x14b02311b02314, 0x103c7bcb3c7bd0,
      0x10bc73cbbc73d0, 0xc44cc4044cc4c,  0xcc4c440c4c44c,  0x181a19969a1198, 0x189a11961a1998,
      0x14322b11b22314, 0x14b22311322b14, 0x103e7bcbbe73d0, 0x10be73cb3e7bd0, 0xc46cc40c6c44c,
      0xcc6c44046cc4c,  0x161a6998046058, 0x16986198066058, 0x1132cb1408c094, 0x11b0c3140ac094,
      0xb3efbd01cf1d0,  0xbbcf3d01ef1d0,  0x471c4c31130c,   0xc5144c33130c,   0x169a6198846858,
      0x16186998866858, 0x11b2c31488c894, 0x1130cb148ac894, 0xbbef3d09cf9d0,  0xb3cfbd09ef9d0,
      0xc7144cb11b0c,   0x451c4cb31b0c,   0x171638d71458d7, 0x178c51578e3157, 0x13368a1334da13,
      0x13a0d353a28353, 0xe4a9a4e492a4e,  0xea5248ea6948e,  0xa56490a55590a,  0xa91554a92454a,
      0x178e51570c3957, 0x171438d79650d7, 0x13a2d353208b53, 0x13348a13b6d213, 0xea7248e249c8e,
      0xe489a4ecb224e,  0xa93554a104d4a,  0xa54490ad7510a,  0x170c59570e3957, 0x179630d79450d7,
      0x1320db53228b53, 0x13b68213b4d213, 0xe252c8e269c8e,  0xeca924ec9224e,  0xa115d4a124d4a,
      0xad6410ad5510a,  0x179430d71658d7, 0x170e59578c3157, 0x13b4821336da13, 0x1322db53a08353,
      0xec8924e4b2a4e,  0xe272c8ea4948e,  0xad4410a57590a,  0xa135d4a90454a};
  static constexpr uint32_t magnitudes[] = {
      0x00000000, 0x3d1be50c, 0x3e15f61a, 0x3e484336, 0x3e79df93, 0x3e7c3a3a, 0x3e8a8bd4,
      0x3e9e0875, 0x3e9f0938, 0x3ea7bf1b, 0x3eaaaaab, 0x3ec3ef15, 0x3f000000, 0x3f01814f,
      0x3f16a507, 0x3f273d75, 0x3f30fbc5, 0x3f3504f3, 0x3f3d3a87, 0x3f4e034d, 0x3f5a827a,
      0x3f692290, 0x3f6c835e, 0x3f73023f, 0x3f7641af, 0x3f800000, 0x00000000, 0x00000000,
      0x00000000, 0x00000000, 0x00000000, 0x00000000};
  std::array<float, 12> matrix{};
  if (index < std::size(rotations)) {
    for (unsigned i = 0; i < 9; ++i) {
      const unsigned selector = (rotations[index] >> (6 * i)) & 63;
      matrix[4 * (i / 3) + i % 3] =
          std::bit_cast<float>(magnitudes[selector & 31] | ((selector >> 5) << 31));
    }
  }
  return matrix;
}

} // namespace rocjitsu::amdgpu::gfx12_bvh
