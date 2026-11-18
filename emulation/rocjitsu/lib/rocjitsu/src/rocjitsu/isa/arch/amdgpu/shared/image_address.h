// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file image_address.h
/// @brief Dependency-light image address encoding helpers.
/// @details Generated decoder-model sources include this header to describe
/// register operands, so it must remain free of VM dependencies. Keep only
/// encoding-level value types here; execution helpers belong in image_access.h.

#pragma once

#include <array>
#include <bit>
#include <cstdint>

namespace rocjitsu::amdgpu {

/// @brief LOD input mode of a qualified sampling instruction.
enum class ImageSampleMode { Implicit, Zero, Explicit, Bias, Derivatives, Derivatives16 };

/// Encoded controls selecting active and derivative coordinates for RDNA sampler operations.
struct ImageSampleSources {
  uint16_t sampler;
  uint8_t dim;
  ImageSampleMode mode;
  bool query;
  bool linear_candidate;
  bool unorm;
  bool offset;
  bool compare;
  bool clamp;
  bool gather;
  bool lod_warning = false;
  bool filtered_feedback_candidate = false;
};

/// The top and left edges provide implicit texture gradients for a 2x2 pixel quad.
class ImageDerivativeQuad {
public:
  explicit constexpr ImageDerivativeQuad(unsigned lane) : origin(lane & ~3u) {}
  unsigned origin;
  constexpr unsigned right() const { return origin + 1; }
  constexpr unsigned bottom() const { return origin + 2; }
  static constexpr uint64_t source_lanes(uint64_t exec) {
    const uint64_t origins = (exec | exec >> 1 | exec >> 2 | exec >> 3) & 0x1111111111111111ull;
    return origins | (origins << 1) | (origins << 2);
  }
};

constexpr bool linear_image_needs_implicit_lod(unsigned mip_filter, unsigned base, unsigned last) {
  return mip_filter != 0 && base != last;
}

/// Unnormalized repeat clamps to the edge and mirror reflects once before clamping.
constexpr uint32_t unnormalized_image_clamp(uint32_t wrap) { return wrap < 2 ? wrap + 2 : wrap; }

constexpr bool graphics_sample_has_fixed_lod(bool query, bool lod_warning, uint32_t filter,
                                             bool native_texture) {
  return !query && !lod_warning && !((filter >> 26) & 3u) && native_texture;
}

constexpr bool graphics_image_needs_gradients(bool query, unsigned maximum_mip, unsigned dim,
                                              bool gather, uint32_t sampler_filter) {
  if (query || maximum_mip)
    return true;
  // Gather fixes both XY filters to linear. Cube uses non-anisotropic XY filtering.
  const unsigned mask = dim == 3 ? 1 : 3;
  const unsigned mag = (sampler_filter >> 20) & mask;
  const unsigned min = (sampler_filter >> 22) & mask;
  return !gather && (min != mag || (min & 2) || (mag & 2));
}

/// @brief Candidate wave-relative VGPR selectors for image coordinates.
/// @details Printing and dependency metadata consume the prefix selected by
/// the instruction's DIM and A16 fields. Linear execution may consume a
/// shorter prefix selected by the resource type after checking that DIM
/// supplies enough coordinates. Entries outside a caller's prefix need not
/// name valid registers.
using ImageAddressRegisters = std::array<uint32_t, 13>;

/// SQ_RSRC_IMG_* resource types.
inline constexpr uint32_t kImage1d = 8;
inline constexpr uint32_t kImage2d = 9;
inline constexpr uint32_t kImage3d = 10;
inline constexpr uint32_t kImageCube = 11;
inline constexpr uint32_t kImage1dArray = 12;
inline constexpr uint32_t kImage2dArray = 13;
inline constexpr uint32_t kImage2dMsaa = 14;
inline constexpr uint32_t kImage2dMsaaArray = 15;

/// @brief Coordinate shape selected by the descriptor for execution and wait checks.
class ImageCoordinateShape {
public:
  bool has_y;
  bool volume;
  bool layered;
  bool has_slice;

  constexpr uint32_t spatial_dimensions() const { return 1u + has_y + volume; }
  constexpr uint32_t coordinates() const { return 1u + has_y + has_slice; }
};

/// @brief Shape of a 1D, 2D, 3D, or layered image. Callers qualify resource
/// types separately. Cube maps use the third component to select a face.
inline constexpr ImageCoordinateShape image_coordinate_shape(uint32_t type, bool array_layer) {
  const bool volume = type == kImage3d;
  const bool layered = type == kImage1dArray || type == kImage2dArray || type == kImageCube;
  return {type != kImage1d && type != kImage1dArray, volume, layered,
          volume || (layered && array_layer)};
}

/// MIMG/VIMAGE DIM encodings. These are instruction fields (0-7), distinct
/// from the SQ_RSRC_IMG_* descriptor TYPE encodings (8-15).
inline constexpr uint8_t kImageDim1d = 0;
inline constexpr uint8_t kImageDim2d = 1;
inline constexpr uint8_t kImageDim3d = 2;
inline constexpr uint8_t kImageDimCube = 3;
inline constexpr uint8_t kImageDim1dArray = 4;
inline constexpr uint8_t kImageDim2dArray = 5;
inline constexpr uint8_t kImageDim2dMsaa = 6;
inline constexpr uint8_t kImageDim2dMsaaArray = 7;

/// A 1D sampling view of a 2D allocation supplies Y=0 to its filter.
constexpr bool image_sample_has_implicit_y(uint32_t dim, uint32_t type) {
  return (dim == kImageDim1d && type == kImage2d) ||
         (dim == kImageDim1dArray && type == kImage2dArray);
}

constexpr bool image_is_1d_or_2d_sample(uint32_t dim) {
  return dim == kImageDim1d || dim == kImageDim1dArray || dim == kImageDim2d ||
         dim == kImageDim2dArray;
}

/// Array resources need a layer operand, and cube faces require the cube DIM.
constexpr bool image_dim_has_resource_layer(uint32_t dim, uint32_t type) {
  if ((dim == kImageDimCube) != (type == kImageCube))
    return false;
  return !image_coordinate_shape(type, true).layered || dim == kImageDimCube ||
         dim == kImageDim2dMsaaArray || dim == kImageDim1dArray || dim == kImageDim2dArray ||
         dim == kImageDim3d;
}

/// Position of the slice/layer among the coordinate inputs consumed by a transfer.
constexpr uint32_t image_transfer_slice_coordinate(uint32_t type, uint32_t dim, bool gfx9,
                                                   bool has_y) {
  // CLR can address a 1D array through a 2D-array or 3D DIM, leaving room for Y.
  return !gfx9 && type == kImage1dArray && (dim == kImageDim2dArray || dim == kImageDim3d) ? 2u
         : has_y                                                                           ? 2u
                                                                                           : 1u;
}

constexpr uint32_t image_transfer_coordinate_count(uint32_t type, bool has_y, bool has_slice,
                                                   uint32_t slice_coordinate) {
  return (has_slice ? slice_coordinate + 1 : 1u + has_y) + (type >= kImage2dMsaa);
}

/// @brief Number of logical coordinates encoded by an instruction DIM.
/// @details In DIM order the coordinates are X; X/Y; X/Y/Z; cube X/Y/face;
/// X/layer; X/Y/layer; X/Y/sample; and X/Y/layer/sample.
inline uint32_t image_coordinate_count(uint32_t dim) {
  static constexpr uint8_t coordinates[] = {1, 2, 3, 3, 2, 3, 3, 4};
  return coordinates[dim & 7u];
}

/// @brief Number of VGPRs containing the DIM's coordinates.
/// @details A16 packs each adjacent pair of logical 16-bit coordinates into
/// one VGPR; an odd final coordinate occupies the low half of its VGPR.
inline uint32_t image_coordinate_register_count(uint32_t dim, bool a16) {
  const uint32_t count = image_coordinate_count(dim);
  return a16 ? (count + 1) / 2 : count;
}

/// @brief Return candidate consecutive selectors, validated by their consumer.
inline ImageAddressRegisters contiguous_image_address_registers(uint32_t first) {
  ImageAddressRegisters registers;
  for (uint32_t i = 0; i < registers.size(); ++i)
    registers[i] = first + i;
  return registers;
}

/// @brief Decode GFX10/GFX11 MIMG coordinate selectors, including derivative groups.
/// GFX10 lists every NSA selector in its encoded extension words. GFX11 uses
/// one word with VADDR1-4 and extends its final selector for longer tuples.
inline ImageAddressRegisters mimg_image_address_registers(uint32_t extension_words,
                                                          bool extend_last, uint32_t first,
                                                          const uint32_t *raw_encoding) {
  auto registers = contiguous_image_address_registers(first);
  if (!extension_words)
    return registers;
  for (uint32_t i = 1; i < registers.size(); ++i) {
    const uint32_t word = (i - 1) / 4;
    registers[i] = word < extension_words ? (raw_encoding[2 + word] >> (8 * ((i - 1) % 4))) & 0xffu
                   : extend_last          ? registers[4] + i - 4
                                          : 256u;
  }
  return registers;
}

/// @brief Decode GFX11 NSA selectors and extend its trailing consecutive group.
/// @pre When @p nsa is true, @p raw_encoding points to at least three words.
inline ImageAddressRegisters mimg_nsa_image_address_registers(bool nsa, uint32_t first,
                                                              const uint32_t *raw_encoding) {
  return mimg_image_address_registers(nsa, true, first, raw_encoding);
}

/// @brief Decode GFX12 explicit selectors and its trailing consecutive group.
/// VIMAGE has five explicit selectors; VSAMPLE has four.
template <typename ImageMachineInst>
inline ImageAddressRegisters gfx12_image_address_registers(const ImageMachineInst &inst) {
  ImageAddressRegisters registers{inst.vaddr0, inst.vaddr1, inst.vaddr2, inst.vaddr3};
  uint32_t last = 3;
  if constexpr (requires { inst.vaddr4; }) {
    registers[4] = inst.vaddr4;
    last = 4;
  }
  for (uint32_t i = last + 1; i < registers.size(); ++i)
    registers[i] = registers[last] + i - last;
  return registers;
}

/// @brief Number of components in each derivative vector; cube uses two.
inline uint32_t image_derivative_component_count(uint32_t dim) {
  static constexpr uint8_t dimensions[] = {1, 2, 3, 2, 1, 2, 2, 2};
  return dimensions[dim & 7u];
}

/// GFX11+ replaces an empty read mask with one zero component. The GFX10
/// ISA requires a nonzero mask; execution rejects the empty encoding.
constexpr unsigned image_result_components(unsigned dmask, bool drops_empty_fetch) {
  return !dmask && drops_empty_fetch ? 1u : std::popcount(dmask);
}

/// G16 packs each horizontal or vertical gradient independently.
constexpr uint32_t image_gradient_vector_words(uint32_t components, bool g16) {
  return g16 ? (components + 1) / 2 : components;
}

/// Each bit marks a prefix word that uses only its low half. A16 bias
/// follows the optional offset. An odd G16 dimension leaves the last high
/// half of both the horizontal and vertical gradient vectors unused.
constexpr uint16_t image_prefix_low16_words(uint32_t control_words, uint32_t g16_components,
                                            bool a16, bool bias, bool offset) {
  uint16_t words = a16 && bias ? 1u << unsigned(offset) : 0;
  if (g16_components & 1u) {
    const uint32_t stride = image_gradient_vector_words(g16_components, true);
    const uint32_t horizontal_tail = control_words + stride - 1;
    words |= (1u << horizontal_tail) | (1u << (horizontal_tail + stride));
  }
  return words;
}

constexpr uint8_t image_prefix_word_bytes(uint32_t word, uint16_t low16_words) {
  return low16_words & (1u << word) ? 0x3 : 0xf;
}

constexpr uint8_t image_coordinate_component_bytes(uint32_t component, bool a16) {
  return a16 ? (component % 2 ? 0xc : 0x3) : 0xf;
}

/// GFX10 packs each gradient vector separately from the coordinate/LOD body.
inline uint32_t gfx10_sample_address_words(uint32_t dim, bool a16, bool offset, bool bias,
                                           bool compare, bool derivatives, bool g16, bool lod) {
  const unsigned axes = image_derivative_component_count(dim);
  const unsigned gradient_words = derivatives ? 2 * image_gradient_vector_words(axes, g16) : 0;
  const unsigned body = image_coordinate_count(dim) + lod;
  return offset + bias + compare + gradient_words + (a16 ? (body + 1) / 2 : body);
}

} // namespace rocjitsu::amdgpu
