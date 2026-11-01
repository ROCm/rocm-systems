// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file image_address.h
/// @brief Dependency-light image address encoding helpers.
/// @details Generated decoder-model sources include this header to describe
/// register operands, so it must remain free of VM dependencies. Keep only
/// encoding-level value types here; execution helpers belong in image_access.h.

#pragma once

#include <array>
#include <cstdint>

namespace rocjitsu::amdgpu {

/// @brief Candidate wave-relative VGPR selectors for image coordinates.
/// @details Printing and dependency metadata consume the prefix selected by
/// the instruction's DIM and A16 fields. Linear execution may consume a
/// shorter prefix selected by the resource type after checking that DIM
/// supplies enough coordinates. Entries outside a caller's prefix need not
/// name valid registers.
using ImageAddressRegisters = std::array<uint32_t, 12>;

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
inline ImageAddressRegisters gfx11_image_address_registers(bool nsa, uint32_t first,
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

} // namespace rocjitsu::amdgpu
