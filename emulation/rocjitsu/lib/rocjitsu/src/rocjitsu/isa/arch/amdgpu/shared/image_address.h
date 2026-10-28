// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_IMAGE_ADDRESS_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_IMAGE_ADDRESS_H_

#include <array>
#include <cstdint>

namespace rocjitsu::amdgpu {

using ImageAddressRegisters = std::array<uint32_t, 4>;

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
/// @details In DIM order the coordinates are X; X/Y; X/Y/Z; cube X/Y/Z;
/// X/layer; X/Y/layer; X/Y/sample; and X/Y/sample/layer.
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

inline ImageAddressRegisters contiguous_image_address_registers(uint32_t first) {
  return {first, first + 1, first + 2, first + 3};
}

inline ImageAddressRegisters gfx11_image_address_registers(bool nsa, uint32_t first,
                                                           const uint32_t *raw_encoding) {
  if (!nsa)
    return contiguous_image_address_registers(first);
  // GFX11's NSA extension stores VADDR1-3 selectors in encoding word 2.
  // These are coordinate VGPR numbers, not a width, and need not be consecutive.
  return {first, raw_encoding[2] & 0xffu, (raw_encoding[2] >> 8) & 0xffu,
          (raw_encoding[2] >> 16) & 0xffu};
}

template <typename VimageMachineInst>
inline ImageAddressRegisters gfx12_image_address_registers(const VimageMachineInst &inst) {
  return {inst.vaddr0, inst.vaddr1, inst.vaddr2, inst.vaddr3};
}

} // namespace rocjitsu::amdgpu

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_IMAGE_ADDRESS_H_
