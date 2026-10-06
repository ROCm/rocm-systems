// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>

// The code object kernel descriptor is a 64-byte binary record, not an HSA API
// header. Rocjitsu decodes these fields itself, including GFX12.5 extensions
// that are absent from the private descriptor in some ROCR releases.
namespace rocjitsu::amdhsa {

struct kernel_descriptor_t {
  uint32_t group_segment_fixed_size;
  uint32_t private_segment_fixed_size;
  uint32_t kernarg_size;
  uint8_t reserved0[4];
  int64_t kernel_code_entry_byte_offset;
  uint8_t reserved1[20];
  uint32_t compute_pgm_rsrc3;
  uint32_t compute_pgm_rsrc1;
  uint32_t compute_pgm_rsrc2;
  uint16_t kernel_code_properties;
  uint16_t kernarg_preload;
  uint8_t reserved3[4];
};

static_assert(sizeof(kernel_descriptor_t) == 64);
static_assert(offsetof(kernel_descriptor_t, group_segment_fixed_size) == 0);
static_assert(offsetof(kernel_descriptor_t, private_segment_fixed_size) == 4);
static_assert(offsetof(kernel_descriptor_t, kernarg_size) == 8);
static_assert(offsetof(kernel_descriptor_t, reserved0) == 12);
static_assert(offsetof(kernel_descriptor_t, kernel_code_entry_byte_offset) == 16);
static_assert(offsetof(kernel_descriptor_t, reserved1) == 24);
static_assert(offsetof(kernel_descriptor_t, compute_pgm_rsrc3) == 44);
static_assert(offsetof(kernel_descriptor_t, compute_pgm_rsrc1) == 48);
static_assert(offsetof(kernel_descriptor_t, compute_pgm_rsrc2) == 52);
static_assert(offsetof(kernel_descriptor_t, kernel_code_properties) == 56);
static_assert(offsetof(kernel_descriptor_t, kernarg_preload) == 58);
static_assert(offsetof(kernel_descriptor_t, reserved3) == 60);

// Each field supplies a mask and shift for the descriptor decoders and test
// builders. The bit positions follow the AMDGPU code object ABI register fields.
#define RJ_DESCRIPTOR_FIELD(name, shift, width)                                                    \
  name##_SHIFT = shift, name##_WIDTH = width, name = ((uint32_t{1} << width) - 1) << shift

enum : uint32_t {
  RJ_DESCRIPTOR_FIELD(COMPUTE_PGM_RSRC1_GRANULATED_WORKITEM_VGPR_COUNT, 0, 6),
  RJ_DESCRIPTOR_FIELD(COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT, 6, 4),
  RJ_DESCRIPTOR_FIELD(COMPUTE_PGM_RSRC1_FLOAT_ROUND_MODE_32, 12, 2),
  RJ_DESCRIPTOR_FIELD(COMPUTE_PGM_RSRC1_FLOAT_ROUND_MODE_16_64, 14, 2),
  RJ_DESCRIPTOR_FIELD(COMPUTE_PGM_RSRC1_FLOAT_DENORM_MODE_32, 16, 2),
  RJ_DESCRIPTOR_FIELD(COMPUTE_PGM_RSRC1_FLOAT_DENORM_MODE_16_64, 18, 2),
  RJ_DESCRIPTOR_FIELD(COMPUTE_PGM_RSRC1_ENABLE_DX10_CLAMP, 21, 1),
  RJ_DESCRIPTOR_FIELD(COMPUTE_PGM_RSRC1_DEBUG_MODE, 22, 1),
  RJ_DESCRIPTOR_FIELD(COMPUTE_PGM_RSRC1_ENABLE_IEEE_MODE, 23, 1),
  RJ_DESCRIPTOR_FIELD(COMPUTE_PGM_RSRC1_FP16_OVFL, 26, 1),
  RJ_DESCRIPTOR_FIELD(COMPUTE_PGM_RSRC1_WGP_MODE, 29, 1),
  RJ_DESCRIPTOR_FIELD(COMPUTE_PGM_RSRC1_MEM_ORDERED, 30, 1),
  RJ_DESCRIPTOR_FIELD(COMPUTE_PGM_RSRC1_FWD_PROGRESS, 31, 1),

  RJ_DESCRIPTOR_FIELD(COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT, 0, 1),
  RJ_DESCRIPTOR_FIELD(COMPUTE_PGM_RSRC2_USER_SGPR_COUNT, 1, 5),
  RJ_DESCRIPTOR_FIELD(COMPUTE_PGM_RSRC2_GFX125_USER_SGPR_COUNT, 1, 6),
  RJ_DESCRIPTOR_FIELD(COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X, 7, 1),
  RJ_DESCRIPTOR_FIELD(COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y, 8, 1),
  RJ_DESCRIPTOR_FIELD(COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z, 9, 1),
  RJ_DESCRIPTOR_FIELD(COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_INFO, 10, 1),
  RJ_DESCRIPTOR_FIELD(COMPUTE_PGM_RSRC2_ENABLE_VGPR_WORKITEM_ID, 11, 2),
  RJ_DESCRIPTOR_FIELD(COMPUTE_PGM_RSRC2_GRANULATED_LDS_SIZE, 15, 9),

  RJ_DESCRIPTOR_FIELD(COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET, 0, 6),
  RJ_DESCRIPTOR_FIELD(COMPUTE_PGM_RSRC3_GFX10_PLUS_SHARED_VGPR_COUNT, 0, 4),
  RJ_DESCRIPTOR_FIELD(COMPUTE_PGM_RSRC3_GFX10_PLUS_INST_PREF_SIZE, 4, 6),
  RJ_DESCRIPTOR_FIELD(COMPUTE_PGM_RSRC3_GFX10_PLUS_IMAGE_OP, 31, 1),
  RJ_DESCRIPTOR_FIELD(COMPUTE_PGM_RSRC3_GFX12_PLUS_INST_PREF_SIZE, 4, 8),
  RJ_DESCRIPTOR_FIELD(COMPUTE_PGM_RSRC3_GFX125_NAMED_BAR_CNT, 14, 3),
  RJ_DESCRIPTOR_FIELD(COMPUTE_PGM_RSRC3_GFX125_ENABLE_DYNAMIC_VGPR, 17, 1),
  RJ_DESCRIPTOR_FIELD(COMPUTE_PGM_RSRC3_GFX125_TCP_SPLIT, 18, 3),

  RJ_DESCRIPTOR_FIELD(KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER, 0, 1),
  RJ_DESCRIPTOR_FIELD(KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR, 1, 1),
  RJ_DESCRIPTOR_FIELD(KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR, 2, 1),
  RJ_DESCRIPTOR_FIELD(KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR, 3, 1),
  RJ_DESCRIPTOR_FIELD(KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID, 4, 1),
  RJ_DESCRIPTOR_FIELD(KERNEL_CODE_PROPERTY_ENABLE_SGPR_FLAT_SCRATCH_INIT, 5, 1),
  RJ_DESCRIPTOR_FIELD(KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE, 6, 1),
  RJ_DESCRIPTOR_FIELD(KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32, 10, 1),
  RJ_DESCRIPTOR_FIELD(KERNEL_CODE_PROPERTY_USES_DYNAMIC_STACK, 11, 1),

  RJ_DESCRIPTOR_FIELD(KERNARG_PRELOAD_SPEC_LENGTH, 0, 7),
  RJ_DESCRIPTOR_FIELD(KERNARG_PRELOAD_SPEC_OFFSET, 7, 9),
};
#undef RJ_DESCRIPTOR_FIELD

inline constexpr uint8_t FLOAT_ROUND_MODE_PLUS_INFINITY = 1;
inline constexpr uint8_t FLOAT_ROUND_MODE_ZERO = 3;
inline constexpr uint8_t FLOAT_DENORM_MODE_FLUSH_SRC = 2;
inline constexpr uint8_t FLOAT_DENORM_MODE_FLUSH_NONE = 3;

} // namespace rocjitsu::amdhsa

// Keep the existing field access spelling while moving descriptor ownership
// from ROCR's private loader header into rocjitsu.
#define AMDHSA_BITS_GET(src, mask) (((src) & (mask)) >> mask##_SHIFT)
#define AMDHSA_BITS_SET(dst, mask, value)                                                          \
  (dst) = ((dst) & ~(mask)) | (((value) << mask##_SHIFT) & (mask))
