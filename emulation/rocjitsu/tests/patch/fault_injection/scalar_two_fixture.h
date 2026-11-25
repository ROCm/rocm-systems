// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

#include "lds_fixture.h"

namespace rocjitsu::test {
#define RDNA4_LDS_CASE(target_name, ...)                                                           \
  {                                                                                                \
    .target = ROCJITSU_CODE_TARGET_##target_name, .machine = EF_AMDGPU_MACH_AMDGCN_##target_name,  \
    .arch = IsaFamily::Rdna4, __VA_ARGS__                                                          \
  }
#define CDNA3_LDS_CASE(...)                                                                        \
  {                                                                                                \
    .target = ROCJITSU_CODE_TARGET_GFX942, .machine = EF_AMDGPU_MACH_AMDGCN_GFX942,                \
    .arch = IsaFamily::Cdna3, __VA_ARGS__                                                          \
  }
#define CDNA4_LDS_CASE(...)                                                                        \
  {                                                                                                \
    .target = ROCJITSU_CODE_TARGET_GFX950, .machine = EF_AMDGPU_MACH_AMDGCN_GFX950,                \
    .arch = IsaFamily::Cdna4, __VA_ARGS__                                                          \
  }
#define CDNA5_LDS_CASE(...)                                                                        \
  {                                                                                                \
    .target = ROCJITSU_CODE_TARGET_GFX1250, .machine = EF_AMDGPU_MACH_AMDGCN_GFX1250,              \
    .arch = IsaFamily::Cdna5, __VA_ARGS__                                                          \
  }

// Concrete-target llvm-mc witnesses with nonzero offsets and distinct
// address, data, and destination selectors.
inline constexpr std::array<LdsFormCase, 44> kScalarTwoCases = {{
    RDNA4_LDS_CASE(GFX1200, .form = "ds_load_b64", .words = {0xd9d80004u, 0x04000002u}),
    RDNA4_LDS_CASE(GFX1200, .form = "ds_store_b64", .words = {0xd9340004u, 0x00000402u}),
    RDNA4_LDS_CASE(GFX1200, .form = "ds_load_b128", .words = {0xdbfc0004u, 0x04000002u}),
    RDNA4_LDS_CASE(GFX1200, .form = "ds_store_b128", .words = {0xdb7c0004u, 0x00000402u}),
    RDNA4_LDS_CASE(GFX1200, .form = "ds_load_2addr_b32", .words = {0xd8dc0301u, 0x04000002u}),
    RDNA4_LDS_CASE(GFX1200, .form = "ds_load_2addr_stride64_b32",
                   .words = {0xd8e00301u, 0x04000002u}),
    RDNA4_LDS_CASE(GFX1200, .form = "ds_store_2addr_stride64_b32",
                   .words = {0xd83c0301u, 0x00050402u}),
    RDNA4_LDS_CASE(GFX1200, .form = "ds_store_2addr_b64", .words = {0xd9380301u, 0x00060402u}),
    RDNA4_LDS_CASE(GFX1200, .form = "ds_load_2addr_stride64_b64",
                   .words = {0xd9e00301u, 0x04000002u}),
    RDNA4_LDS_CASE(GFX1200, .form = "ds_store_2addr_stride64_b64",
                   .words = {0xd93c0301u, 0x00060402u}),
    RDNA4_LDS_CASE(GFX1201, .form = "ds_load_b64", .words = {0xd9d80004u, 0x04000002u}),
    RDNA4_LDS_CASE(GFX1201, .form = "ds_store_b64", .words = {0xd9340004u, 0x00000402u}),
    RDNA4_LDS_CASE(GFX1201, .form = "ds_load_b128", .words = {0xdbfc0004u, 0x04000002u}),
    RDNA4_LDS_CASE(GFX1201, .form = "ds_store_b128", .words = {0xdb7c0004u, 0x00000402u}),
    RDNA4_LDS_CASE(GFX1201, .form = "ds_load_2addr_b32", .words = {0xd8dc0301u, 0x04000002u}),
    RDNA4_LDS_CASE(GFX1201, .form = "ds_load_2addr_stride64_b32",
                   .words = {0xd8e00301u, 0x04000002u}),
    RDNA4_LDS_CASE(GFX1201, .form = "ds_store_2addr_stride64_b32",
                   .words = {0xd83c0301u, 0x00050402u}),
    RDNA4_LDS_CASE(GFX1201, .form = "ds_store_2addr_b64", .words = {0xd9380301u, 0x00060402u}),
    RDNA4_LDS_CASE(GFX1201, .form = "ds_load_2addr_stride64_b64",
                   .words = {0xd9e00301u, 0x04000002u}),
    RDNA4_LDS_CASE(GFX1201, .form = "ds_store_2addr_stride64_b64",
                   .words = {0xd93c0301u, 0x00060402u}),
    CDNA5_LDS_CASE(.form = "ds_load_b64", .words = {0xd9d80004u, 0x04000002u}),
    CDNA5_LDS_CASE(.form = "ds_store_b64", .words = {0xd9340004u, 0x00000402u}),
    CDNA5_LDS_CASE(.form = "ds_load_b128", .words = {0xdbfc0004u, 0x04000002u}),
    CDNA5_LDS_CASE(.form = "ds_store_b128", .words = {0xdb7c0004u, 0x00000402u}),
    CDNA3_LDS_CASE(.form = "ds_read_b32", .words = {0xd86c0004u, 0x04000002u}),
    CDNA3_LDS_CASE(.form = "ds_read_b64", .words = {0xd8ec0004u, 0x04000002u}),
    CDNA3_LDS_CASE(.form = "ds_write_b64", .words = {0xd89a0004u, 0x00000402u}),
    CDNA3_LDS_CASE(.form = "ds_read_b128", .words = {0xd9fe0004u, 0x04000002u}),
    CDNA3_LDS_CASE(.form = "ds_read2_b32", .words = {0xd86e0301u, 0x04000002u}),
    CDNA3_LDS_CASE(.form = "ds_write2_b32", .words = {0xd81c0301u, 0x00050402u}),
    CDNA3_LDS_CASE(.form = "ds_read2st64_b32", .words = {0xd8700301u, 0x04000002u}),
    CDNA3_LDS_CASE(.form = "ds_write2st64_b32", .words = {0xd81e0301u, 0x00050402u}),
    CDNA3_LDS_CASE(.form = "ds_write2_b64", .words = {0xd89c0301u, 0x00060402u}),
    CDNA3_LDS_CASE(.form = "ds_read2st64_b64", .words = {0xd8f00301u, 0x04000002u}),
    CDNA4_LDS_CASE(.form = "ds_read_b32", .words = {0xd86c0004u, 0x04000002u}),
    CDNA4_LDS_CASE(.form = "ds_read_b64", .words = {0xd8ec0004u, 0x04000002u}),
    CDNA4_LDS_CASE(.form = "ds_write_b64", .words = {0xd89a0004u, 0x00000402u}),
    CDNA4_LDS_CASE(.form = "ds_read_b128", .words = {0xd9fe0004u, 0x04000002u}),
    CDNA4_LDS_CASE(.form = "ds_read2_b32", .words = {0xd86e0301u, 0x04000002u}),
    CDNA4_LDS_CASE(.form = "ds_write2_b32", .words = {0xd81c0301u, 0x00050402u}),
    CDNA4_LDS_CASE(.form = "ds_read2st64_b32", .words = {0xd8700301u, 0x04000002u}),
    CDNA4_LDS_CASE(.form = "ds_write2st64_b32", .words = {0xd81e0301u, 0x00050402u}),
    CDNA4_LDS_CASE(.form = "ds_write2_b64", .words = {0xd89c0301u, 0x00060402u}),
    CDNA4_LDS_CASE(.form = "ds_read2st64_b64", .words = {0xd8f00301u, 0x04000002u}),
}};
#undef RDNA4_LDS_CASE
#undef CDNA3_LDS_CASE
#undef CDNA4_LDS_CASE
#undef CDNA5_LDS_CASE

inline std::vector<uint8_t> make_scalar_two_elf(const LdsFormCase &fixture, bool banked = false,
                                                bool wave32 = false) {
  return make_lds_form_elf(fixture, banked, wave32);
}
} // namespace rocjitsu::test
