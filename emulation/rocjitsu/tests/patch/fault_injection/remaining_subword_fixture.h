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

// Concrete llvm-mc and target-decoder witnesses; each source uses v2 as address.
inline constexpr std::array<LdsFormCase, 57> kRemainingSubwordCases = {{
    RDNA4_LDS_CASE(GFX1200, .form = "ds_load_i8", .words = {0xd8e40004u, 0x04000002u}),
    RDNA4_LDS_CASE(GFX1200, .form = "ds_load_u8", .words = {0xd8e80004u, 0x04000002u}),
    RDNA4_LDS_CASE(GFX1200, .form = "ds_load_i16", .words = {0xd8ec0004u, 0x04000002u}),
    RDNA4_LDS_CASE(GFX1200, .form = "ds_load_u16", .words = {0xd8f00004u, 0x04000002u}),
    RDNA4_LDS_CASE(GFX1200, .form = "ds_load_u8_d16", .words = {0xda880004u, 0x04000002u}),
    RDNA4_LDS_CASE(GFX1200, .form = "ds_load_u8_d16_hi", .words = {0xda8c0004u, 0x04000002u}),
    RDNA4_LDS_CASE(GFX1200, .form = "ds_load_i8_d16", .words = {0xda900004u, 0x04000002u}),
    RDNA4_LDS_CASE(GFX1200, .form = "ds_load_i8_d16_hi", .words = {0xda940004u, 0x04000002u}),
    RDNA4_LDS_CASE(GFX1200, .form = "ds_load_u16_d16", .words = {0xda980004u, 0x04000002u}),
    RDNA4_LDS_CASE(GFX1200, .form = "ds_load_u16_d16_hi", .words = {0xda9c0004u, 0x04000002u}),
    RDNA4_LDS_CASE(GFX1200, .form = "ds_store_b8", .words = {0xd8780004u, 0x00000402u}),
    RDNA4_LDS_CASE(GFX1200, .form = "ds_store_b16", .words = {0xd87c0004u, 0x00000402u}),
    RDNA4_LDS_CASE(GFX1200, .form = "ds_store_b8_d16_hi", .words = {0xda800004u, 0x00000402u}),
    RDNA4_LDS_CASE(GFX1200, .form = "ds_store_b16_d16_hi", .words = {0xda840004u, 0x00000402u}),
    RDNA4_LDS_CASE(GFX1201, .form = "ds_load_i8", .words = {0xd8e40004u, 0x04000002u}),
    RDNA4_LDS_CASE(GFX1201, .form = "ds_load_u8", .words = {0xd8e80004u, 0x04000002u}),
    RDNA4_LDS_CASE(GFX1201, .form = "ds_load_i16", .words = {0xd8ec0004u, 0x04000002u}),
    RDNA4_LDS_CASE(GFX1201, .form = "ds_load_u16", .words = {0xd8f00004u, 0x04000002u}),
    RDNA4_LDS_CASE(GFX1201, .form = "ds_load_u8_d16", .words = {0xda880004u, 0x04000002u}),
    RDNA4_LDS_CASE(GFX1201, .form = "ds_load_u8_d16_hi", .words = {0xda8c0004u, 0x04000002u}),
    RDNA4_LDS_CASE(GFX1201, .form = "ds_load_i8_d16", .words = {0xda900004u, 0x04000002u}),
    RDNA4_LDS_CASE(GFX1201, .form = "ds_load_i8_d16_hi", .words = {0xda940004u, 0x04000002u}),
    RDNA4_LDS_CASE(GFX1201, .form = "ds_load_u16_d16", .words = {0xda980004u, 0x04000002u}),
    RDNA4_LDS_CASE(GFX1201, .form = "ds_load_u16_d16_hi", .words = {0xda9c0004u, 0x04000002u}),
    RDNA4_LDS_CASE(GFX1201, .form = "ds_store_b8", .words = {0xd8780004u, 0x00000402u}),
    RDNA4_LDS_CASE(GFX1201, .form = "ds_store_b16", .words = {0xd87c0004u, 0x00000402u}),
    RDNA4_LDS_CASE(GFX1201, .form = "ds_store_b8_d16_hi", .words = {0xda800004u, 0x00000402u}),
    RDNA4_LDS_CASE(GFX1201, .form = "ds_store_b16_d16_hi", .words = {0xda840004u, 0x00000402u}),
    CDNA5_LDS_CASE(.form = "ds_load_u8_d16", .words = {0xda880004u, 0x04000002u}),
    CDNA5_LDS_CASE(.form = "ds_load_u8_d16_hi", .words = {0xda8c0004u, 0x04000002u}),
    CDNA5_LDS_CASE(.form = "ds_load_i8_d16", .words = {0xda900004u, 0x04000002u}),
    CDNA5_LDS_CASE(.form = "ds_load_i8_d16_hi", .words = {0xda940004u, 0x04000002u}),
    CDNA5_LDS_CASE(.form = "ds_load_u16_d16", .words = {0xda980004u, 0x04000002u}),
    CDNA5_LDS_CASE(.form = "ds_load_u16_d16_hi", .words = {0xda9c0004u, 0x04000002u}),
    CDNA5_LDS_CASE(.form = "ds_load_tr8_b64", .words = {0xdbf40004u, 0x04000002u}),
    CDNA5_LDS_CASE(.form = "ds_load_tr16_b128", .words = {0xdbf00004u, 0x04000002u}),
    CDNA3_LDS_CASE(.form = "ds_read_i8", .words = {0xd8720004u, 0x04000002u}),
    CDNA3_LDS_CASE(.form = "ds_read_u8", .words = {0xd8740004u, 0x04000002u}),
    CDNA3_LDS_CASE(.form = "ds_read_i16", .words = {0xd8760004u, 0x04000002u}),
    CDNA3_LDS_CASE(.form = "ds_read_u16", .words = {0xd8780004u, 0x04000002u}),
    CDNA3_LDS_CASE(.form = "ds_read_u8_d16", .words = {0xd8ac0004u, 0x04000002u}),
    CDNA3_LDS_CASE(.form = "ds_read_u8_d16_hi", .words = {0xd8ae0004u, 0x04000002u}),
    CDNA3_LDS_CASE(.form = "ds_read_i8_d16", .words = {0xd8b00004u, 0x04000002u}),
    CDNA3_LDS_CASE(.form = "ds_read_i8_d16_hi", .words = {0xd8b20004u, 0x04000002u}),
    CDNA3_LDS_CASE(.form = "ds_read_u16_d16", .words = {0xd8b40004u, 0x04000002u}),
    CDNA3_LDS_CASE(.form = "ds_read_u16_d16_hi", .words = {0xd8b60004u, 0x04000002u}),
    CDNA3_LDS_CASE(.form = "ds_write_b8", .words = {0xd83c0004u, 0x00000402u}),
    CDNA3_LDS_CASE(.form = "ds_write_b16", .words = {0xd83e0004u, 0x00000402u}),
    CDNA3_LDS_CASE(.form = "ds_write_b8_d16_hi", .words = {0xd8a80004u, 0x00000402u}),
    CDNA3_LDS_CASE(.form = "ds_write_b16_d16_hi", .words = {0xd8aa0004u, 0x00000402u}),
    CDNA4_LDS_CASE(.form = "ds_read_i16", .words = {0xd8760004u, 0x04000002u}),
    CDNA4_LDS_CASE(.form = "ds_read_u8_d16", .words = {0xd8ac0004u, 0x04000002u}),
    CDNA4_LDS_CASE(.form = "ds_read_u8_d16_hi", .words = {0xd8ae0004u, 0x04000002u}),
    CDNA4_LDS_CASE(.form = "ds_read_i8_d16", .words = {0xd8b00004u, 0x04000002u}),
    CDNA4_LDS_CASE(.form = "ds_read_i8_d16_hi", .words = {0xd8b20004u, 0x04000002u}),
    CDNA4_LDS_CASE(.form = "ds_read_u16_d16", .words = {0xd8b40004u, 0x04000002u}),
    CDNA4_LDS_CASE(.form = "ds_read_u16_d16_hi", .words = {0xd8b60004u, 0x04000002u}),
}};
#undef RDNA4_LDS_CASE
#undef CDNA3_LDS_CASE
#undef CDNA4_LDS_CASE
#undef CDNA5_LDS_CASE

} // namespace rocjitsu::test
