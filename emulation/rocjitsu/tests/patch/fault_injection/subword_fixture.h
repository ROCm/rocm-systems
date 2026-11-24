// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#pragma once

#include "lds_fixture.h"

namespace rocjitsu::test {
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

inline constexpr std::array<LdsFormCase, 16> kSubwordCases = {{
    CDNA5_LDS_CASE(.form = "ds_load_u16", .words = {0xd8f00004u, 0x05000002u}),
    CDNA5_LDS_CASE(.form = "ds_load_i16", .words = {0xd8ec0004u, 0x05000002u}),
    CDNA5_LDS_CASE(.form = "ds_load_u8", .words = {0xd8e80004u, 0x05000002u}),
    CDNA5_LDS_CASE(.form = "ds_load_i8", .words = {0xd8e40004u, 0x05000002u}),
    CDNA5_LDS_CASE(.form = "ds_store_b8", .words = {0xd8780004u, 0x00000302u}),
    CDNA5_LDS_CASE(.form = "ds_store_b8_d16_hi", .words = {0xda800004u, 0x00000302u}),
    CDNA5_LDS_CASE(.form = "ds_store_b16", .words = {0xd87c0004u, 0x00000302u}),
    CDNA5_LDS_CASE(.form = "ds_store_b16_d16_hi", .words = {0xda840004u, 0x00000302u}),
    CDNA4_LDS_CASE(.form = "ds_read_u16", .words = {0xd8780004u, 0x05000002u}),
    CDNA4_LDS_CASE(.form = "ds_read_b64_tr_b16", .words = {0xd9c60004u, 0x04000002u}),
    CDNA4_LDS_CASE(.form = "ds_write_b16", .words = {0xd83e0004u, 0x00000302u}),
    CDNA4_LDS_CASE(.form = "ds_read_i8", .words = {0xd8720004u, 0x05000002u}),
    CDNA4_LDS_CASE(.form = "ds_read_u8", .words = {0xd8740004u, 0x05000002u}),
    CDNA4_LDS_CASE(.form = "ds_write_b8", .words = {0xd83c0004u, 0x00000302u}),
    CDNA4_LDS_CASE(.form = "ds_write_b8_d16_hi", .words = {0xd8a80004u, 0x00000302u}),
    CDNA4_LDS_CASE(.form = "ds_write_b16_d16_hi", .words = {0xd8aa0004u, 0x00000302u}),
}};
#undef CDNA4_LDS_CASE
#undef CDNA5_LDS_CASE

} // namespace rocjitsu::test
