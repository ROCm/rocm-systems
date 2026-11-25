// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>
#include <string_view>

namespace rocjitsu::test {

struct Gfx1251LdsFixture {
  std::string_view form;
  std::array<uint32_t, 2> source;
  std::array<uint32_t, 2> expected;
};

// llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1251 --show-encoding.
// Sources use address v2; independent expected encodings use address v6.
// Single-address forms use offset:260. Two-address forms use offset0:1
// offset1:3. Data starts at v3 or v8; load destinations start at v12.
inline constexpr std::array<Gfx1251LdsFixture, 34> kGfx1251LdsFixtures = {{
    {.form = "ds_store_b32",
     .source = {0xd8340104u, 0x00000302u},
     .expected = {0xd8340104u, 0x00000306u}},
    {.form = "ds_load_b32",
     .source = {0xd8d80104u, 0x0c000002u},
     .expected = {0xd8d80104u, 0x0c000006u}},
    {.form = "ds_store_b64",
     .source = {0xd9340104u, 0x00000802u},
     .expected = {0xd9340104u, 0x00000806u}},
    {.form = "ds_load_b64",
     .source = {0xd9d80104u, 0x0c000002u},
     .expected = {0xd9d80104u, 0x0c000006u}},
    {.form = "ds_store_b96",
     .source = {0xdb780104u, 0x00000802u},
     .expected = {0xdb780104u, 0x00000806u}},
    {.form = "ds_load_b96",
     .source = {0xdbf80104u, 0x0c000002u},
     .expected = {0xdbf80104u, 0x0c000006u}},
    {.form = "ds_store_b128",
     .source = {0xdb7c0104u, 0x00000802u},
     .expected = {0xdb7c0104u, 0x00000806u}},
    {.form = "ds_load_b128",
     .source = {0xdbfc0104u, 0x0c000002u},
     .expected = {0xdbfc0104u, 0x0c000006u}},
    {.form = "ds_store_b8",
     .source = {0xd8780104u, 0x00000302u},
     .expected = {0xd8780104u, 0x00000306u}},
    {.form = "ds_store_b16",
     .source = {0xd87c0104u, 0x00000302u},
     .expected = {0xd87c0104u, 0x00000306u}},
    {.form = "ds_store_b8_d16_hi",
     .source = {0xda800104u, 0x00000302u},
     .expected = {0xda800104u, 0x00000306u}},
    {.form = "ds_store_b16_d16_hi",
     .source = {0xda840104u, 0x00000302u},
     .expected = {0xda840104u, 0x00000306u}},
    {.form = "ds_load_i8",
     .source = {0xd8e40104u, 0x0c000002u},
     .expected = {0xd8e40104u, 0x0c000006u}},
    {.form = "ds_load_u8",
     .source = {0xd8e80104u, 0x0c000002u},
     .expected = {0xd8e80104u, 0x0c000006u}},
    {.form = "ds_load_i16",
     .source = {0xd8ec0104u, 0x0c000002u},
     .expected = {0xd8ec0104u, 0x0c000006u}},
    {.form = "ds_load_u16",
     .source = {0xd8f00104u, 0x0c000002u},
     .expected = {0xd8f00104u, 0x0c000006u}},
    {.form = "ds_load_u8_d16",
     .source = {0xda880104u, 0x0c000002u},
     .expected = {0xda880104u, 0x0c000006u}},
    {.form = "ds_load_u8_d16_hi",
     .source = {0xda8c0104u, 0x0c000002u},
     .expected = {0xda8c0104u, 0x0c000006u}},
    {.form = "ds_load_i8_d16",
     .source = {0xda900104u, 0x0c000002u},
     .expected = {0xda900104u, 0x0c000006u}},
    {.form = "ds_load_i8_d16_hi",
     .source = {0xda940104u, 0x0c000002u},
     .expected = {0xda940104u, 0x0c000006u}},
    {.form = "ds_load_u16_d16",
     .source = {0xda980104u, 0x0c000002u},
     .expected = {0xda980104u, 0x0c000006u}},
    {.form = "ds_load_u16_d16_hi",
     .source = {0xda9c0104u, 0x0c000002u},
     .expected = {0xda9c0104u, 0x0c000006u}},
    {.form = "ds_load_tr4_b64",
     .source = {0xdbe80104u, 0x0c000002u},
     .expected = {0xdbe80104u, 0x0c000006u}},
    {.form = "ds_load_tr6_b96",
     .source = {0xdbec0104u, 0x0c000002u},
     .expected = {0xdbec0104u, 0x0c000006u}},
    {.form = "ds_load_tr16_b128",
     .source = {0xdbf00104u, 0x0c000002u},
     .expected = {0xdbf00104u, 0x0c000006u}},
    {.form = "ds_load_tr8_b64",
     .source = {0xdbf40104u, 0x0c000002u},
     .expected = {0xdbf40104u, 0x0c000006u}},
    {.form = "ds_store_2addr_b32",
     .source = {0xd8380301u, 0x00040302u},
     .expected = {0xd8380301u, 0x00040306u}},
    {.form = "ds_store_2addr_stride64_b32",
     .source = {0xd83c0301u, 0x00040302u},
     .expected = {0xd83c0301u, 0x00040306u}},
    {.form = "ds_load_2addr_b32",
     .source = {0xd8dc0301u, 0x0c000002u},
     .expected = {0xd8dc0301u, 0x0c000006u}},
    {.form = "ds_load_2addr_stride64_b32",
     .source = {0xd8e00301u, 0x0c000002u},
     .expected = {0xd8e00301u, 0x0c000006u}},
    {.form = "ds_store_2addr_b64",
     .source = {0xd9380301u, 0x000a0802u},
     .expected = {0xd9380301u, 0x000a0806u}},
    {.form = "ds_store_2addr_stride64_b64",
     .source = {0xd93c0301u, 0x000a0802u},
     .expected = {0xd93c0301u, 0x000a0806u}},
    {.form = "ds_load_2addr_b64",
     .source = {0xd9dc0301u, 0x0c000002u},
     .expected = {0xd9dc0301u, 0x0c000006u}},
    {.form = "ds_load_2addr_stride64_b64",
     .source = {0xd9e00301u, 0x0c000002u},
     .expected = {0xd9e00301u, 0x0c000006u}},
}};

// These LDS operations use M0 and lane ID rather than an address VGPR.
inline constexpr std::array<Gfx1251LdsFixture, 2> kGfx1251ImplicitAddressLdsFixtures = {{
    {.form = "ds_store_addtid_b32",
     .source = {0xdac00104u, 0x00000300u},
     .expected = {0xdac00104u, 0x00000306u}},
    {.form = "ds_load_addtid_b32",
     .source = {0xdac40104u, 0x0c000000u},
     .expected = {0xdac40104u, 0x0c000006u}},
}};

} // namespace rocjitsu::test
