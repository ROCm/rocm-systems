// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Selected destination words from the original RDNA matrix captures.
#pragma once

#include "rdna_matrix_sample.h"

namespace rocjitsu::matrix_test {
// RDNA3 IU8, wave32, signed factors and intermediate saturation.
inline constexpr Sample kWmmaIu8Gfx11[] = {
    {0, 0x80003f12u},   {1, 0xffffaab0u},   {2, 0x000016b9u},   {3, 0xffff7231u},
    {4, 0x800025b4u},   {5, 0x00007906u},   {6, 0x1233d723u},   {7, 0x7fffe382u},
    {8, 0x800048f1u},   {9, 0x80007bdbu},   {10, 0x80000693u},  {11, 0xffff335au},
    {12, 0xfffff11du},  {13, 0x7fffe2eeu},  {14, 0x7ffffb04u},  {15, 0x7fff9638u},
    {31, 0x00007adbu},  {32, 0x8000421eu},  {63, 0x7fffffffu},  {64, 0xffffc893u},
    {95, 0x80004a76u},  {96, 0x00000e1cu},  {127, 0x80000000u}, {128, 0x800047eau},
    {159, 0xffffdf4au}, {160, 0x7ffff74eu}, {191, 0x7fffedc9u}, {192, 0x12349b35u},
    {223, 0x7fffaee8u}, {224, 0x80002be5u}, {254, 0x80004453u}, {255, 0x7fffb2aeu},
};
// RDNA4 IU4 K64, wave32, sparse key 0 and saturation.
inline constexpr Sample kSwmmacIu4K64[] = {
    {0, 0x0000001bu},   {1, 0x7fffffffu},   {2, 0x00000068u},   {3, 0xffffff00u},
    {4, 0x8000107eu},   {5, 0x8000103bu},   {6, 0x80001099u},   {7, 0x80000fbdu},
    {8, 0x80000fd6u},   {9, 0x00000001u},   {10, 0x12345682u},  {11, 0x000000a6u},
    {12, 0x80000fdau},  {13, 0xffffff9eu},  {14, 0x80000000u},  {15, 0x7fffef33u},
    {31, 0x00000058u},  {32, 0xffffffe6u},  {63, 0x80000000u},  {64, 0x123456ccu},
    {95, 0x00000042u},  {96, 0xffffff8fu},  {127, 0x000000cdu}, {128, 0x7ffff007u},
    {159, 0x000000e0u}, {160, 0x80001004u}, {191, 0x7fffffffu}, {192, 0x1234560fu},
    {223, 0xfffffffdu}, {224, 0x80001025u}, {254, 0x7fffef74u}, {255, 0xffffff91u},
};
} // namespace rocjitsu::matrix_test
