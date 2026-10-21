// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Selected destination words from the original RDNA matrix captures.
#pragma once

#include "rdna_matrix_sample.h"

namespace rocjitsu::matrix_test {
// RDNA3 FP16, wave64, high-half output with the low half preserved.
inline constexpr Sample kWmmaF16Gfx11High[] = {
    {0, 0x7c000000u},   {1, 0xfc000001u},   {9, 0x68bd0001u},   {26, 0x76ae0001u},
    {31, 0xfc000001u},  {32, 0x7c000001u},  {63, 0x7c000001u},  {64, 0xf3990001u},
    {71, 0xec150001u},  {95, 0x7c000001u},  {96, 0xfc000001u},  {98, 0xfa160001u},
    {115, 0xe1f30001u}, {127, 0x7c000001u}, {128, 0xf4af0000u}, {137, 0x73430000u},
    {159, 0xfc000001u}, {160, 0x7c000001u}, {163, 0xdcf50001u}, {191, 0x7c000001u},
    {192, 0x7c000001u}, {199, 0x64930001u}, {221, 0xd4230000u}, {223, 0xfc000001u},
    {224, 0x66d30000u}, {225, 0x660b0001u}, {231, 0xec750000u}, {237, 0x6e3f0001u},
    {253, 0xfa360001u}, {254, 0x7c000001u}, {255, 0x7c000000u},
};
} // namespace rocjitsu::matrix_test
