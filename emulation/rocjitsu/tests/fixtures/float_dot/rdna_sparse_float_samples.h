// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Selected destination words from the original RDNA matrix captures.
#pragma once

#include "rdna_matrix_sample.h"

namespace rocjitsu::matrix_test {
// FP16 factors, wave64, destination overlaps the sparse index register.
inline constexpr Sample kSwmmacF32F16[] = {
    {0, 0x4babaedfu},   {1, 0xc99a6a3eu},   {2, 0xca32be36u},   {3, 0xcc681f72u},
    {4, 0xc9704ca9u},   {5, 0x4622cf1eu},   {6, 0x45a8bfeeu},   {7, 0x4c8540d1u},
    {8, 0x475f8357u},   {9, 0xc934f0d1u},   {10, 0x48aef528u},  {11, 0xc8242b91u},
    {12, 0xccc20108u},  {13, 0xc4130fedu},  {14, 0x3fcd1b4eu},  {15, 0x488e1ce6u},
    {31, 0x4b1e663au},  {32, 0x4953a408u},  {63, 0x464eefd2u},  {64, 0x4a802bf4u},
    {95, 0xce5ab34au},  {96, 0xc8334f46u},  {127, 0xceb17d0du}, {128, 0xcb0977a9u},
    {159, 0xcb818bbbu}, {160, 0xcb4183e8u}, {191, 0xc6c8e9cau}, {192, 0x4a0ebea4u},
    {223, 0xcd5ea600u}, {224, 0x48a5d075u}, {254, 0xed4dc8d4u}, {255, 0xd8948ed9u},
};
} // namespace rocjitsu::matrix_test
