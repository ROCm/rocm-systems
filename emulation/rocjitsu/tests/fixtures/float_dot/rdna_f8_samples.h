// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Selected destination words from the original RDNA matrix captures.
#pragma once

#include "rdna_matrix_sample.h"

namespace rocjitsu::matrix_test {
// FP8 factors, wave64, destination overlaps A.
inline constexpr Sample kWmmaFp8[] = {
    {0, 0xc69085f6u},   {1, 0xc74056e1u},   {2, 0xc6680734u},   {3, 0x4406f6c0u},
    {4, 0x45204c13u},   {5, 0xc6a4dd9fu},   {6, 0xc589b720u},   {7, 0x443fa944u},
    {8, 0xc77fc235u},   {9, 0xc5e3756cu},   {10, 0x473d2b0au},  {11, 0xc41bcd27u},
    {12, 0xc4bdc782u},  {13, 0xc6200940u},  {14, 0x45ed3179u},  {15, 0xc700ed18u},
    {31, 0x460115a6u},  {32, 0xc32b5cc0u},  {63, 0xc3f0c9bcu},  {64, 0x43c2d704u},
    {95, 0x45c136e6u},  {96, 0xc585d157u},  {127, 0x45c78ca0u}, {128, 0xc5a8bbacu},
    {159, 0xc3806c74u}, {160, 0xc63a2e02u}, {191, 0x460383b6u}, {192, 0xc3dc272cu},
    {223, 0x468ea880u}, {224, 0x45ab0417u}, {254, 0x466bb365u}, {255, 0xc3260a6au},
};
// BF8 factors, wave32, destination overlaps B.
inline constexpr Sample kWmmaBf8[] = {
    {0, 0x49d2ff7eu},   {1, 0x4a48145du},   {2, 0x47441e06u},   {3, 0xcb0d3ac1u},
    {4, 0xc5c85e40u},   {5, 0xc8a83c0au},   {6, 0xc98cbe4du},   {7, 0x4a51a57au},
    {8, 0xc7a9702du},   {9, 0x4a275ff0u},   {10, 0xc4c33e95u},  {11, 0xc4bfb968u},
    {12, 0xcbeffcd5u},  {13, 0x473ba19du},  {14, 0x4b0ef17fu},  {15, 0x4aa62a58u},
    {31, 0xcb0c3c82u},  {32, 0xcba00898u},  {63, 0x4c23b99bu},  {64, 0x4df10830u},
    {95, 0x48ddc9cbu},  {96, 0x4cffd93eu},  {127, 0xcca807a8u}, {128, 0x4c3c08c8u},
    {159, 0xcc6c8e41u}, {160, 0x4c978cf4u}, {191, 0x497cb718u}, {192, 0x4c7cd405u},
    {223, 0x494a8800u}, {224, 0x4cdfd50au}, {254, 0x4b1961ffu}, {255, 0x4a201f5fu},
};
} // namespace rocjitsu::matrix_test
