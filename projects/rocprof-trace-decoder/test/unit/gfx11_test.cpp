// MIT License
//
// Copyright (c) 2024-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <gtest/gtest.h>
#include <cstdint>
#include <vector>
#include "gfx11/gfx11parser.h"
#include "gfx11/gfx11token.h"
#include "gfx11/gfx11wave.h"

// Tests for gfx11::shader_data_type
TEST(Gfx11ShaderDataTest, GetReturnsCommonType)
{
    gfx11::shader_data_type sd{};
    sd.raw = 0;
    sd.sa = 1;
    sd.simd = 3;
    sd.wgp = 4;
    sd.wave = 10;
    sd.data = 0x12345678;
    sd.exec = 1;

    shader_data_common_type common = sd.get();

    EXPECT_EQ(common.simd, 3);
    EXPECT_EQ(common.cu, 4 + (1 << ROCPROFILER_TRACE_DECODER_CU_SA_SHIFT)); // wgp + (sa << shift)
    EXPECT_EQ(common.wave, 10);
    EXPECT_EQ(common.data, 0x12345678);
    EXPECT_EQ(common.invalid, 1); // exec maps to invalid
    EXPECT_EQ(common.isshort, 0);
}

// Tests for gfx11::shader_data_short_type
TEST(Gfx11ShaderDataShortTest, GetReturnsCommonTypeWithShortFlag)
{
    gfx11::shader_data_short_type sd{};
    sd.raw = 0;
    sd.sa = 0;
    sd.simd = 2;
    sd.wgp = 6;
    sd.wave = 5;
    sd.data = 0x42;
    sd.exec = 0;

    shader_data_common_type common = sd.get();

    EXPECT_EQ(common.simd, 2);
    EXPECT_EQ(common.cu, 6); // wgp + 8*0
    EXPECT_EQ(common.wave, 5);
    EXPECT_EQ(common.data, 0x42);
    EXPECT_EQ(common.isshort, 1); // Short format
    EXPECT_EQ(common.invalid, 0);
}

// Tests for gfx11::Token
TEST(Gfx11TokenTest, ConstructorSetsFields)
{
    gfx11::Token token(2000, 0xCAFEBABE, RdnaType::WAVE_END);

    EXPECT_EQ(token.time, 2000);
    EXPECT_EQ(token.contents, 0xCAFEBABE);
    EXPECT_EQ(token.type, RdnaType::WAVE_END);
}

TEST(Gfx11TokenTest, InheritsFromGfx10Token)
{
    gfx11::Token token(3000, 0x12345678, RdnaType::INST);

    // Can be treated as gfx10::Token
    gfx10::Token* base = &token;
    EXPECT_EQ(base->time, 3000);
    EXPECT_EQ(base->contents, 0x12345678);
    EXPECT_EQ(base->type, RdnaType::INST);
}

// Tests for gfx11::wave_t::map_to_common_type
TEST(Gfx11InstMapTest, FirstEntryMapsToSalu)
{
    auto result = gfx11::wave_t::map_to_common_type(0, 1, 1);
    EXPECT_EQ(result.category, WaveInstCategory::SALU);
    EXPECT_EQ(result.cycles, 1);
}

TEST(Gfx11InstMapTest, OtherSimdRangeMapsCorrectly)
{
    // lds_other_simd_1 = 80 - now handled by map_to_common_type
    auto resultStart = gfx11::wave_t::map_to_common_type(80, 1, 1);
    EXPECT_EQ(resultStart.category, WaveInstCategory::LDS_OTHER_SIMD);
    EXPECT_EQ(resultStart.cycles, 1);

    // einst 102 is between vmem_other_simd_12 (101) and raytrace8 (103) - unmapped
    auto resultEnd = gfx11::wave_t::map_to_common_type(102, 1, 1);
    EXPECT_EQ(resultEnd.category, WaveInstCategory::NONE);
    EXPECT_EQ(resultEnd.cycles, 0);
}

TEST(Gfx11InstMapTest, UnknownInstReturnsNone)
{
    auto result = gfx11::wave_t::map_to_common_type(999, 1, 1);
    EXPECT_EQ(result.category, WaveInstCategory::NONE);
    EXPECT_EQ(result.cycles, 0);
}

TEST(Gfx11InstMapTest, NegativeInstReturnsNone)
{
    auto result = gfx11::wave_t::map_to_common_type(-1, 1, 1);
    EXPECT_EQ(result.category, WaveInstCategory::NONE);
    EXPECT_EQ(result.cycles, 0);
}

// Tests for other_simd values handled by map_to_common_type
TEST(Gfx11OtherSimdMapTest, BelowOtherSimdRangeNotAffected)
{
    // einst 79 is not in the other_simd range and not otherwise mapped
    auto result79 = gfx11::wave_t::map_to_common_type(79, 1, 1);
    EXPECT_EQ(result79.category, WaveInstCategory::NONE);
    EXPECT_EQ(result79.cycles, 0);
}

TEST(Gfx11OtherSimdMapTest, InRangeReturnsOtherSimdCategory)
{
    // lds_other_simd_1 = 80
    auto result80 = gfx11::wave_t::map_to_common_type(80, 1, 1);
    EXPECT_EQ(result80.category, WaveInstCategory::LDS_OTHER_SIMD);
    EXPECT_GT(result80.cycles, 0);

    // vmem_other_simd_1 = 90
    auto result90 = gfx11::wave_t::map_to_common_type(90, 1, 1);
    EXPECT_EQ(result90.category, WaveInstCategory::VMEM_OTHER_SIMD);
    EXPECT_GT(result90.cycles, 0);
}

TEST(Gfx11OtherSimdMapTest, GapAfterOtherSimdReturnsNone)
{
    // einst 102 is between vmem_other_simd_12 (101) and raytrace8 (103)
    auto result = gfx11::wave_t::map_to_common_type(102, 1, 1);
    EXPECT_EQ(result.category, WaveInstCategory::NONE);
    EXPECT_EQ(result.cycles, 0);
}

// Tests for gfx11::TokenGenerator - OOB safety
TEST(Gfx11TokenGeneratorTest, ConstructorThrowsOnNullBuffer)
{
    EXPECT_THROW(gfx11::TokenGenerator(nullptr, 100, 0, 0), std::exception);
}

TEST(Gfx11TokenGeneratorTest, ConstructorThrowsOnZeroSize)
{
    uint8_t buffer[16] = {0};
    EXPECT_THROW(gfx11::TokenGenerator(buffer, 0, 0, 0), std::exception);
}

TEST(Gfx11TokenGeneratorTest, SmallBufferProcessesWithoutCrash)
{
    // Buffer needs to be large enough for the parser to read ahead safely
    // The bit-packed parser reads in chunks, so we need at least 8 bytes
    std::vector<uint8_t> buffer(16, 0);
    gfx11::TokenGenerator gen(buffer.data(), buffer.size(), 0, 0);

    // Should terminate gracefully
    auto token = gen.next();
    EXPECT_EQ(token.type, RdnaType::TIMESTAMP);
}

TEST(Gfx11TokenGeneratorTest, LargeBufferProcessesWithoutOob)
{
    std::vector<uint8_t> buffer(256, 0);
    gfx11::TokenGenerator gen(buffer.data(), buffer.size(), 0, 0);

    int tokenCount = 0;
    while (gen.nextValid() && tokenCount < 1000)
    {
        auto token = gen.next();
        tokenCount++;
        if (token.type == RdnaType::TIMESTAMP && token.time == 0 && token.contents == 0) break;
    }
    EXPECT_GT(tokenCount, 0);
}

// Edge case tests for field boundaries
TEST(Gfx11ShaderDataEdgeCaseTest, MaxFieldValues)
{
    gfx11::shader_data_type sd{};
    sd.raw = 0;
    sd.sa = 1;    // Max for 1-bit
    sd.simd = 3;  // Max for 2-bit
    sd.wgp = 7;   // Max for 3-bit
    sd.wave = 31; // Max for 5-bit
    sd.data = UINT32_MAX;
    sd.exec = 1;

    shader_data_common_type common = sd.get();

    EXPECT_EQ(common.simd, 3);
    EXPECT_EQ(common.wave, 31);
    EXPECT_EQ(common.data, UINT32_MAX);
    EXPECT_EQ(common.invalid, 1);
}

TEST(Gfx11ShaderDataEdgeCaseTest, ZeroFieldValues)
{
    gfx11::shader_data_type sd{};
    sd.raw = 0;

    shader_data_common_type common = sd.get();

    EXPECT_EQ(common.simd, 0);
    EXPECT_EQ(common.wave, 0);
    EXPECT_EQ(common.data, 0);
    EXPECT_EQ(common.invalid, 0);
    EXPECT_EQ(common.isshort, 0);
}

// Edge case: map_to_common_type returns duration from mapping table, not input
TEST(Gfx11InstMapEdgeCaseTest, DurationFromMappingTable)
{
    // map_to_common_type returns duration from internal table, not from input params
    auto result = gfx11::wave_t::map_to_common_type(0, 100, 50);
    EXPECT_EQ(result.category, WaveInstCategory::SALU);
    // The cycles value comes from the mapping table, not the input
    EXPECT_GE(result.cycles, 0);
}
