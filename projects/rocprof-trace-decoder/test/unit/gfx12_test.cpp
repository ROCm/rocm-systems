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
#include "gfx12/gfx12parser.h"
#include "gfx12/gfx12token.h"
#include "gfx12/gfx12wave.h"

// Tests for gfx12::inst_type
TEST(Gfx12InstTest, GetReturnsCommonType)
{
    gfx12::inst_type inst{};
    inst.raw = 0;
    inst.tm = 4;
    inst.w64h = 0;
    inst.wid = 15;
    inst.inst = 128;

    inst_type_common common = inst.get();

    EXPECT_EQ(common.tm, 4);
    EXPECT_EQ(common.w64h, 0);
    EXPECT_EQ(common.wid, 15);
    EXPECT_EQ(common.inst, 128);
}

// Tests for gfx12::wend_type
TEST(Gfx12WaveEndTest, GetReturnsCommonType)
{
    gfx12::wend_type we{};
    we.raw = 0;
    we.sa = 0;
    we.simd = 2;
    we.wgp = 7;
    we.wid = 12;

    wend_type_common common = we.get();

    EXPECT_EQ(common.sa, 0);
    EXPECT_EQ(common.simd, 2);
    EXPECT_EQ(common.wgp, 7);
    EXPECT_EQ(common.wid, 12);
}

// Tests for gfx12::wstart_type
TEST(Gfx12WaveStartTest, GetReturnsCommonType)
{
    gfx12::wstart_type ws{};
    ws.raw = 0;
    ws.sa = 0;
    ws.simd = 1;
    ws.wgp = 8;
    ws.wid = 10;
    ws.dispatcher = 0b10101; // 5 bits: pipe=01, me=1, upper bits ignored
    ws.count = 50;

    wstart_type_common common = ws.get();

    EXPECT_EQ(common.sa, 0);
    EXPECT_EQ(common.simd, 1);
    EXPECT_EQ(common.wgp, 8);
    EXPECT_EQ(common.wid, 10);
    EXPECT_EQ(common.pipe, 0b01);             // dispatcher & 0x3
    EXPECT_EQ(common.me, (0b10101 >> 2) & 1); // (dispatcher >> 2) & 1
    EXPECT_EQ(common.count, 50);
}

// Tests for new_pc_type_gfx12gfx12
TEST(Gfx12ShaderDataTest, GetReturnsCommonType)
{
    gfx12::shader_data_type sd{};
    sd.raw = 0;
    sd.sa = 1;
    sd.simd = 3;
    sd.wgp = 5;
    sd.wave = 15;
    sd.data = 0x12345678;
    sd.sdt = 1;

    shader_data_common_type common = sd.get();

    EXPECT_EQ(common.simd, 3);
    EXPECT_EQ(common.cu, 5 + (1 << ROCPROFILER_TRACE_DECODER_CU_SA_SHIFT)); // wgp + (sa << shift)
    EXPECT_EQ(common.wave, 15);
    EXPECT_EQ(common.data, 0x12345678);
    EXPECT_EQ(common.priv, 1); // sdt != 0
    EXPECT_EQ(common.isshort, 0);
}

TEST(Gfx12ShaderDataTest, PrivFlagSetWhenSdtNonZero)
{
    gfx12::shader_data_type sd{};
    sd.raw = 0;
    sd.sdt = 0;

    shader_data_common_type common0 = sd.get();
    EXPECT_EQ(common0.priv, 0);

    sd.sdt = 1;
    shader_data_common_type common1 = sd.get();
    EXPECT_EQ(common1.priv, 1);

    sd.sdt = 2;
    shader_data_common_type common2 = sd.get();
    EXPECT_EQ(common2.priv, 1);

    sd.sdt = 3;
    shader_data_common_type common3 = sd.get();
    EXPECT_EQ(common3.priv, 1);
}

// Tests for gfx12::shader_data_short_type
TEST(Gfx12ShaderDataShortTest, GetReturnsCommonTypeWithShortFlag)
{
    gfx12::shader_data_short_type sd{};
    sd.raw = 0;
    sd.sa = 0;
    sd.simd = 2;
    sd.wgp = 3;
    sd.wave = 8;
    sd.data = 0x42;
    sd.sdt = 0;

    shader_data_common_type common = sd.get();

    EXPECT_EQ(common.simd, 2);
    EXPECT_EQ(common.cu, 3); // wgp + 8*0
    EXPECT_EQ(common.wave, 8);
    EXPECT_EQ(common.data, 0x42);
    EXPECT_EQ(common.isshort, 1);
    EXPECT_EQ(common.priv, 0);
}

// Tests for gfx12::Token
TEST(Gfx12TokenTest, ConstructorSetsFields)
{
    gfx12::Token token(5000, 0xABCD1234, RdnaType::NEW_PC_GFX12);

    EXPECT_EQ(token.time, 5000);
    EXPECT_EQ(token.contents, 0xABCD1234);
    EXPECT_EQ(token.type, RdnaType::NEW_PC_GFX12);
}

TEST(Gfx12TokenTest, InheritsFromGfx11Token)
{
    gfx12::Token token(6000, 0x87654321, RdnaType::EXEC_POPCOUNT1);

    // Can be treated as gfx11::Token or gfx10::Token
    gfx11::Token* gfx11 = &token;
    gfx10::Token* gfx10 = &token;

    EXPECT_EQ(gfx11->time, 6000);
    EXPECT_EQ(gfx10->contents, 0x87654321);
}

// Tests for gfx12::wave_t::map_to_common_type
TEST(Gfx12InstMapTest, FirstEntryMapsToSalu)
{
    auto result = gfx12::wave_t::map_to_common_type(0, 1, 1);
    EXPECT_EQ(result.category, WaveInstCategory::SALU);
    EXPECT_EQ(result.cycles, 1);
}

TEST(Gfx12InstMapTest, OtherSimdRangeMapsCorrectly)
{
    // lds_other_simd_1 = 80 - now handled by map_to_common_type
    auto resultStart = gfx12::wave_t::map_to_common_type(80, 1, 1);
    EXPECT_EQ(resultStart.category, WaveInstCategory::LDS_OTHER_SIMD);
    EXPECT_EQ(resultStart.cycles, 1);

    // einst 102 is not in the table - unmapped
    auto resultEnd = gfx12::wave_t::map_to_common_type(102, 1, 1);
    EXPECT_EQ(resultEnd.category, WaveInstCategory::NONE);
    EXPECT_EQ(resultEnd.cycles, 0);
}

TEST(Gfx12InstMapTest, VmemOtherSimdRangeMapsCorrectly)
{
    // vmem_other_simd_start = 188 - now handled by map_to_common_type
    auto resultStart = gfx12::wave_t::map_to_common_type(188, 1, 1);
    EXPECT_EQ(resultStart.category, WaveInstCategory::VMEM_OTHER_SIMD);
    EXPECT_EQ(resultStart.cycles, 1);

    auto resultBeforeEnd = gfx12::wave_t::map_to_common_type(221, 1, 1);
    EXPECT_EQ(resultBeforeEnd.category, WaveInstCategory::VMEM_OTHER_SIMD);
    EXPECT_EQ(resultBeforeEnd.cycles, 34);
}

TEST(Gfx12InstMapTest, HighUnmappedInstReturnsNone)
{
    // Values between other_simd_end and vmem_other_simd_start that aren't mapped
    auto result = gfx12::wave_t::map_to_common_type(160, 1, 1);
    EXPECT_EQ(result.category, WaveInstCategory::NONE);
    EXPECT_EQ(result.cycles, 0);
}

TEST(Gfx12InstMapTest, NegativeInstReturnsNone)
{
    auto result = gfx12::wave_t::map_to_common_type(-1, 1, 1);
    EXPECT_EQ(result.category, WaveInstCategory::NONE);
    EXPECT_EQ(result.cycles, 0);
}

// Tests for other_simd values handled by map_to_common_type
TEST(Gfx12OtherSimdMapTest, BelowOtherSimdRangeNotAffected)
{
    // einst 79 is not in the other_simd range and not otherwise mapped
    auto result79 = gfx12::wave_t::map_to_common_type(79, 1, 1);
    EXPECT_EQ(result79.category, WaveInstCategory::NONE);
    EXPECT_EQ(result79.cycles, 0);
}

TEST(Gfx12OtherSimdMapTest, LdsOtherSimdMapsCorrectly)
{
    // lds_other_simd_1 = 80
    auto result80 = gfx12::wave_t::map_to_common_type(80, 1, 1);
    EXPECT_EQ(result80.category, WaveInstCategory::LDS_OTHER_SIMD);
    EXPECT_EQ(result80.cycles, 1);
}

TEST(Gfx12OtherSimdMapTest, VmemOtherSimdMapsCorrectly)
{
    // vmem_other_simd_start = 188
    auto result190 = gfx12::wave_t::map_to_common_type(190, 1, 1);
    EXPECT_EQ(result190.category, WaveInstCategory::VMEM_OTHER_SIMD);
    EXPECT_EQ(result190.cycles, 3);
}

TEST(Gfx12OtherSimdMapTest, AtBlockStoreReturnsVmem)
{
    // block_store = 222 - maps to VMEM
    auto result222 = gfx12::wave_t::map_to_common_type(222, 1, 1);
    EXPECT_EQ(result222.category, WaveInstCategory::VMEM);
    EXPECT_EQ(result222.cycles, 1);
}

// Tests for gfx12::TokenGenerator - OOB safety
TEST(Gfx12TokenGeneratorTest, ConstructorThrowsOnNullBuffer)
{
    EXPECT_THROW(gfx12::TokenGenerator(nullptr, 100, 0, 0), std::exception);
}

TEST(Gfx12TokenGeneratorTest, ConstructorThrowsOnZeroSize)
{
    uint8_t buffer[16] = {0};
    EXPECT_THROW(gfx12::TokenGenerator(buffer, 0, 0, 0), std::exception);
}

TEST(Gfx12TokenGeneratorTest, SmallBufferProcessesWithoutCrash)
{
    // Buffer needs to be large enough for the parser to read ahead safely
    // The bit-packed parser reads in chunks, so we need at least 8 bytes
    std::vector<uint8_t> buffer(16, 0);
    gfx12::TokenGenerator gen(buffer.data(), buffer.size(), 0, 0);

    // Should terminate gracefully - zero-filled buffer decodes as NOP tokens
    auto token = gen.next();
    // NOP tokens are skipped by next(), so we get whatever the final token type is
    // Just verify it processed without crashing
    EXPECT_GE(static_cast<int>(token.type), 0);
}

TEST(Gfx12TokenGeneratorTest, LargeBufferProcessesWithoutOob)
{
    std::vector<uint8_t> buffer(256, 0);
    gfx12::TokenGenerator gen(buffer.data(), buffer.size(), 0, 0);

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
TEST(Gfx12WaveStartEdgeCaseTest, MaxFieldValues)
{
    gfx12::wstart_type ws{};
    ws.raw = 0;
    ws.sa = 1;          // Max for 1-bit
    ws.simd = 3;        // Max for 2-bit
    ws.wgp = 15;        // Max for 4-bit
    ws.wid = 31;        // Max for 5-bit
    ws.dispatcher = 31; // Max for 5-bit
    ws.count = 127;     // Max for 7-bit

    wstart_type_common common = ws.get();

    EXPECT_EQ(common.sa, 1);
    EXPECT_EQ(common.simd, 3);
    EXPECT_EQ(common.wgp, 15);
    EXPECT_EQ(common.wid, 31);
    EXPECT_EQ(common.count, 127);
}

TEST(Gfx12InstEdgeCaseTest, MaxFieldValues)
{
    gfx12::inst_type inst{};
    inst.raw = 0;
    inst.tm = 7;     // Max for 3-bit
    inst.w64h = 1;   // Max for 1-bit
    inst.wid = 31;   // Max for 5-bit
    inst.inst = 255; // Max for 8-bit

    inst_type_common common = inst.get();

    EXPECT_EQ(common.tm, 7);
    EXPECT_EQ(common.w64h, 1);
    EXPECT_EQ(common.wid, 31);
    EXPECT_EQ(common.inst, 255);
}

TEST(Gfx12ShaderDataEdgeCaseTest, MaxFieldValues)
{
    gfx12::shader_data_type sd{};
    sd.raw = 0;
    sd.sa = 1;
    sd.simd = 3;
    sd.wgp = 15;
    sd.wave = 31;
    sd.data = UINT32_MAX;
    sd.sdt = 3; // Max for 2-bit

    shader_data_common_type common = sd.get();

    EXPECT_EQ(common.simd, 3);
    EXPECT_EQ(common.wave, 31);
    EXPECT_EQ(common.data, UINT32_MAX);
    EXPECT_EQ(common.priv, 1); // sdt != 0
}

// Edge case: map_to_common_type at block_store boundary - 221 is in the excluded rangeapping table
TEST(Gfx12InstMapEdgeCaseTest, DurationFromMappingTable)
{
    auto result = gfx12::wave_t::map_to_common_type(0, 100, 50);
    EXPECT_EQ(result.category, WaveInstCategory::SALU);
    // The cycles value comes from the mapping table, not the input
    EXPECT_GE(result.cycles, 0);
}

// Edge case: gfx12::wend_type max values
TEST(Gfx12WaveEndEdgeCaseTest, MaxFieldValues)
{
    gfx12::wend_type we{};
    we.raw = 0;
    we.sa = 1;
    we.simd = 3;
    we.wgp = 15;
    we.wid = 31;

    wend_type_common common = we.get();

    EXPECT_EQ(common.sa, 1);
    EXPECT_EQ(common.simd, 3);
    EXPECT_EQ(common.wgp, 15);
    EXPECT_EQ(common.wid, 31);
}

// Tests for map_to_common_type edge cases
TEST(Gfx12WaveTest, MapToCommonTypeBlockStore)
{
    auto result = gfx12::wave_t::map_to_common_type(222, 1, 1); // block_store=222
    EXPECT_EQ(result.category, WaveInstCategory::VMEM);
    EXPECT_EQ(result.cycles, 1);
}

TEST(Gfx12WaveTest, MapToCommonTypeValuDpfpAndDerate)
{
    // valu_dpfp=146 => VALU with dprate
    auto dpfp = gfx12::wave_t::map_to_common_type(146, 4, 2);
    EXPECT_EQ(dpfp.category, WaveInstCategory::VALU);
    EXPECT_EQ(dpfp.cycles, 4);

    // valu_derate=147 => VALU with dprate*derate
    auto derate = gfx12::wave_t::map_to_common_type(147, 4, 2);
    EXPECT_EQ(derate.category, WaveInstCategory::VALU);
    EXPECT_EQ(derate.cycles, 8);
}

// Tests for map_to_common_type other_simd handling
TEST(Gfx12WaveTest, MapToCommonTypeOtherSimd)
{
    // LDS other simd: lds_other_simd_1=80
    EXPECT_EQ(gfx12::wave_t::map_to_common_type(80, 1, 1).category, WaveInstCategory::LDS_OTHER_SIMD);

    // VMEM other simd: vmem_other_simd_start=188
    EXPECT_EQ(gfx12::wave_t::map_to_common_type(190, 1, 1).category, WaveInstCategory::VMEM_OTHER_SIMD);

    // Below other_simd range (79 not in table)
    EXPECT_EQ(gfx12::wave_t::map_to_common_type(79, 1, 1).category, WaveInstCategory::NONE);

    // block_store=222 maps to VMEM
    EXPECT_EQ(gfx12::wave_t::map_to_common_type(222, 1, 1).category, WaveInstCategory::VMEM);
}