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
#include "gfx10/gfx10parser.h"
#include "gfx10/gfx10token.h"
#include "gfx10/gfx10wave.h"
#include "segment.hpp"

// Tests for get_sa_wgp helper function
TEST(Gfx10HelperTest, GetSaWgpWithSaZero)
{
    uint64_t result = get_sa_wgp(0, 5);
    EXPECT_EQ(result & ROCPROFILER_TRACE_DECODER_CU_WGP_MASK, 5);
}

TEST(Gfx10HelperTest, GetSaWgpWithSaOne)
{
    uint64_t result = get_sa_wgp(1, 3);
    // SA is shifted by ROCPROFILER_TRACE_DECODER_CU_SA_SHIFT
    EXPECT_EQ(result & ROCPROFILER_TRACE_DECODER_CU_WGP_MASK, 3);
    EXPECT_NE(result, 3); // Should have SA contribution
}

// Tests for gfx10::wstart_type parsing
TEST(Gfx10WaveStartTest, GetReturnsCommonType)
{
    gfx10::wstart_type ws{};
    ws.raw = 0;
    ws.sa = 1;
    ws.simd = 2;
    ws.wgp = 4;
    ws.wid = 7;
    ws.pipe = 1;
    ws.me = 0;
    ws.count = 25;

    wstart_type_common common = ws.get();

    EXPECT_EQ(common.sa, 1);
    EXPECT_EQ(common.simd, 2);
    EXPECT_EQ(common.wgp, 4);
    EXPECT_EQ(common.wid, 7);
    EXPECT_EQ(common.pipe, 1);
    EXPECT_EQ(common.me, 0);
    EXPECT_EQ(common.count, 25);
}

// Tests for gfx10::wend_type parsing
TEST(Gfx10WaveEndTest, GetReturnsCommonType)
{
    gfx10::wend_type we{};
    we.raw = 0;
    we.sa = 0;
    we.simd = 3;
    we.wgp = 2;
    we.wid = 8;

    wend_type_common common = we.get();

    EXPECT_EQ(common.sa, 0);
    EXPECT_EQ(common.simd, 3);
    EXPECT_EQ(common.wgp, 2);
    EXPECT_EQ(common.wid, 8);
}

// Tests for gfx10::inst_type
TEST(Gfx10InstTest, GetReturnsCommonType)
{
    gfx10::inst_type inst{};
    inst.raw = 0;
    inst.tm = 7;
    inst.w64h = 0;
    inst.wid = 12;
    inst.inst = 55;

    inst_type_common common = inst.get();

    EXPECT_EQ(common.tm, 7);
    EXPECT_EQ(common.w64h, 0);
    EXPECT_EQ(common.wid, 12);
    EXPECT_EQ(common.inst, 55);
}

// Tests for gfx10::Token
TEST(Gfx10TokenTest, ConstructorSetsFields)
{
    gfx10::Token token(1000, 0xDEADBEEF, RdnaType::WAVE_START);

    EXPECT_EQ(token.time, 1000);
    EXPECT_EQ(token.contents, 0xDEADBEEF);
    EXPECT_EQ(token.type, RdnaType::WAVE_START);
}

// Tests for gfx10::CSRegisterHandler
TEST(CSRegisterHandlerGFX10Test, IsPgmLoHi)
{
    gfx10::CSRegisterHandler handler;

    EXPECT_TRUE(handler.IsPgmLo(0xC));
    EXPECT_FALSE(handler.IsPgmLo(0xD));
    EXPECT_TRUE(handler.IsPgmHi(0xD));
    EXPECT_FALSE(handler.IsPgmHi(0xC));
}

TEST(CSRegisterHandlerGFX10Test, IsUserdata)
{
    gfx10::CSRegisterHandler handler;

    EXPECT_TRUE(handler.IsUserdata(0xC340));
    EXPECT_TRUE(handler.IsUserdata(0xC341));
    EXPECT_TRUE(handler.IsUserdata(0xC342));
    EXPECT_TRUE(handler.IsUserdata(0xC343));
    EXPECT_FALSE(handler.IsUserdata(0xC33F));
    EXPECT_FALSE(handler.IsUserdata(0xC344));
}

TEST(CSRegisterHandlerGFX10Test, IsUserdataSpecific)
{
    gfx10::CSRegisterHandler handler;

    EXPECT_TRUE(handler.IsUserdata0(0xC340));
    EXPECT_FALSE(handler.IsUserdata0(0xC341));

    EXPECT_TRUE(handler.IsUserdata1(0xC341));
    EXPECT_FALSE(handler.IsUserdata1(0xC340));

    EXPECT_TRUE(handler.IsUserdata2(0xC342));
    EXPECT_FALSE(handler.IsUserdata2(0xC343));

    EXPECT_TRUE(handler.IsUserdata3(0xC343));
    EXPECT_FALSE(handler.IsUserdata3(0xC342));
}

// Tests for wstart_type_common GPU location
TEST(Gfx10WaveStartCommonTest, GetGPULocationCombinesFields)
{
    wstart_type_common ws{};
    ws.sa = 0;
    ws.simd = 2;
    ws.wgp = 3;
    ws.wid = 5;

    uint64_t loc = ws.getGPULocation();

    // Location = (SACU<<7) | (simd<<5) | wid
    // SACU = get_sa_wgp(sa, wgp) = 3
    // loc = (3<<7) | (2<<5) | 5 = 384 + 64 + 5 = 453
    uint64_t expected_sacu = get_sa_wgp(0, 3);
    uint64_t expected = (expected_sacu << 7) | (2 << 5) | 5;
    EXPECT_EQ(loc, expected);
}

// Tests for gfx10::wave_t::map_to_common_type
TEST(Gfx10InstMapTest, FirstEntryMapsToSalu)
{
    auto result = gfx10::wave_t::map_to_common_type(0, 1, 1);
    EXPECT_EQ(result.first, WaveInstCategory::SALU);
    EXPECT_EQ(result.second, 1);
}

TEST(Gfx10InstMapTest, OtherSimdRangeReturnsNone)
{
    // other_simd_start = 79, other_simd_end = 102 - boundary values
    auto resultStart = gfx10::wave_t::map_to_common_type(79, 1, 1);
    EXPECT_EQ(resultStart.first, WaveInstCategory::NONE);
    EXPECT_EQ(resultStart.second, 0);

    auto resultMid = gfx10::wave_t::map_to_common_type(90, 1, 1);
    EXPECT_EQ(resultMid.first, WaveInstCategory::NONE);
    EXPECT_EQ(resultMid.second, 0);

    auto resultEnd = gfx10::wave_t::map_to_common_type(102, 1, 1);
    EXPECT_EQ(resultEnd.first, WaveInstCategory::NONE);
    EXPECT_EQ(resultEnd.second, 0);
}

TEST(Gfx10InstMapTest, UnknownInstReturnsNone)
{
    auto result = gfx10::wave_t::map_to_common_type(999, 1, 1);
    EXPECT_EQ(result.first, WaveInstCategory::NONE);
    EXPECT_EQ(result.second, 0);
}

TEST(Gfx10InstMapTest, NegativeInstReturnsNone)
{
    auto result = gfx10::wave_t::map_to_common_type(-1, 1, 1);
    EXPECT_EQ(result.first, WaveInstCategory::NONE);
    EXPECT_EQ(result.second, 0);
}

// Tests for gfx10::TokenGenerator - OOB safety
TEST(Gfx10TokenGeneratorTest, ConstructorThrowsOnNullBuffer)
{
    EXPECT_THROW(gfx10::TokenGenerator(nullptr, 100, 0, 0), std::exception);
}

TEST(Gfx10TokenGeneratorTest, ConstructorThrowsOnZeroSize)
{
    uint8_t buffer[16] = {0};
    EXPECT_THROW(gfx10::TokenGenerator(buffer, 0, 0, 0), std::exception);
}

TEST(Gfx10TokenGeneratorTest, SmallBufferProcessesWithoutCrash)
{
    // Buffer needs to be large enough for the parser to read ahead safely
    // The bit-packed parser reads in chunks, so we need at least 8 bytes
    std::vector<uint8_t> buffer(16, 0);
    gfx10::TokenGenerator gen(buffer.data(), buffer.size(), 0, 0);

    // Should terminate gracefully
    auto token = gen.next();
    EXPECT_EQ(token.type, RdnaType::TIMESTAMP);
}

TEST(Gfx10TokenGeneratorTest, ValidTimestampTokenParsing)
{
    // TIMESTAMP encoding: {1,0,0,0,0,0,0} = 0x01, length = 64 bits
    std::vector<uint8_t> buffer(16, 0);
    buffer[0] = 0x01; // TIMESTAMP header

    gfx10::TokenGenerator gen(buffer.data(), buffer.size(), 0, 0);

    // Should complete without crashing
    auto token = gen.next();
    // TIMESTAMP should be filtered (returns final token)
    EXPECT_EQ(token.type, RdnaType::TIMESTAMP);
}

TEST(Gfx10TokenGeneratorTest, MultipleTokensNoOob)
{
    // Create a buffer with valid token patterns
    std::vector<uint8_t> buffer(64, 0);

    // Fill with NOP patterns (0x00) which are valid
    gfx10::TokenGenerator gen(buffer.data(), buffer.size(), 0, 0);

    // Consume all tokens - should not crash
    while (gen.nextValid())
    {
        auto token = gen.next();
        if (token.type == RdnaType::TIMESTAMP && token.time == 0 && token.contents == 0) break; // End marker
    }
}

TEST(Gfx10TokenGeneratorTest, LargeBufferProcessesWithoutOob)
{
    std::vector<uint8_t> buffer(256, 0);
    gfx10::TokenGenerator gen(buffer.data(), buffer.size(), 0, 0);

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
TEST(Gfx10WaveStartEdgeCaseTest, MaxFieldValues)
{
    gfx10::wstart_type ws{};
    ws.raw = 0;
    ws.sa = 1;      // Max for 1-bit
    ws.simd = 3;    // Max for 2-bit
    ws.wgp = 7;     // Max for 3-bit
    ws.wid = 31;    // Max for 5-bit
    ws.pipe = 3;    // Max for 2-bit
    ws.me = 1;      // Max for 1-bit
    ws.count = 127; // Max for 7-bit

    wstart_type_common common = ws.get();

    EXPECT_EQ(common.sa, 1);
    EXPECT_EQ(common.simd, 3);
    EXPECT_EQ(common.wgp, 7);
    EXPECT_EQ(common.wid, 31);
    EXPECT_EQ(common.pipe, 3);
    EXPECT_EQ(common.me, 1);
    EXPECT_EQ(common.count, 127);
}

TEST(Gfx10WaveEndEdgeCaseTest, MaxFieldValues)
{
    gfx10::wend_type we{};
    we.raw = 0;
    we.sa = 1;
    we.simd = 3;
    we.wgp = 7;
    we.wid = 31;

    wend_type_common common = we.get();

    EXPECT_EQ(common.sa, 1);
    EXPECT_EQ(common.simd, 3);
    EXPECT_EQ(common.wgp, 7);
    EXPECT_EQ(common.wid, 31);
}

TEST(Gfx10InstEdgeCaseTest, MaxFieldValues)
{
    gfx10::inst_type inst{};
    inst.raw = 0;
    inst.tm = 7;     // Max for 3-bit
    inst.w64h = 1;   // Max for 1-bit
    inst.wid = 31;   // Max for 5-bit
    inst.inst = 127; // Max for 7-bit

    inst_type_common common = inst.get();

    EXPECT_EQ(common.tm, 7);
    EXPECT_EQ(common.w64h, 1);
    EXPECT_EQ(common.wid, 31);
    EXPECT_EQ(common.inst, 127);
}

// Edge case: zero values
TEST(Gfx10WaveStartEdgeCaseTest, ZeroFieldValues)
{
    gfx10::wstart_type ws{};
    ws.raw = 0;

    wstart_type_common common = ws.get();

    EXPECT_EQ(common.sa, 0);
    EXPECT_EQ(common.simd, 0);
    EXPECT_EQ(common.wgp, 0);
    EXPECT_EQ(common.wid, 0);
    EXPECT_EQ(common.pipe, 0);
    EXPECT_EQ(common.me, 0);
    EXPECT_EQ(common.count, 0);
}

// Tests for map_to_common_type
TEST(Gfx10WaveTest, MapToCommonTypeBasic)
{
    EXPECT_EQ(gfx10::wave_t::map_to_common_type(0, 1, 1).first, WaveInstCategory::SALU);
    EXPECT_EQ(gfx10::wave_t::map_to_common_type(1, 1, 1).first, WaveInstCategory::SMEM);
}

// Tests for new_pc
TEST(Gfx10WaveTest, NewPcPushesEntry)
{
    gfx10::Token start_tok{0, 0, RdnaType::WAVE_START};
    gfx10::wave_t wave(0, 0, 0, pcinfo_t{100, 1}, start_tok, false);
    wave.trap_status = WaveTrapStatus::TRAP_RESTORED;

    CodeobjTableTranslator table;
    size_t before = wave.pc_infos.size();
    wave.new_pc(200, 0x1000, table);
    EXPECT_GT(wave.pc_infos.size(), before);
}
