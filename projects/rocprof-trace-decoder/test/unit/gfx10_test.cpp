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
#include <string>
#include <vector>
#include "gfx10/gfx10parser.h"
#include "gfx10/gfx10token.h"
#include "gfx10/gfx10wave.h"
#include "gfx11/gfx11token.h"
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

TEST(Gfx10TokenDiagnosticsTest, PrintAndTypeStringsExposeDecodedFields)
{
    auto has = [](const std::string& text, const char* part) { EXPECT_NE(text.find(part), std::string::npos); };

    wstart_type_common start{};
    start.sa = 1;
    start.simd = 2;
    start.wgp = 4;
    start.wid = 7;
    start.me = 1;
    start.pipe = 3;
    EXPECT_STREQ(start.typestr(), "WAVE_START");
    auto text = start.print().str();
    has(text, "wgp:4");
    has(text, "simd:2");
    has(text, "wid:7");
    has(text, "me:1");
    has(text, "pipe:3");

    wend_type_common end{};
    end.sa = 1;
    end.simd = 3;
    end.wgp = 5;
    end.wid = 8;
    EXPECT_STREQ(end.typestr(), "WAVE_END");
    has(end.print().str(), "wid:8");

    header_type header{};
    header.version = 2;
    header.NWGP = 6;
    header.DWGP = 3;
    header.DSIMD = 2;
    header.DSA = 1;
    header.UCF = 1;
    header.DPRate = 7;
    header.WSM = 2;
    EXPECT_STREQ(header.typestr(), "HEADER");
    text = header.print().str();
    has(text, "TT Version:2");
    has(text, "NWGP:6");
    has(text, "DPRate:7");

    inst_type_common inst{};
    inst.wid = 9;
    inst.inst = 55;
    inst.w64h = 1;
    EXPECT_STREQ(inst.typestr(), "INST");
    has(inst.print().str(), "inst:55");

    valu_inst_type valu{};
    valu.wid = 10;
    EXPECT_STREQ(valu.typestr(), "VALU");
    has(valu.print().str(), "wid:10");

    immed_one_type immed_one{};
    immed_one.wid = 11;
    EXPECT_STREQ(immed_one.typestr(), "IMMEDONE");
    has(immed_one.print().str(), "wid:11");

    immediate_type immediate{};
    immediate.waves = 0x1234;
    EXPECT_STREQ(immediate.typestr(), "IMMED");
    has(immediate.print().str(), "mask:0x1234");

    wave_ready_type ready{};
    ready.waves = 0x4321;
    EXPECT_STREQ(ready.typestr(), "WAVERDY");
    has(ready.print().str(), "mask:0x4321");

    gfx10::misc_type misc{};
    misc.tm = 17;
    misc.fields = 0x5A;
    EXPECT_STREQ(misc.typestr(), "MISC");
    has(misc.print().str(), "fields:0x5a");

    gfx10::new_pc_type pc{};
    pc.wave = 4;
    pc.pc = 0x123;
    EXPECT_STREQ(pc.typestr(), "NEW_PC");
    has(pc.print().str(), "0x123");

    gfx10::reg_write_type reg{};
    reg.pipe = 2;
    reg.regaddr = 0xC340;
    reg.regdata = 0xABCD;
    EXPECT_STREQ(reg.typestr(), "REG_WRITE");
    text = reg.print().str();
    has(text, "pipe:2");
    has(text, "addr:0xc340");

    gfx10::reg_init_type init{};
    init.pipe = 1;
    init.type = 2;
    init.data = 0x77;
    EXPECT_STREQ(init.typestr(), "REG_INIT");
    has(init.print().str(), "type:2");

    gfx10::event_type event{};
    event.me = 1;
    event.pipe = 2;
    event.evtype = 3;
    event.id = 7;
    EXPECT_STREQ(event.typestr(), "EVENT");
    has(event.print().str(), "id:7");

    gfx11::shader_data_type shader{};
    shader.sa = 1;
    shader.simd = 2;
    shader.wgp = 3;
    shader.wave = 4;
    shader_data_common_type common{shader};
    common.data = 0xCAFE;
    EXPECT_STREQ(common.typestr(), "SHADER_DATA");
    has(common.print().str(), "data:0xcafe");

    gfx11::shader_data_short_type shader_short{};
    shader_short.simd = 1;
    shader_short.wgp = 2;
    shader_short.wave = 3;
    shader_data_common_type short_common{shader_short};
    EXPECT_EQ(short_common.simd, 1u);
    EXPECT_EQ(short_common.wave, 3u);
}

// Tests for gfx10::CSRegisterHandler
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

// Tests for gfx10::map_to_common_type
TEST(Gfx10InstMapTest, FirstEntryMapsToSalu)
{
    auto result = gfx10::map_to_common_type(0, 1, 1);
    EXPECT_EQ(result.category, WaveInstCategory::SALU);
    EXPECT_EQ(result.cycles, 1);
}

TEST(Gfx10InstMapTest, OtherSimdRangeReturnsNone)
{
    // other_simd_start = 79, other_simd_end = 102 - boundary values
    auto resultStart = gfx10::map_to_common_type(79, 1, 1);
    EXPECT_EQ(resultStart.category, WaveInstCategory::NONE);
    EXPECT_EQ(resultStart.cycles, 0);

    auto resultMid = gfx10::map_to_common_type(90, 1, 1);
    EXPECT_EQ(resultMid.category, WaveInstCategory::NONE);
    EXPECT_EQ(resultMid.cycles, 0);

    auto resultEnd = gfx10::map_to_common_type(102, 1, 1);
    EXPECT_EQ(resultEnd.category, WaveInstCategory::NONE);
    EXPECT_EQ(resultEnd.cycles, 0);
}

TEST(Gfx10InstMapTest, UnknownInstReturnsNone)
{
    auto result = gfx10::map_to_common_type(999, 1, 1);
    EXPECT_EQ(result.category, WaveInstCategory::NONE);
    EXPECT_EQ(result.cycles, 0);
}

TEST(Gfx10InstMapTest, NegativeInstReturnsNone)
{
    auto result = gfx10::map_to_common_type(-1, 1, 1);
    EXPECT_EQ(result.category, WaveInstCategory::NONE);
    EXPECT_EQ(result.cycles, 0);
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
    EXPECT_EQ(gfx10::map_to_common_type(0, 1, 1).category, WaveInstCategory::SALU);
    EXPECT_EQ(gfx10::map_to_common_type(1, 1, 1).category, WaveInstCategory::SMEM);
}

// Tests for new_pc
TEST(Gfx10WaveTest, NewPcPushesEntry)
{
    gfx10::Token start_tok{0, 0, RdnaType::WAVE_START};
    gfx10::wave_t wave(0, 0, 0, pcinfo_t{100, 1}, start_tok, false);
    wave.trap_status = WaveTrapStatus::TRAP_RESTORED;

    CachedTable table;
    size_t before = wave.pc_infos.size();
    wave.new_pc(200, 0x1000, table);
    EXPECT_GT(wave.pc_infos.size(), before);
}
