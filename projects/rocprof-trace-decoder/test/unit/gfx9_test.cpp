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
#include <string>
#include "gfx9/gfx9token.h"
#include "gfx9/gfx9wave.h"

namespace gfx9
{
extern std::string print_token(Token& token);
}
namespace gfx9
{
extern void convertPerfEventsTs(std::vector<att_perfevent_t>& vec, int64_t interval);
}

using namespace gfx9;

// Tests for Token field parsing (using get_bits macro behavior)

TEST(Gfx9GroupIdTest, ParsesFieldsCorrectly)
{
    // cu is bits 6-9, sh is bit 5, wave is bits 10-13, simd is bits 14-15
    // cu=5 (0101), sh=1, wave=3, simd=2
    // bits: simd(14-15)=10, wave(10-13)=0011, cu(6-9)=0101, sh(5)=1
    uint64_t val = (2ull << 14) | (3ull << 10) | (5ull << 6) | (1ull << 5);

    group_id gid(val);

    EXPECT_EQ(gid.cu, 5);
    EXPECT_EQ(gid.sh, 1);
    EXPECT_EQ(gid.wave, 3);
    EXPECT_EQ(gid.simd, 2);
}

TEST(Gfx9WaveTest, ParsesFieldsCorrectly)
{
    // Additional fields: pipe(16-17), me(18-19), count(22-28)
    // pipe=2, me=1, count=50
    uint64_t val = (2ull << 14) | (3ull << 10) | (5ull << 6) | (1ull << 5); // base group_id
    val |= (2ull << 16);                                                    // pipe
    val |= (1ull << 18);                                                    // me
    val |= (50ull << 22);                                                   // count

    Wave wave(val);

    EXPECT_EQ(wave.cu, 5);
    EXPECT_EQ(wave.pipe, 2);
    EXPECT_EQ(wave.me, 1);
    EXPECT_EQ(wave.count, 50);
}

TEST(Gfx9RegTest, ParsesFieldsCorrectly)
{
    // pipe(5-6), me(7-8), regaddr(16-31), regdata(32-63)
    uint64_t val = (3ull << 5);   // pipe=3
    val |= (2ull << 7);           // me=2 -> (2+1)&1 = 1
    val |= (0x1234ull << 16);     // regaddr
    val |= (0xDEADBEEFull << 32); // regdata

    Reg reg(val);

    EXPECT_EQ(reg.pipe, 3);
    EXPECT_EQ(reg.me, 1); // (2+1)&1
    EXPECT_EQ(reg.regaddr, 0x1234);
    EXPECT_EQ(reg.regdata, 0xDEADBEEF);
}

TEST(Gfx9RegCsTest, ParsesFieldsCorrectly)
{
    // pipe(5-6), me(7-8), regaddr(9-15), regdata(16-47)
    uint64_t val = (1ull << 5);   // pipe=1
    val |= (0ull << 7);           // me=0 -> (0+1)&1 = 1
    val |= (0x42ull << 9);        // regaddr (7 bits)
    val |= (0xCAFEBABEull << 16); // regdata (32 bits)

    RegCs regcs(val);

    EXPECT_EQ(regcs.pipe, 1);
    EXPECT_EQ(regcs.me, 1); // (0+1)&1
    EXPECT_EQ(regcs.regaddr, 0x42);
    EXPECT_EQ(regcs.regdata, 0xCAFEBABE);
}

TEST(Gfx9MiscTest, ParsesFieldsCorrectly)
{
    // sh(12), misc_type(13-15)
    uint64_t val = (1ull << 12); // sh=1
    val |= (5ull << 13);         // misc_type=5

    Misc misc(val);

    EXPECT_EQ(misc.sh, 1);
    EXPECT_EQ(misc.misc_type, 5);
}

TEST(Gfx9MsgInstTest, ParsesFieldsCorrectly)
{
    // wave(5-8), simd(9-10), inst_type(11-15)
    uint64_t val = (7ull << 5); // wave=7
    val |= (3ull << 9);         // simd=3
    val |= (12ull << 11);       // inst_type=12

    MsgInst inst(val);

    EXPECT_EQ(inst.wave, 7);
    EXPECT_EQ(inst.simd, 3);
    EXPECT_EQ(inst.inst_type, 12);
}

TEST(Gfx9MsgInstPcTest, ParsesFieldsCorrectly)
{
    // wave(5-8), simd(9-10), err(15), pc(16-63)
    uint64_t val = (5ull << 5);       // wave=5
    val |= (2ull << 9);               // simd=2
    val |= (1ull << 15);              // err=1
    val |= (0x123456789ABCull << 16); // pc

    MsgInstPc instpc(val);

    EXPECT_EQ(instpc.wave, 5);
    EXPECT_EQ(instpc.simd, 2);
    EXPECT_EQ(instpc.err, 1);
    EXPECT_EQ(instpc.pc, 0x123456789ABCull);
}

TEST(Gfx9IssueTest, ParsesFieldsCorrectly)
{
    // simd(5-6), inst[0-9] each 2 bits starting at bit 8
    uint64_t val = (2ull << 5); // simd=2

    // Set inst[0]=1, inst[1]=2, inst[2]=3, rest=0
    val |= (1ull << 8);  // inst[0]
    val |= (2ull << 10); // inst[1]
    val |= (3ull << 12); // inst[2]

    Issue issue(val);

    EXPECT_EQ(issue.simd, 2);
    EXPECT_EQ(issue.inst[0], 1);
    EXPECT_EQ(issue.inst[1], 2);
    EXPECT_EQ(issue.inst[2], 3);
    EXPECT_EQ(issue.inst[3], 0);
}

TEST(Gfx9MsgPerfTest, ParsesFieldsCorrectly)
{
    // sh(5), cu(6-9), cntr_bank(10-11), cntr[0-3] each 13 bits
    uint64_t val = (1ull << 5); // sh=1
    val |= (10ull << 6);        // cu=10
    val |= (2ull << 10);        // cntr_bank=2
    val |= (100ull << 12);      // cntr[0]
    val |= (200ull << 25);      // cntr[1]
    val |= (300ull << 38);      // cntr[2]
    val |= (400ull << 51);      // cntr[3]

    MsgPerf perf(val);

    EXPECT_EQ(perf.sh, 1);
    EXPECT_EQ(perf.cu, 10);
    EXPECT_EQ(perf.cntr_bank, 2);
    EXPECT_EQ(perf.cntr[0], 100);
    EXPECT_EQ(perf.cntr[1], 200);
    EXPECT_EQ(perf.cntr[2], 300);
    EXPECT_EQ(perf.cntr[3], 400);
}

TEST(Gfx9UserDataTest, ParsesFieldsCorrectly)
{
    // Inherits from group_id, adds data(16-47)
    uint64_t val = (2ull << 14) | (3ull << 10) | (5ull << 6) | (1ull << 5); // group_id
    val |= (0xABCD1234ull << 16);                                           // data

    UserData ud(val);

    EXPECT_EQ(ud.cu, 5);
    EXPECT_EQ(ud.simd, 2);
    EXPECT_EQ(ud.data, 0xABCD1234);
}

// Tests for Token class
TEST(Gfx9TokenTest, ParsesTimeToken)
{
    // type=TOKEN_TIME (1), time in bits 16-63
    uint64_t val = TOKEN_TIME;        // type in bits 0-3
    val |= (0x123456789ABCull << 16); // time

    Token token(TOKEN_TIME, val);

    EXPECT_EQ(token.type, TOKEN_TIME);
    EXPECT_EQ(token.time, 0x123456789ABCull);
}

TEST(Gfx9TokenTest, ParsesMiscToken)
{
    uint64_t val = TOKEN_MISC;
    val |= (1ull << 12); // sh
    val |= (3ull << 13); // misc_type

    Token token(TOKEN_MISC, val);

    EXPECT_EQ(token.type, TOKEN_MISC);
    EXPECT_EQ(token.fields.misc.sh, 1);
    EXPECT_EQ(token.fields.misc.misc_type, 3);
}

TEST(Gfx9TokenTest, DeltaForMiscToken)
{
    // For type==0 (MISC), delta is bits 4-11
    uint64_t val = TOKEN_MISC;
    val |= (0xABull << 4); // delta

    Token token(TOKEN_MISC, val);

    EXPECT_EQ(token.delta, 0xAB);
}

TEST(Gfx9TokenTest, DeltaForOtherTokens)
{
    // For type!=0, delta is just bit 4
    uint64_t val = TOKEN_TIME;
    val |= (1ull << 4); // delta

    Token token(TOKEN_TIME, val);

    EXPECT_EQ(token.delta, 1);
}

// Tests for CSRegisterHandlerGFX9
TEST(CSRegisterHandlerGFX9Test, IsRegCS)
{
    CSRegisterHandlerGFX9 handler;

    EXPECT_TRUE(handler.IsRegCS(5));  // SQTT_TOKEN_REG_CS
    EXPECT_TRUE(handler.IsRegCS(15)); // SQTT_TOKEN_REG_CS_PRIV
    EXPECT_FALSE(handler.IsRegCS(2)); // SQTT_TOKEN_REG (not CS)
    EXPECT_FALSE(handler.IsRegCS(0));
}

TEST(CSRegisterHandlerGFX9Test, IsRegNoCS)
{
    CSRegisterHandlerGFX9 handler;

    EXPECT_TRUE(handler.IsRegNoCS(2));   // SQTT_TOKEN_REG
    EXPECT_FALSE(handler.IsRegNoCS(5));  // SQTT_TOKEN_REG_CS
    EXPECT_FALSE(handler.IsRegNoCS(15)); // SQTT_TOKEN_REG_CS_PRIV
}

TEST(CSRegisterHandlerGFX9Test, IsPgmLoHi)
{
    CSRegisterHandlerGFX9 handler;

    EXPECT_TRUE(handler.IsPgmLo(0xC));
    EXPECT_FALSE(handler.IsPgmLo(0xD));
    EXPECT_TRUE(handler.IsPgmHi(0xD));
    EXPECT_FALSE(handler.IsPgmHi(0xC));
}

TEST(CSRegisterHandlerGFX9Test, IsUserdata)
{
    CSRegisterHandlerGFX9 handler;

    EXPECT_TRUE(handler.IsUserdata(0xC340));
    EXPECT_TRUE(handler.IsUserdata(0xC341));
    EXPECT_TRUE(handler.IsUserdata(0xC342));
    EXPECT_TRUE(handler.IsUserdata(0xC343));
    EXPECT_FALSE(handler.IsUserdata(0xC33F));
    EXPECT_FALSE(handler.IsUserdata(0xC344));
}

TEST(CSRegisterHandlerGFX9Test, IsUserdataSpecific)
{
    CSRegisterHandlerGFX9 handler;

    EXPECT_TRUE(handler.IsUserdata0(0xC340));
    EXPECT_FALSE(handler.IsUserdata0(0xC341));

    EXPECT_TRUE(handler.IsUserdata1(0xC341));
    EXPECT_FALSE(handler.IsUserdata1(0xC340));

    EXPECT_TRUE(handler.IsUserdata2(0xC342));
    EXPECT_FALSE(handler.IsUserdata2(0xC343));

    EXPECT_TRUE(handler.IsUserdata3(0xC343));
    EXPECT_FALSE(handler.IsUserdata3(0xC342));
}

// Tests for MITokenGenerator - OOB safety and basic functionality
TEST(MITokenGeneratorTest, ConstructorThrowsOnNullBuffer)
{
    EXPECT_THROW(MITokenGenerator(nullptr, 100, 0, 0), std::exception);
}

TEST(MITokenGeneratorTest, ConstructorThrowsOnZeroSize)
{
    uint8_t buffer[16] = {0};
    EXPECT_THROW(MITokenGenerator(buffer, 0, 0, 0), std::exception);
}

TEST(MITokenGeneratorTest, ValidBufferCreatesGenerator)
{
    std::vector<uint8_t> buffer(32, 0);
    MITokenGenerator gen(buffer.data(), buffer.size(), 0, 0);

    EXPECT_TRUE(gen.valid());
}

TEST(MITokenGeneratorTest, SmallBufferProcessesWithoutCrash)
{
    // Minimum buffer size for one 16-bit token
    std::vector<uint8_t> buffer(2, 0);
    // Type 0 (MISC) = 16 bits
    buffer[0] = TOKEN_MISC; // type = 0
    buffer[1] = 0;

    MITokenGenerator gen(buffer.data(), buffer.size(), 0, 0);

    // Should process without crashing
    auto token = gen.next();
    EXPECT_EQ(token.type, TOKEN_MISC);
}

TEST(MITokenGeneratorTest, ParsesTimeToken)
{
    // TIME token is 64 bits (8 bytes)
    // Type 1 = TOKEN_TIME, length = 64 bits
    // TIME tokens are consumed internally by MITokenGenerator
    // So we need to follow with another token to verify TIME was processed
    std::vector<uint8_t> buffer(10, 0);

    // TIME token (8 bytes)
    buffer[0] = TOKEN_TIME; // type = 1
    buffer[1] = 0;
    buffer[2] = 0x12;
    buffer[3] = 0x34;
    buffer[4] = 0x56;
    buffer[5] = 0x78;
    buffer[6] = 0x9A;
    buffer[7] = 0xBC;

    // Follow with MISC token (2 bytes) to have something returned
    buffer[8] = TOKEN_MISC;
    buffer[9] = 0;

    MITokenGenerator gen(buffer.data(), buffer.size(), 0, 0);

    // TIME token is consumed internally, so next() returns MISC
    auto token = gen.next();
    EXPECT_EQ(token.type, TOKEN_MISC);
}

TEST(MITokenGeneratorTest, ParsesWaveStartToken)
{
    // WAVE_START token is 32 bits (4 bytes) per token_len_dict[3]
    // Type 3 = TOKEN_WAVE_START
    std::vector<uint8_t> buffer(4, 0);

    // Build a valid WAVE_START token
    // type(0-3)=3, delta(4)=0, sh(5)=0, cu(6-9)=5, wave(10-13)=3, simd(14-15)=2
    // pipe(16-17)=0, me(18-19)=0, count(22-28)=0
    uint32_t token_val = TOKEN_WAVE_START; // type = 3
    token_val |= (0 << 4);                 // delta
    token_val |= (0 << 5);                 // sh
    token_val |= (5 << 6);                 // cu = 5
    token_val |= (3 << 10);                // wave = 3
    token_val |= (2 << 14);                // simd = 2

    buffer[0] = token_val & 0xFF;
    buffer[1] = (token_val >> 8) & 0xFF;
    buffer[2] = (token_val >> 16) & 0xFF;
    buffer[3] = (token_val >> 24) & 0xFF;

    MITokenGenerator gen(buffer.data(), buffer.size(), 0, 0);

    auto token = gen.next();
    EXPECT_EQ(token.type, TOKEN_WAVE_START);
    EXPECT_EQ(token.fields.wave.cu, 5);
    EXPECT_EQ(token.fields.wave.wave, 3);
    EXPECT_EQ(token.fields.wave.simd, 2);
}

TEST(MITokenGeneratorTest, ParsesWaveEndToken)
{
    // WAVE_END token is 16 bits
    // Type 6 = TOKEN_WAVE_END
    std::vector<uint8_t> buffer(2, 0);

    uint16_t token_val = TOKEN_WAVE_END; // type = 6
    token_val |= (0 << 4);               // delta
    token_val |= (1 << 5);               // sh = 1
    token_val |= (7 << 6);               // cu = 7
    token_val |= (5 << 10);              // wave = 5
    token_val |= (1 << 14);              // simd = 1

    buffer[0] = token_val & 0xFF;
    buffer[1] = (token_val >> 8) & 0xFF;

    MITokenGenerator gen(buffer.data(), buffer.size(), 0, 0);

    auto token = gen.next();
    EXPECT_EQ(token.type, TOKEN_WAVE_END);
    EXPECT_EQ(token.fields.wave.cu, 7);
    EXPECT_EQ(token.fields.wave.wave, 5);
    EXPECT_EQ(token.fields.wave.simd, 1);
    EXPECT_EQ(token.fields.wave.sh, 1);
}

TEST(MITokenGeneratorTest, ParsesInstToken)
{
    // INST token is 16 bits
    // Type 10 = TOKEN_INST
    std::vector<uint8_t> buffer(2, 0);

    uint16_t token_val = TOKEN_INST; // type = 10 (0xA)
    token_val |= (1 << 4);           // delta = 1
    token_val |= (7 << 5);           // wave = 7
    token_val |= (2 << 9);           // simd = 2
    token_val |= (15 << 11);         // inst_type = 15

    buffer[0] = token_val & 0xFF;
    buffer[1] = (token_val >> 8) & 0xFF;

    MITokenGenerator gen(buffer.data(), buffer.size(), 0, 0);

    auto token = gen.next();
    EXPECT_EQ(token.type, TOKEN_INST);
    EXPECT_EQ(token.fields.inst.wave, 7);
    EXPECT_EQ(token.fields.inst.simd, 2);
    EXPECT_EQ(token.fields.inst.inst_type, 15);
}

TEST(MITokenGeneratorTest, ParsesMiscToken)
{
    // MISC token is 16 bits
    // Type 0 = TOKEN_MISC
    std::vector<uint8_t> buffer(2, 0);

    uint16_t token_val = TOKEN_MISC; // type = 0
    token_val |= (0x5A << 4);        // delta (8 bits for MISC)
    token_val |= (1 << 12);          // sh = 1
    token_val |= (3 << 13);          // misc_type = 3

    buffer[0] = token_val & 0xFF;
    buffer[1] = (token_val >> 8) & 0xFF;

    MITokenGenerator gen(buffer.data(), buffer.size(), 0, 0);

    auto token = gen.next();
    EXPECT_EQ(token.type, TOKEN_MISC);
    EXPECT_EQ(token.fields.misc.sh, 1);
    EXPECT_EQ(token.fields.misc.misc_type, 3);
    EXPECT_EQ(token.delta, 0x5A);
}

TEST(MITokenGeneratorTest, MultipleTokensInSequence)
{
    // Create buffer with multiple tokens
    // gfx9 uses byte-aligned tokens, not bit-packed like gfx10+
    // Token lengths: WAVE_START=32bits, WAVE_END=16bits
    std::vector<uint8_t> buffer;

    // Token 1: WAVE_START (32 bits = 4 bytes)
    // type(0-3)=3, delta(4)=0, sh(5)=0, cu(6-9)=2, wave(10-13)=1, simd(14-15)=0
    // pipe(16-17)=0, me(18-19)=0, count(22-28)=0
    uint32_t ws_val = TOKEN_WAVE_START; // type = 3
    ws_val |= (0 << 4);                 // delta
    ws_val |= (0 << 5);                 // sh
    ws_val |= (2 << 6);                 // cu = 2
    ws_val |= (1 << 10);                // wave = 1
    ws_val |= (0 << 14);                // simd = 0
    ws_val |= (0 << 16);                // pipe = 0
    ws_val |= (0 << 18);                // me = 0
    ws_val |= (10 << 22);               // count = 10
    buffer.push_back(ws_val & 0xFF);
    buffer.push_back((ws_val >> 8) & 0xFF);
    buffer.push_back((ws_val >> 16) & 0xFF);
    buffer.push_back((ws_val >> 24) & 0xFF);

    // Token 2: WAVE_END (16 bits = 2 bytes)
    // type(0-3)=6, delta(4)=0, sh(5)=0, cu(6-9)=2, wave(10-13)=1, simd(14-15)=0
    uint16_t we_val = TOKEN_WAVE_END; // type = 6
    we_val |= (0 << 4);               // delta
    we_val |= (0 << 5);               // sh
    we_val |= (2 << 6);               // cu = 2
    we_val |= (1 << 10);              // wave = 1
    we_val |= (0 << 14);              // simd = 0
    buffer.push_back(we_val & 0xFF);
    buffer.push_back((we_val >> 8) & 0xFF);

    MITokenGenerator gen(buffer.data(), buffer.size(), 0, 0);

    // Token 1: WAVE_START
    auto token1 = gen.next();
    EXPECT_EQ(token1.type, TOKEN_WAVE_START);
    EXPECT_EQ(token1.fields.wave.cu, 2);
    EXPECT_EQ(token1.fields.wave.wave, 1);
    EXPECT_EQ(token1.fields.wave.count, 10);

    // Token 2: WAVE_END
    auto token2 = gen.next();
    EXPECT_EQ(token2.type, TOKEN_WAVE_END);
    EXPECT_EQ(token2.fields.wave.cu, 2);
    EXPECT_EQ(token2.fields.wave.wave, 1);
}

TEST(MITokenGeneratorTest, LargeBufferProcessesWithoutOob)
{
    // Create a large buffer with valid MISC tokens (type 0, 16 bits each)
    std::vector<uint8_t> buffer(256, 0);
    // Fill with MISC tokens (type 0)
    for (size_t i = 0; i < buffer.size(); i += 2)
    {
        buffer[i] = TOKEN_MISC; // type = 0
        buffer[i + 1] = 0;
    }

    MITokenGenerator gen(buffer.data(), buffer.size(), 0, 0);

    int tokenCount = 0;
    while (gen.valid() && tokenCount < 1000)
    {
        auto token = gen.next();
        tokenCount++;
        EXPECT_EQ(token.type, TOKEN_MISC);
    }
    EXPECT_GT(tokenCount, 0);
    EXPECT_FALSE(gen.valid());
}

// Tests for parsing multiple tokens via MITokenGenerator
TEST(Gfx9TokenParseTest, ParsesMultipleTokens)
{
    std::vector<uint8_t> buffer;

    // Add a MISC token (16 bits = 2 bytes), misc_type != 1 so it is returned
    uint16_t misc_val = TOKEN_MISC;
    misc_val |= (1 << 12); // sh = 1
    misc_val |= (2 << 13); // misc_type = 2
    buffer.push_back(misc_val & 0xFF);
    buffer.push_back((misc_val >> 8) & 0xFF);

    // Add a WAVE_START token (32 bits = 4 bytes per token_len_dict[3])
    uint32_t ws_val = TOKEN_WAVE_START;
    ws_val |= (3 << 6);  // cu = 3
    ws_val |= (2 << 10); // wave = 2
    ws_val |= (1 << 14); // simd = 1
    buffer.push_back(ws_val & 0xFF);
    buffer.push_back((ws_val >> 8) & 0xFF);
    buffer.push_back((ws_val >> 16) & 0xFF);
    buffer.push_back((ws_val >> 24) & 0xFF);

    MITokenGenerator gen(buffer.data(), buffer.size(), 0, 0);

    auto token0 = gen.next();
    EXPECT_EQ(token0.type, TOKEN_MISC);

    auto token1 = gen.next();
    EXPECT_EQ(token1.type, TOKEN_WAVE_START);
    EXPECT_EQ(token1.fields.wave.cu, 3);
    EXPECT_EQ(token1.fields.wave.wave, 2);
}

// Edge case tests for field boundaries
TEST(Gfx9WaveEdgeCaseTest, MaxFieldValues)
{
    // Max values for wave fields
    // cu(6-9)=15, sh(5)=1, wave(10-13)=9 (max allowed before CHECK_WAVE throws), simd(14-15)=3
    uint64_t val = 0;
    val |= (1ull << 5);    // sh = 1
    val |= (15ull << 6);   // cu = 15 (max 4-bit)
    val |= (9ull << 10);   // wave = 9 (max before exception)
    val |= (3ull << 14);   // simd = 3 (max 2-bit)
    val |= (3ull << 16);   // pipe = 3 (max 2-bit)
    val |= (3ull << 18);   // me = 3 (max 2-bit)
    val |= (127ull << 22); // count = 127 (max 7-bit)

    Wave wave(val);

    EXPECT_EQ(wave.cu, 15);
    EXPECT_EQ(wave.sh, 1);
    EXPECT_EQ(wave.wave, 9);
    EXPECT_EQ(wave.simd, 3);
    EXPECT_EQ(wave.pipe, 3);
    EXPECT_EQ(wave.count, 127);
}

TEST(Gfx9WaveEdgeCaseTest, WaveValueAbove9Throws)
{
    // wave = 10 should throw due to CHECK_WAVE()
    uint64_t val = 0;
    val |= (10ull << 10); // wave = 10 (exceeds max of 9)

    EXPECT_THROW(Wave wave(val), std::exception);
}

TEST(Gfx9MsgInstEdgeCaseTest, MaxFieldValues)
{
    // wave(5-8)=9, simd(9-10)=3, inst_type(11-15)=31
    uint64_t val = 0;
    val |= (9ull << 5);   // wave = 9 (max before exception)
    val |= (3ull << 9);   // simd = 3
    val |= (31ull << 11); // inst_type = 31 (max 5-bit)

    MsgInst inst(val);

    EXPECT_EQ(inst.wave, 9);
    EXPECT_EQ(inst.simd, 3);
    EXPECT_EQ(inst.inst_type, 31);
}

TEST(Gfx9MsgInstPcEdgeCaseTest, MaxPcValue)
{
    // Test with large PC value
    uint64_t val = 0;
    val |= (5ull << 5);               // wave = 5
    val |= (2ull << 9);               // simd = 2
    val |= (1ull << 15);              // err = 1
    val |= (0xFFFFFFFFFFFFull << 16); // pc = max 48-bit value

    MsgInstPc instpc(val);

    EXPECT_EQ(instpc.wave, 5);
    EXPECT_EQ(instpc.err, 1);
    EXPECT_EQ(instpc.pc, 0xFFFFFFFFFFFFull);
}

TEST(Gfx9IssueEdgeCaseTest, AllInstSlotsFilled)
{
    // Fill all 10 inst slots with max value (3)
    uint64_t val = 0;
    val |= (3ull << 5); // simd = 3
    for (int i = 0; i < 10; i++)
    {
        val |= (3ull << (8 + i * 2)); // inst[i] = 3
    }

    Issue issue(val);

    EXPECT_EQ(issue.simd, 3);
    for (int i = 0; i < 10; i++) { EXPECT_EQ(issue.inst[i], 3); }
}

// Test that parseOne does not read out of bounds when a token near the end
// of the buffer requires more bytes than are available.
TEST(Gfx9OobTest, TruncatedTokenDoesNotReadPastBuffer)
{
    // Build a 6-byte buffer:
    //   bytes 0-1: WAVE_END token (type 6, 16-bit → 2 bytes, valid)
    //   bytes 2-5: TIME header (type 1 needs 64 bits = 8 bytes, but only 4 remain)
    std::vector<uint8_t> buffer(6, 0);

    // WAVE_END (type 6) with wave=3, cu=1, simd=0, sh=0
    uint16_t we_val = TOKEN_WAVE_END;
    we_val |= (1 << 6);  // cu = 1
    we_val |= (3 << 10); // wave = 3
    buffer[0] = we_val & 0xFF;
    buffer[1] = (we_val >> 8) & 0xFF;

    // TIME token header byte (type 1) — needs 8 bytes but only 4 remain
    buffer[2] = TOKEN_TIME;
    buffer[3] = 0;
    buffer[4] = 0;
    buffer[5] = 0;

    MITokenGenerator gen(buffer.data(), buffer.size(), 0, 0);

    // First next() should return the valid WAVE_END token
    auto token = gen.next();
    EXPECT_EQ(token.type, TOKEN_WAVE_END);
    EXPECT_EQ(token.fields.wave.cu, 1);
    EXPECT_EQ(token.fields.wave.wave, 3);

    // Second next() encounters the truncated TIME token.  The safe path
    // detects there are only 4 bytes remaining (needs 8) and terminates
    // without reading past the buffer.  Under ASAN this would flag a
    // heap-buffer-overflow if bounds were not checked.
    EXPECT_TRUE(gen.valid()); // bytes still in buffer
    auto sentinel = gen.next();
    EXPECT_EQ(sentinel.type, TOKEN_TIME); // sentinel / consumed
    EXPECT_FALSE(gen.valid());
}

// Tests for print_token
TEST(Gfx9PrintTokenTest, MiscWaveStartOther)
{
    // MISC with PACKET_LOST
    uint64_t misc_val = (2ull << 13);
    Token misc_tok(0, misc_val);
    misc_tok.type = TOKEN_MISC;
    auto misc_str = print_token(misc_tok);
    EXPECT_NE(misc_str.find("TOKEN_MISC"), std::string::npos);
    EXPECT_NE(misc_str.find("PACKET_LOST"), std::string::npos);

    // WAVE_START
    Token ws_tok(TOKEN_WAVE_START, (3ull << 6) | (1ull << 5));
    ws_tok.type = TOKEN_WAVE_START;
    auto ws_str = print_token(ws_tok);
    EXPECT_NE(ws_str.find("TOKEN_WAVE_START"), std::string::npos);
    EXPECT_NE(ws_str.find("start:"), std::string::npos);

    // Other token (INST) - no special fields
    Token inst_tok(TOKEN_INST, 0);
    inst_tok.type = TOKEN_INST;
    auto inst_str = print_token(inst_tok);
    EXPECT_NE(inst_str.find("TOKEN_INST"), std::string::npos);
    EXPECT_EQ(inst_str.find("misc:"), std::string::npos);
}

// Tests for CSRegisterHandlerGFX9::HandleRealtimeClock
TEST(Gfx9CSRegTest, HandleRealtimeClockNotROCM)
{
    CSRegisterHandlerGFX9 csreg;
    csreg.bIsROCMFormat = false;
    csreg.HandleRealtimeClock(1000, 0x5);
    EXPECT_TRUE(csreg.realtime.empty());
}

TEST(Gfx9CSRegTest, HandleRealtimeClockFullSequence)
{
    CSRegisterHandlerGFX9 csreg;
    csreg.bIsROCMFormat = true;

    csreg.HandleRealtimeClock(1000, 0x5); // init
    EXPECT_TRUE(csreg.realtime.empty());
    csreg.HandleRealtimeClock(1004, 0xAABBCCDD); // RT_LOW
    csreg.HandleRealtimeClock(1008, 0x11223344); // RT_HI
    csreg.HandleRealtimeClock(1012, 0xAABBCCFF); // RT_DELTA > RT_LOW => no wrap
    ASSERT_EQ(csreg.realtime.size(), 1u);
    EXPECT_EQ(csreg.realtime[0].shader_clock, 1012);
}

TEST(Gfx9CSRegTest, HandleRealtimeClockWrapping)
{
    CSRegisterHandlerGFX9 csreg;
    csreg.bIsROCMFormat = true;

    csreg.HandleRealtimeClock(1000, 0x5);
    csreg.HandleRealtimeClock(1004, 0xFFFFFFF0); // RT_LOW
    csreg.HandleRealtimeClock(1008, 0x00000001); // RT_HI
    csreg.HandleRealtimeClock(1012, 0x00000010); // RT_DELTA < RT_LOW => wrap
    ASSERT_EQ(csreg.realtime.size(), 1u);
    uint64_t expected = 0x00000010 | ((0x00000001ull << 32) + (1ull << 32));
    EXPECT_EQ(csreg.realtime[0].realtime_clock, expected);
}

TEST(Gfx9CSRegTest, HandleRealtimeClockNonInitIgnored)
{
    CSRegisterHandlerGFX9 csreg;
    csreg.bIsROCMFormat = true;
    csreg.HandleRealtimeClock(1000, 0x3); // data != 0x5 when count==0
    EXPECT_TRUE(csreg.realtime.empty());
}

// Tests for apply_issue
TEST(Gfx9WaveTest, ApplyIssueAllPaths)
{
    auto make_wave = []()
    {
        Wave ws(0);
        ws.cu = 0;
        ws.simd = 0;
        ws.wave = 0;
        return wave_t(0, pcinfo_t{100, 1}, ws, 0);
    };

    // IMMED
    {
        auto w = make_wave();
        w.instructions.push_back(Instruction(0, WaveInstCategory::VALU, 4, 0));
        EXPECT_EQ(w.apply_issue(3, 100), 0);
        ASSERT_GE(w.instructions.size(), 2u);
        EXPECT_EQ(w.instructions.back().category, WaveInstCategory::IMMED);
    }
    // STALL
    {
        auto w = make_wave();
        EXPECT_EQ(w.apply_issue(1, 100), 0);
        EXPECT_EQ(w.stall_start_times.back(), 100);
    }
    // INST
    {
        auto w = make_wave();
        EXPECT_EQ(w.apply_issue(2, 100), 1);
        EXPECT_EQ(w.instructions.back().category, WaveInstCategory::WAVE_NOT_FINISHED);
    }
    // Not restored
    {
        auto w = make_wave();
        w.trap_status = WaveTrapStatus::TRAP_STANDBY;
        EXPECT_EQ(w.apply_issue(2, 100), 0);
        EXPECT_TRUE(w.instructions.empty());
    }
}

// Tests for apply_inst type dispatch
class Gfx9ApplyInstTest : public ::testing::Test
{
protected:
    wave_t make_wave()
    {
        Wave ws(0);
        ws.cu = 0;
        ws.simd = 0;
        ws.wave = 0;
        wave_t w(0, pcinfo_t{100, 1}, ws, 0);
        w.apply_issue(2, 100);
        return w;
    }

    unsigned int apply_type(int inst_type)
    {
        auto w = make_wave();
        Token tok(TOKEN_INST, 0);
        tok.type = TOKEN_INST;
        tok.time = 104;
        tok.fields.inst.inst_type = inst_type;
        tok.fields.inst.simd = 0;
        tok.fields.inst.wave = 0;
        int64_t phase = 15;
        w.apply_inst(tok, phase);
        for (auto& i : w.instructions)
            if (i.category != WaveInstCategory::WAVE_NOT_FINISHED) return i.category;
        return WaveInstCategory::NONE;
    }
};

TEST_F(Gfx9ApplyInstTest, AllInstTypes)
{
    EXPECT_EQ(apply_type(0), WaveInstCategory::SMEM);  // SMEM_RD
    EXPECT_EQ(apply_type(1), WaveInstCategory::SALU);  // SALU_32
    EXPECT_EQ(apply_type(2), WaveInstCategory::VMEM);  // VMEM_RD
    EXPECT_EQ(apply_type(5), WaveInstCategory::VALU);  // VALU_32
    EXPECT_EQ(apply_type(6), WaveInstCategory::LDS);   // LDS
    EXPECT_EQ(apply_type(12), WaveInstCategory::JUMP); // JUMP
    EXPECT_EQ(apply_type(13), WaveInstCategory::NEXT); // NEXT
    EXPECT_EQ(apply_type(14), WaveInstCategory::FLAT); // FLAT_RD
    EXPECT_EQ(apply_type(15), WaveInstCategory::MSG);  // OTHER_MSG
    EXPECT_EQ(apply_type(18), WaveInstCategory::VALU); // VALU_64
}

TEST_F(Gfx9ApplyInstTest, NotRestoredReturnsEarly)
{
    auto w = make_wave();
    w.trap_status = WaveTrapStatus::TRAP_STANDBY;
    w.instructions.clear();
    Token tok(TOKEN_INST, 0);
    tok.type = TOKEN_INST;
    tok.time = 104;
    tok.fields.inst.inst_type = 1;
    int64_t phase = 15;
    w.apply_inst(tok, phase);
    EXPECT_TRUE(w.instructions.empty());
}

TEST_F(Gfx9ApplyInstTest, GoToTrapAndPCRestore)
{
    auto w = make_wave();

    // GO_TO_TRAP
    Token trap_tok(TOKEN_INST, 0);
    trap_tok.type = TOKEN_INST;
    trap_tok.time = 104;
    trap_tok.fields.inst.inst_type = 29;
    int64_t phase = 15;
    w.apply_inst(trap_tok, phase);
    EXPECT_EQ(w.trap_status, WaveTrapStatus::TRAP_WAIT_FOR_NEW_PC);

    // Issue + PC type restores
    w.apply_issue(2, 200);
    Token pc_tok(TOKEN_INST, 0);
    pc_tok.type = TOKEN_INST;
    pc_tok.time = 204;
    pc_tok.fields.inst.inst_type = 7;
    w.apply_inst(pc_tok, phase);
    EXPECT_EQ(w.trap_status, WaveTrapStatus::TRAP_RESTORED);
}

TEST_F(Gfx9ApplyInstTest, StallApplied)
{
    Wave ws(0);
    ws.cu = 0;
    ws.simd = 0;
    ws.wave = 0;
    wave_t w(0, pcinfo_t{100, 1}, ws, 0);
    w.apply_issue(1, 50);  // STALL
    w.apply_issue(2, 100); // INST
    Token tok(TOKEN_INST, 0);
    tok.type = TOKEN_INST;
    tok.time = 200;
    tok.fields.inst.inst_type = 5;
    int64_t phase = 15;
    w.apply_inst(tok, phase);
    bool found = false;
    for (auto& i : w.instructions)
        if (i.category == WaveInstCategory::VALU && i.stall > 0) found = true;
    EXPECT_TRUE(found);
}

TEST_F(Gfx9ApplyInstTest, SmemRdReplay)
{
    auto w = make_wave();
    // First issue a SMEM_RD (inst_type=0) so last_inst_for_xnack[0] is set
    Token smem_tok(TOKEN_INST, 0);
    smem_tok.type = TOKEN_INST;
    smem_tok.time = 104;
    smem_tok.fields.inst.inst_type = 0; // SMEM_RD
    smem_tok.fields.inst.simd = 0;
    smem_tok.fields.inst.wave = 0;
    int64_t phase = 15;
    w.apply_inst(smem_tok, phase);

    // Issue another instruction slot
    w.apply_issue(2, 200);

    // Now issue SMEM_RD_REPLAY (inst_type=19) to trigger the replay/xnack path
    Token replay_tok(TOKEN_INST, 0);
    replay_tok.type = TOKEN_INST;
    replay_tok.time = 300;
    replay_tok.fields.inst.inst_type = 19; // SMEM_RD_REPLAY
    replay_tok.fields.inst.simd = 0;
    replay_tok.fields.inst.wave = 0;
    w.apply_inst(replay_tok, phase);

    // The replay should erase the instruction (not add a new category)
    bool found_replay_cat = false;
    for (auto& i : w.instructions)
        if (i.category == 19) found_replay_cat = true;
    EXPECT_FALSE(found_replay_cat);
}

// Tests for convertPerfEventsTs
TEST(Gfx9PerfEventsTest, InsertsGapEvents)
{
    std::vector<att_perfevent_t> vec;
    att_perfevent_t ev1{};
    ev1.time = 100;
    ev1.CU = 0;
    ev1.bank = 0;
    att_perfevent_t ev2{};
    ev2.time = 300;
    ev2.CU = 0;
    ev2.bank = 0;
    vec.push_back(ev1);
    vec.push_back(ev2);
    convertPerfEventsTs(vec, 50);
    EXPECT_GT(vec.size(), 2u);
}
