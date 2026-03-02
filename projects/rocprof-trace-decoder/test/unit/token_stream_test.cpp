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
#include <cstring>
#include <vector>
#include "gfx10/gfx10parser.h"
#include "gfx10/gfx10token.h"
#include "gfx11/gfx11parser.h"
#include "gfx11/gfx11token.h"
#include "gfx12/gfx12parser.h"
#include "gfx12/gfx12token.h"

// Helper class to build token streams bit by bit
class TokenStreamBuilder
{
public:
    TokenStreamBuilder() : bit_pos(0) {}

    // Write bits LSB first (matching the token encoding format)
    void writeBits(uint64_t value, int num_bits)
    {
        for (int i = 0; i < num_bits; i++)
        {
            size_t byte_idx = bit_pos / 8;
            size_t bit_idx = bit_pos % 8;

            // Ensure buffer is large enough
            if (byte_idx >= buffer.size()) buffer.resize(byte_idx + 1, 0);

            // Write bit (LSB first)
            if (value & (1ull << i)) buffer[byte_idx] |= (1 << bit_idx);

            bit_pos++;
        }
    }

    // Pad to next byte boundary
    void padToByteBoundary()
    {
        while (bit_pos % 8 != 0)
        {
            size_t byte_idx = bit_pos / 8;
            if (byte_idx >= buffer.size()) buffer.resize(byte_idx + 1, 0);
            bit_pos++;
        }
    }

    const uint8_t* data() const { return buffer.data(); }
    size_t size() const { return buffer.size(); }

    // Finalize the buffer - pad to byte boundary
    void finalize() { padToByteBoundary(); }

private:
    std::vector<uint8_t> buffer;
    size_t bit_pos;
};

// Bit encodings for gfx10 tokens (LSB first)
// VALU_INST: {1,1,0} = 0b011 (3 bits)
// TIME: {0,0,0,1} = 0b1000 (4 bits)
// NOP: {0,0,0,0} = 0b0000 (4 bits)
// WAVE_START: {0,0,1,1,0} = 0b01100 (5 bits)
// WAVE_END: {1,0,1,0,1} = 0b10101 (5 bits)
// INST: {0,1,0} = 0b010 (3 bits)

//=============================================================================
// GFX10 Token Stream Tests
// These tests verify token decoding with exact buffer sizes (no byte padding).
// The parser uses readOne_unsafe() when 64+ bits of padding remain, and
// readOne_safe() when near buffer end (returns 0 for out-of-bounds reads).
//=============================================================================

TEST(Gfx10TokenStreamTest, DecodesSingleValuInst)
{
    TokenStreamBuilder builder;

    // VALU_INST encoding: {1,1,0} = 0b011, token length = 12 bits
    uint64_t token_data = 0;
    token_data |= 0b011;        // header (3 bits): VALU_INST
    token_data |= (0b101 << 3); // tm (3 bits): time delta = 5
    token_data |= (0 << 6);     // w64h (1 bit)
    token_data |= (7 << 7);     // wid (5 bits): wave id = 7

    builder.writeBits(token_data, 12);
    builder.finalize(); // Align to byte for valid buffer

    gfx10::TokenGenerator gen(builder.data(), builder.size(), 0, 0);

    auto token = gen.next();
    EXPECT_EQ(token.type, RdnaType::VALU_INST);

    valu_inst_type valu{.raw = token.contents};
    EXPECT_EQ(valu.wid, 7);
}

TEST(Gfx10TokenStreamTest, DecodesMultipleTokens)
{
    TokenStreamBuilder builder;

    // Token 1: VALU_INST (12 bits)
    uint64_t valu_data = 0;
    valu_data |= 0b011;        // header: VALU_INST
    valu_data |= (0b010 << 3); // tm = 2
    valu_data |= (0 << 6);     // w64h
    valu_data |= (5 << 7);     // wid = 5
    builder.writeBits(valu_data, 12);

    // Token 2: IMM_ONE (12 bits)
    uint64_t imm_data = 0;
    imm_data |= 0b1101;       // header: IMM_ONE
    imm_data |= (0b010 << 4); // tm = 2
    imm_data |= (13 << 7);    // wid = 13
    builder.writeBits(imm_data, 12);

    builder.finalize();

    gfx10::TokenGenerator gen(builder.data(), builder.size(), 0, 0);

    auto token1 = gen.next();
    EXPECT_EQ(token1.type, RdnaType::VALU_INST);
    valu_inst_type valu{.raw = token1.contents};
    EXPECT_EQ(valu.wid, 5);

    auto token2 = gen.next();
    EXPECT_EQ(token2.type, RdnaType::IMM_ONE);
    immed_one_type imm{.raw = token2.contents};
    EXPECT_EQ(imm.wid, 13);
}

TEST(Gfx10TokenStreamTest, DecodesWaveStartToken)
{
    TokenStreamBuilder builder;

    // WAVE_START: 32 bits total
    uint64_t ws_data = 0;
    ws_data |= 0b01100;       // header: WAVE_START (5 bits)
    ws_data |= (0b11 << 5);   // tm = 3 (2 bits)
    ws_data |= (1ull << 7);   // sa = 1
    ws_data |= (2ull << 8);   // simd = 2 (2 bits)
    ws_data |= (5ull << 10);  // wgp = 5 (3 bits)
    ws_data |= (10ull << 13); // wid = 10 (5 bits)
    ws_data |= (1ull << 18);  // queue = 1 (3 bits)
    ws_data |= (2ull << 21);  // pipe = 2 (2 bits)
    ws_data |= (1ull << 23);  // me = 1
    ws_data |= (0ull << 24);  // dispatcher = 0
    ws_data |= (15ull << 25); // count = 15 (7 bits)

    builder.writeBits(ws_data, 32);
    // 32 bits = 4 bytes, already aligned

    gfx10::TokenGenerator gen(builder.data(), builder.size(), 0, 0);

    auto token = gen.next();
    EXPECT_EQ(token.type, RdnaType::WAVE_START);

    gfx10::wstart_type ws{.raw = token.contents};
    EXPECT_EQ(ws.sa, 1);
    EXPECT_EQ(ws.simd, 2);
    EXPECT_EQ(ws.wgp, 5);
    EXPECT_EQ(ws.wid, 10);
    EXPECT_EQ(ws.pipe, 2);
    EXPECT_EQ(ws.me, 1);
}

TEST(Gfx10TokenStreamTest, DecodesWaveEndToken)
{
    TokenStreamBuilder builder;

    // WAVE_END: 20 bits
    uint64_t we_data = 0;
    we_data |= 0b10101;       // header: WAVE_END (5 bits)
    we_data |= (0b011 << 5);  // tm = 3 (3 bits)
    we_data |= (0ull << 8);   // sa = 0
    we_data |= (3ull << 9);   // simd = 3 (2 bits)
    we_data |= (7ull << 11);  // wgp = 7 (3 bits)
    we_data |= (0ull << 14);  // unused
    we_data |= (12ull << 15); // wid = 12 (5 bits)

    builder.writeBits(we_data, 20);
    builder.finalize(); // 20 bits -> 3 bytes

    gfx10::TokenGenerator gen(builder.data(), builder.size(), 0, 0);

    auto token = gen.next();
    EXPECT_EQ(token.type, RdnaType::WAVE_END);

    gfx10::wend_type we{.raw = token.contents};
    EXPECT_EQ(we.sa, 0);
    EXPECT_EQ(we.simd, 3);
    EXPECT_EQ(we.wgp, 7);
    EXPECT_EQ(we.wid, 12);
}

TEST(Gfx10TokenStreamTest, DecodesInstToken)
{
    TokenStreamBuilder builder;

    // INST: 20 bits
    uint64_t inst_data = 0;
    inst_data |= 0b010;         // header: INST (3 bits)
    inst_data |= (0ull << 3);   // unused (1 bit)
    inst_data |= (0b101 << 4);  // tm = 5 (3 bits)
    inst_data |= (1ull << 7);   // w64h = 1
    inst_data |= (15ull << 8);  // wid = 15 (5 bits)
    inst_data |= (42ull << 13); // inst = 42 (7 bits)

    builder.writeBits(inst_data, 20);
    builder.finalize();

    gfx10::TokenGenerator gen(builder.data(), builder.size(), 0, 0);

    auto token = gen.next();
    EXPECT_EQ(token.type, RdnaType::INST);

    gfx10::inst_type inst{.raw = token.contents};
    EXPECT_EQ(inst.w64h, 1);
    EXPECT_EQ(inst.wid, 15);
    EXPECT_EQ(inst.inst, 42);
}

TEST(Gfx10TokenStreamTest, SkipsNopTokens)
{
    TokenStreamBuilder builder;

    // NOP: 4 bits
    builder.writeBits(0b0000, 4);

    // VALU_INST: 12 bits
    uint64_t valu_data = 0;
    valu_data |= 0b011;
    valu_data |= (0b001 << 3);
    valu_data |= (0 << 6);
    valu_data |= (3 << 7);
    builder.writeBits(valu_data, 12);

    builder.finalize();

    gfx10::TokenGenerator gen(builder.data(), builder.size(), 0, 0);

    auto token = gen.next();
    EXPECT_EQ(token.type, RdnaType::VALU_INST);
    valu_inst_type valu{.raw = token.contents};
    EXPECT_EQ(valu.wid, 3);
}

TEST(Gfx10TokenStreamTest, TimeTokenUpdatesGlobalTime)
{
    TokenStreamBuilder builder;

    // TIME: 8 bits
    uint64_t time_data = 0;
    time_data |= 0b1000;
    time_data |= (0b1010 << 4);
    builder.writeBits(time_data, 8);

    // VALU_INST: 12 bits
    uint64_t valu_data = 0;
    valu_data |= 0b011;
    valu_data |= (0b000 << 3);
    valu_data |= (0 << 6);
    valu_data |= (1 << 7);
    builder.writeBits(valu_data, 12);

    builder.finalize();

    gfx10::TokenGenerator gen(builder.data(), builder.size(), 0, 0);

    auto token = gen.next();
    EXPECT_EQ(token.type, RdnaType::VALU_INST);
    EXPECT_GT(token.time, 0);
}

//=============================================================================
// GFX11 Token Stream Tests
//=============================================================================

TEST(Gfx11TokenStreamTest, DecodesSingleValuInst)
{
    TokenStreamBuilder builder;

    uint64_t token_data = 0;
    token_data |= 0b011;
    token_data |= (0b100 << 3);
    token_data |= (1 << 6);
    token_data |= (9 << 7);

    builder.writeBits(token_data, 12);
    builder.finalize();

    gfx11::TokenGenerator gen(builder.data(), builder.size(), 0, 0);

    auto token = gen.next();
    EXPECT_EQ(token.type, RdnaType::VALU_INST);

    valu_inst_type valu{.raw = token.contents};
    EXPECT_EQ(valu.wid, 9);
    EXPECT_EQ(valu.w64h, 1);
}

TEST(Gfx11TokenStreamTest, DecodesMultipleTokensSequence)
{
    TokenStreamBuilder builder;

    // WAVE_START (32 bits)
    uint64_t ws_data = 0;
    ws_data |= 0b01100;
    ws_data |= (0b10 << 5);
    ws_data |= (0ull << 7);
    ws_data |= (1ull << 8);
    ws_data |= (3ull << 10);
    ws_data |= (4ull << 13);
    ws_data |= (0ull << 18);
    ws_data |= (1ull << 21);
    ws_data |= (0ull << 23);
    ws_data |= (0ull << 24);
    ws_data |= (20ull << 25);
    builder.writeBits(ws_data, 32);

    // INST (20 bits)
    uint64_t inst_data = 0;
    inst_data |= 0b010;
    inst_data |= (0ull << 3);
    inst_data |= (0b011 << 4);
    inst_data |= (0ull << 7);
    inst_data |= (4ull << 8);
    inst_data |= (10ull << 13);
    builder.writeBits(inst_data, 20);

    // WAVE_END (20 bits)
    uint64_t we_data = 0;
    we_data |= 0b10101;
    we_data |= (0b010 << 5);
    we_data |= (0ull << 8);
    we_data |= (1ull << 9);
    we_data |= (3ull << 11);
    we_data |= (0ull << 14);
    we_data |= (4ull << 15);
    builder.writeBits(we_data, 20);

    builder.finalize();

    gfx11::TokenGenerator gen(builder.data(), builder.size(), 0, 0);

    auto token1 = gen.next();
    EXPECT_EQ(token1.type, RdnaType::WAVE_START);
    gfx10::wstart_type ws{.raw = token1.contents};
    EXPECT_EQ(ws.wid, 4);
    EXPECT_EQ(ws.wgp, 3);

    auto token2 = gen.next();
    EXPECT_EQ(token2.type, RdnaType::INST);
    gfx10::inst_type inst{.raw = token2.contents};
    EXPECT_EQ(inst.wid, 4);
    EXPECT_EQ(inst.inst, 10);

    auto token3 = gen.next();
    EXPECT_EQ(token3.type, RdnaType::WAVE_END);
    gfx10::wend_type we{.raw = token3.contents};
    EXPECT_EQ(we.wid, 4);
}

//=============================================================================
// GFX12 Token Stream Tests
//=============================================================================

TEST(Gfx12TokenStreamTest, DecodesSingleValuInst)
{
    TokenStreamBuilder builder;

    uint64_t token_data = 0;
    token_data |= 0b011;
    token_data |= (0b110 << 3);
    token_data |= (0 << 6);
    token_data |= (11 << 7);

    builder.writeBits(token_data, 12);
    builder.finalize();

    gfx12::TokenGenerator gen(builder.data(), builder.size(), 0, 0);

    auto token = gen.next();
    EXPECT_EQ(token.type, RdnaType::VALU_INST);

    valu_inst_type valu{.raw = token.contents};
    EXPECT_EQ(valu.wid, 11);
}

TEST(Gfx12TokenStreamTest, DecodesWaveStartGfx12)
{
    TokenStreamBuilder builder;

    uint64_t ws_data = 0;
    ws_data |= 0b01100;
    ws_data |= (0b01 << 5);
    ws_data |= (1ull << 7);
    ws_data |= (0ull << 8);
    ws_data |= (6ull << 10);
    ws_data |= (0ull << 14);
    ws_data |= (8ull << 15);
    ws_data |= (3ull << 20);
    ws_data |= (25ull << 25);

    builder.writeBits(ws_data, 32);

    gfx12::TokenGenerator gen(builder.data(), builder.size(), 0, 0);

    auto token = gen.next();
    EXPECT_EQ(token.type, RdnaType::WAVE_START);

    gfx12::wstart_type ws{.raw = token.contents};
    EXPECT_EQ(ws.sa, 1);
    EXPECT_EQ(ws.simd, 0);
    EXPECT_EQ(ws.wgp, 6);
    EXPECT_EQ(ws.wid, 8);
}

TEST(Gfx12TokenStreamTest, DecodesInstGfx12)
{
    TokenStreamBuilder builder;

    uint64_t inst_data = 0;
    inst_data |= 0b010;
    inst_data |= (0b100 << 3);
    inst_data |= (1ull << 6);
    inst_data |= (20ull << 7);
    inst_data |= (100ull << 12);

    builder.writeBits(inst_data, 20);
    builder.finalize();

    gfx12::TokenGenerator gen(builder.data(), builder.size(), 0, 0);

    auto token = gen.next();
    EXPECT_EQ(token.type, RdnaType::INST);

    gfx12::inst_type inst{.raw = token.contents};
    EXPECT_EQ(inst.w64h, 1);
    EXPECT_EQ(inst.wid, 20);
    EXPECT_EQ(inst.inst, 100);
}

TEST(Gfx12TokenStreamTest, DecodesCompleteWaveLifecycle)
{
    TokenStreamBuilder builder;

    // WAVE_START (32 bits)
    uint64_t ws_data = 0;
    ws_data |= 0b01100;
    ws_data |= (0b01 << 5);
    ws_data |= (0ull << 7);
    ws_data |= (2ull << 8);
    ws_data |= (4ull << 10);
    ws_data |= (0ull << 14);
    ws_data |= (5ull << 15);
    ws_data |= (0ull << 20);
    ws_data |= (10ull << 25);
    builder.writeBits(ws_data, 32);

    // 3x INST (20 bits each)
    for (int i = 0; i < 3; i++)
    {
        uint64_t inst_data = 0;
        inst_data |= 0b010;
        inst_data |= ((i + 1) << 3);
        inst_data |= (0ull << 6);
        inst_data |= (5ull << 7);
        inst_data |= ((i * 10) << 12);
        builder.writeBits(inst_data, 20);
    }

    // WAVE_END (20 bits)
    uint64_t we_data = 0;
    we_data |= 0b10101;
    we_data |= (0b001 << 5);
    we_data |= (0ull << 8);
    we_data |= (2ull << 9);
    we_data |= (4ull << 11);
    we_data |= (0ull << 14);
    we_data |= (5ull << 15);
    builder.writeBits(we_data, 20);

    builder.finalize();

    gfx12::TokenGenerator gen(builder.data(), builder.size(), 0, 0);

    auto token = gen.next();
    EXPECT_EQ(token.type, RdnaType::WAVE_START);

    for (int i = 0; i < 3; i++)
    {
        token = gen.next();
        EXPECT_EQ(token.type, RdnaType::INST);
    }

    token = gen.next();
    EXPECT_EQ(token.type, RdnaType::WAVE_END);
}

//=============================================================================
// Edge Cases - Basic
//=============================================================================

TEST(TokenStreamEdgeCaseTest, HandlesNopPadding)
{
    TokenStreamBuilder builder;

    // Multiple NOPs (4 bits each) followed by a real token
    for (int i = 0; i < 5; i++) builder.writeBits(0b0000, 4);

    // VALU_INST (12 bits)
    uint64_t valu_data = 0;
    valu_data |= 0b011;
    valu_data |= (0b111 << 3);
    valu_data |= (0 << 6);
    valu_data |= (1 << 7);
    builder.writeBits(valu_data, 12);

    builder.finalize();

    gfx10::TokenGenerator gen(builder.data(), builder.size(), 0, 0);

    auto token = gen.next();
    EXPECT_EQ(token.type, RdnaType::VALU_INST);
}

TEST(TokenStreamEdgeCaseTest, TimeAccumulatesCorrectly)
{
    TokenStreamBuilder builder;

    // Two VALU_INST tokens (12 bits each)
    uint64_t valu1 = 0;
    valu1 |= 0b011;
    valu1 |= (0b000 << 3);
    valu1 |= (0 << 6);
    valu1 |= (1 << 7);
    builder.writeBits(valu1, 12);

    uint64_t valu2 = 0;
    valu2 |= 0b011;
    valu2 |= (0b101 << 3);
    valu2 |= (0 << 6);
    valu2 |= (2 << 7);
    builder.writeBits(valu2, 12);

    builder.finalize();

    gfx10::TokenGenerator gen(builder.data(), builder.size(), 0, 0);

    auto token1 = gen.next();
    auto token2 = gen.next();

    EXPECT_GT(token2.time, token1.time);
}

TEST(TokenStreamEdgeCaseTest, AllNopBuffer)
{
    TokenStreamBuilder builder;

    // Buffer full of NOPs (4 bits each)
    for (int i = 0; i < 32; i++) builder.writeBits(0b0000, 4);

    gfx10::TokenGenerator gen(builder.data(), builder.size(), 0, 0);

    int iterations = 0;
    while (gen.nextValid() && iterations < 100)
    {
        gen.next();
        iterations++;
    }
    EXPECT_LT(iterations, 100);
}

TEST(TokenStreamEdgeCaseTest, MaxBitfieldValues)
{
    TokenStreamBuilder builder;

    // VALU_INST with max values
    uint64_t valu = 0;
    valu |= 0b011;
    valu |= (0b111 << 3);
    valu |= (1 << 6);
    valu |= (31 << 7);
    builder.writeBits(valu, 12);

    builder.finalize();

    gfx10::TokenGenerator gen(builder.data(), builder.size(), 0, 0);

    auto token = gen.next();
    EXPECT_EQ(token.type, RdnaType::VALU_INST);

    valu_inst_type parsed{.raw = token.contents};
    EXPECT_EQ(parsed.wid, 31);
    EXPECT_EQ(parsed.w64h, 1);
    EXPECT_EQ(parsed.tm, 7);
}

TEST(TokenStreamEdgeCaseTest, AlternatingTokenTypes)
{
    TokenStreamBuilder builder;

    for (int i = 0; i < 5; i++)
    {
        // VALU_INST (12 bits)
        uint64_t valu = 0;
        valu |= 0b011;
        valu |= (i << 3);
        valu |= (0 << 6);
        valu |= (i << 7);
        builder.writeBits(valu, 12);

        // INST (20 bits)
        uint64_t inst = 0;
        inst |= 0b010;
        inst |= (0 << 3);
        inst |= (i << 4);
        inst |= (0 << 7);
        inst |= (i << 8);
        inst |= (i << 13);
        builder.writeBits(inst, 20);
    }

    builder.finalize();

    gfx10::TokenGenerator gen(builder.data(), builder.size(), 0, 0);

    for (int i = 0; i < 5; i++)
    {
        auto valu_token = gen.next();
        EXPECT_EQ(valu_token.type, RdnaType::VALU_INST);

        auto inst_token = gen.next();
        EXPECT_EQ(inst_token.type, RdnaType::INST);
    }
}

TEST(Gfx10TokenStreamEdgeCaseTest, TimestampToken)
{
    TokenStreamBuilder builder;

    // TIMESTAMP (64 bits)
    uint64_t timestamp_value = 0x123456789ABCull;
    uint64_t ts_data = 0;
    ts_data |= 0b0000001;
    ts_data |= (timestamp_value << 7);
    builder.writeBits(ts_data, 64);

    // VALU_INST (12 bits)
    uint64_t valu = 0;
    valu |= 0b011;
    valu |= (0 << 3);
    valu |= (0 << 6);
    valu |= (1 << 7);
    builder.writeBits(valu, 12);

    builder.finalize();

    gfx10::TokenGenerator gen(builder.data(), builder.size(), 0, 0);

    auto token = gen.next();
    EXPECT_GE(token.time, 0);
}

TEST(Gfx12TokenStreamEdgeCaseTest, Gfx12SpecificInst)
{
    TokenStreamBuilder builder;

    // INST with 8-bit inst field (20 bits)
    uint64_t inst_data = 0;
    inst_data |= 0b010;
    inst_data |= (0b011 << 3);
    inst_data |= (1ull << 6);
    inst_data |= (25ull << 7);
    inst_data |= (200ull << 12);

    builder.writeBits(inst_data, 20);
    builder.finalize();

    gfx12::TokenGenerator gen(builder.data(), builder.size(), 0, 0);

    auto token = gen.next();
    EXPECT_EQ(token.type, RdnaType::INST);

    gfx12::inst_type inst{.raw = token.contents};
    EXPECT_EQ(inst.inst, 200);
}

//=============================================================================
// Safe/Unsafe Reader Boundary Tests
// The parser uses readOne_unsafe() when 64+ bits of buffer remain (bufferPadded).
// When near buffer end, it uses readOne_safe() which returns 0 for OOB reads.
// Minimum buffer size is 8 bytes (64 bits) for the read window.
//=============================================================================

TEST(TokenStreamBoundaryTest, SmallTokenInMinBuffer)
{
    // A 12-bit VALU_INST in a small buffer (no padding needed)
    TokenStreamBuilder builder;

    uint64_t valu = 0;
    valu |= 0b011;
    valu |= (0b011 << 3);
    valu |= (0 << 6);
    valu |= (7 << 7);
    builder.writeBits(valu, 12);
    builder.finalize();

    EXPECT_EQ(builder.size(), 2u); // 12 bits -> 2 bytes

    gfx10::TokenGenerator gen(builder.data(), builder.size(), 0, 0);

    auto token = gen.next();
    EXPECT_EQ(token.type, RdnaType::VALU_INST);
    valu_inst_type parsed{.raw = token.contents};
    EXPECT_EQ(parsed.wid, 7);
}

TEST(TokenStreamBoundaryTest, MultipleSmallTokensInMinBuffer)
{
    // Multiple 12-bit tokens in a small buffer (no padding needed)
    TokenStreamBuilder builder;

    // 3 VALU_INST tokens (12 bits each = 36 bits)
    for (int i = 0; i < 3; i++)
    {
        uint64_t valu = 0;
        valu |= 0b011;
        valu |= (i << 3);
        valu |= (0 << 6);
        valu |= ((i + 1) << 7);
        builder.writeBits(valu, 12);
    }
    builder.finalize();

    EXPECT_EQ(builder.size(), 5u); // 36 bits -> 5 bytes

    gfx10::TokenGenerator gen(builder.data(), builder.size(), 0, 0);

    for (int i = 0; i < 3; i++)
    {
        auto token = gen.next();
        EXPECT_EQ(token.type, RdnaType::VALU_INST);
        valu_inst_type parsed{.raw = token.contents};
        EXPECT_EQ(parsed.wid, (uint64_t) (i + 1));
    }
}

TEST(TokenStreamBoundaryTest, Max64BitToken)
{
    // A 64-bit TIMESTAMP fills exactly 8 bytes
    TokenStreamBuilder builder;

    uint64_t ts = 0;
    ts |= 0b0000001; // TIMESTAMP header
    ts |= (0xABCDEF123456ull << 7);
    builder.writeBits(ts, 64);

    EXPECT_GE(builder.size(), 8u);

    gfx10::TokenGenerator gen(builder.data(), builder.size(), 0, 0);

    // TIMESTAMP is consumed internally - verify no crash
    while (gen.nextValid()) gen.next();
}

TEST(TokenStreamBoundaryTest, TokenCrossingSafeUnsafeBoundary)
{
    // >8 bytes buffer where first token uses unsafe, second uses safe reader
    TokenStreamBuilder builder;

    // 64-bit TIMESTAMP (fills initial 8 bytes, uses unsafe reader)
    uint64_t ts = 0;
    ts |= 0b0000001;
    ts |= (0x12345ull << 7);
    builder.writeBits(ts, 64);

    // 12-bit VALU_INST (at bit 64, uses safe reader as <64 bits remain)
    uint64_t valu = 0;
    valu |= 0b011;
    valu |= (0b111 << 3);
    valu |= (1 << 6);
    valu |= (20 << 7);
    builder.writeBits(valu, 12);

    builder.finalize();

    // 76 bits -> 10 bytes (padded to byte boundary)
    EXPECT_EQ(builder.size(), 10u);

    gfx10::TokenGenerator gen(builder.data(), builder.size(), 0, 0);

    // TIMESTAMP consumed internally, VALU_INST returned
    auto token = gen.next();
    EXPECT_EQ(token.type, RdnaType::VALU_INST);
    valu_inst_type parsed{.raw = token.contents};
    EXPECT_EQ(parsed.wid, 20);
}

TEST(TokenStreamBoundaryTest, LargerBufferWithMixedTokens)
{
    // A larger buffer (>16 bytes) with multiple token types
    TokenStreamBuilder builder;

    // WAVE_START (32 bits)
    uint64_t ws = 0;
    ws |= 0b01100;
    ws |= (0b01 << 5);
    ws |= (1ull << 7);
    ws |= (1ull << 8);
    ws |= (3ull << 10);
    ws |= (15ull << 13);
    ws |= (0ull << 18);
    ws |= (1ull << 21);
    ws |= (0ull << 23);
    ws |= (0ull << 24);
    ws |= (5ull << 25);
    builder.writeBits(ws, 32);

    // INST (20 bits)
    uint64_t inst = 0;
    inst |= 0b010;
    inst |= (0ull << 3);
    inst |= (0b011 << 4);
    inst |= (0ull << 7);
    inst |= (15ull << 8);
    inst |= (42ull << 13);
    builder.writeBits(inst, 20);

    // WAVE_END (20 bits)
    uint64_t we = 0;
    we |= 0b10101;
    we |= (0b010 << 5);
    we |= (1ull << 8);
    we |= (1ull << 9);
    we |= (3ull << 11);
    we |= (0ull << 14);
    we |= (15ull << 15);
    builder.writeBits(we, 20);

    builder.finalize();

    gfx10::TokenGenerator gen(builder.data(), builder.size(), 0, 0);

    auto token1 = gen.next();
    EXPECT_EQ(token1.type, RdnaType::WAVE_START);
    gfx10::wstart_type ws_parsed{.raw = token1.contents};
    EXPECT_EQ(ws_parsed.wid, 15);
    EXPECT_EQ(ws_parsed.sa, 1);

    auto token2 = gen.next();
    EXPECT_EQ(token2.type, RdnaType::INST);
    gfx10::inst_type inst_parsed{.raw = token2.contents};
    EXPECT_EQ(inst_parsed.wid, 15);
    EXPECT_EQ(inst_parsed.inst, 42);

    auto token3 = gen.next();
    EXPECT_EQ(token3.type, RdnaType::WAVE_END);
    gfx10::wend_type we_parsed{.raw = token3.contents};
    EXPECT_EQ(we_parsed.wid, 15);
}

TEST(TokenStreamBoundaryTest, GeneratorTerminatesCleanly)
{
    // Verify generator terminates properly when buffer exhausted
    TokenStreamBuilder builder;

    uint64_t valu = 0;
    valu |= 0b011;
    valu |= (0b001 << 3);
    valu |= (0 << 6);
    valu |= (5 << 7);
    builder.writeBits(valu, 12);
    builder.finalize();

    gfx10::TokenGenerator gen(builder.data(), builder.size(), 0, 0);

    auto token1 = gen.next();
    EXPECT_EQ(token1.type, RdnaType::VALU_INST);

    // Consume remaining (should be NOPs from padding)
    int count = 0;
    while (gen.nextValid() && count < 100)
    {
        gen.next();
        count++;
    }
    EXPECT_LT(count, 100); // Should terminate, not infinite loop
}
