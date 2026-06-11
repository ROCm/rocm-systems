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

#include "gfx10token.h"
#include "gfx10parser.h"
#include "trace_parser.hpp"

namespace gfx10
{

// GFX10 base token encodings. Each row defines one SQTT token type:
//   {type, pattern, pattern_len, token_bit_length, time_begin, time_end}
//
// gfx11 and gfx12 inherit this table and override specific entries in their constructors.
// To change a token's bit length or time field for a new generation, add an AddEncoding()
// call in the subclass constructor — it overwrites all matching slots in-place.
static constexpr encoding_t bit_encodings[] = {
  // clang-format off
 //                               type  pattern  plen  toklen  time_begin  time_end
 // Target
    {INST,               0b010,    3,  20,  4,  7},
    {VALU_INST,          0b011,    3,  12,  3,  6},
    {IMM_ONE,            0b1101,   4,  12,  4,  7},
    {IMMEDIATE,          0b00100,  5,  24,  5,  8},
    {WAVE_READY,         0b10100,  5,  24,  5,  8},
    {NEW_PC_GFX10,       0b0100001,7,  64,  8, 11},
 // Global
    {WAVE_START,         0b01100,  5,  32,  5,  7},
    {WAVE_START_EXT,     0b11100,  5,  48,  5,  7},
    {WAVE_ALLOC,         0b00101,  5,  20,  5,  8},
    {WAVE_END,           0b10101,  5,  20,  5,  8},
    {SHADER_DATA,        0b00110,  5,  52,  5,  8},
    {SHADER_DATA_SHORT,  0b10110,  5,  28,  5,  8},
    {UTIL_COUNTER_GFX10, 0b0110001,7,  64,  7,  9},
 // Reference
    {TIME,               0b1000,   4,   8,  4,  8},
    {NOP,                0b0000,   4,   4,  0,  0},
    {MISC_GFX10,         0b1010001,7,  24,  7, 16},
    {EVENT,              0b01100001,8, 24,  8, 11},
    {EVENT_SYNC,         0b11100001,8, 32,  8, 11},
    {REG,                0b1001,   4,  64,  4,  7},
    {REG_INIT,           0b1110001,7,  64,  7, 10},
    {TIMESTAMP,          0b0000001,7,  64, 16, 64},
    {HEADER,             0b0010001,7,  64,  0,  0},
  // clang-format on
};

TokenLookupTable::TokenLookupTable() : std::array<token_info_t, 256>({})
{
    for (const auto& encoding : bit_encodings) AddEncoding(encoding);
}

// Fills all 256-entry table slots whose low pattern_len bits equal pattern.
// A 3-bit prefix fills 256/8 = 32 slots, a 7-bit prefix fills 2, an 8-bit prefix fills 1.
// Later calls overwrite earlier ones, so subclass constructors can override inherited entries.
void TokenLookupTable::AddEncoding(const encoding_t& encoding)
{
    int begin = encoding.pattern;
    int stepsize = 1 << encoding.pattern_len;

    token_info_t info{};
    info.type = encoding.type;
    info.length = encoding.length;
    info.time_begin = encoding.time_begin;
    info.time_end = encoding.time_end;

    for (int i = begin; i < 256; i += stepsize) data()[i] = info;
}

TokenGenerator::TokenGenerator(const uint8_t* _buffer, size_t size, int64_t _globaltime, int64_t _base_time) :
NaviTokenGenerator(_buffer, size, _globaltime, _base_time)
{
    if (!_buffer || !size) throw std::exception();
}

Token TokenGenerator::next()
{
    // Unsafe reads when the buffer is sufficiently padded.
    while (bufferPadded() || !lookahead.empty())
    {
        if (!lookahead.empty())
        {
            if (lookahead.front().time < lookahead.back().time || !bufferPadded())
            {
                auto token = lookahead.front();
                lookahead.pop_front();
                return token;
            }
        }

        readOne_unsafe();

        auto& info = lookupbits.lookup(current);
        RdnaType type = (RdnaType) info.type;
        if (type == RdnaType::NOP)
        {
            bits_toread = 4;
            continue;
        }

        bits_toread = info.length;

        int64_t real = 0;
        globaltime = lookupbits.getTime(info, current, globaltime, real);

        if (type == RdnaType::TIMESTAMP || type == RdnaType::TIME)
        {
            if (real != 0)
            {
                rocprofiler_thread_trace_decoder_realtime_t rt{};
                rt.shader_clock = globaltime;
                rt.realtime_clock = real;
                realtime.emplace_back(rt);
            }
            continue;
        }

        auto token = Token{globaltime, current, type};
        if (type == RdnaType::WAVE_READY) token.time -= 1;

        if (!lookahead.empty() && token.time < lookahead.back().time)
            lookahead.emplace_front(token);
        else
            lookahead.emplace_back(token);
    }

    // Safe reads when the buffer is not padded.
    while (bufferValid_unsafe() || !lookahead.empty())
    {
        if (!lookahead.empty())
        {
            if (lookahead.front().time < lookahead.back().time || !bufferValid_unsafe())
            {
                auto token = lookahead.front();
                lookahead.pop_front();
                return token;
            }
        }

        readOne_safe();

        auto& info = lookupbits.lookup(current);
        RdnaType type = (RdnaType) info.type;
        if (type == RdnaType::NOP)
        {
            bits_toread = 4;
            continue;
        }

        bits_toread = info.length;

        int64_t real = 0;
        globaltime = lookupbits.getTime(info, current, globaltime, real);

        if (type == RdnaType::TIMESTAMP || type == RdnaType::TIME)
        {
            if (real != 0) addRealtime(real);
            continue;
        }

        auto token = Token{globaltime, current, type};
        if (type == RdnaType::WAVE_READY) token.time -= 1;

        if (!lookahead.empty() && token.time < lookahead.back().time)
            lookahead.emplace_front(token);
        else
            lookahead.emplace_back(token);
    }

    return Token{0, 0, RdnaType::TIMESTAMP};
}

} // namespace gfx10

void NaviTokenGenerator::addRealtime(uint64_t new_realtime)
{
    if (!realtime.empty())
    {
        auto& rt_back = realtime.back().realtime_clock;
        auto& sh_back = realtime.back().shader_clock;

        if (rt_back >= new_realtime || sh_back >= globaltime)
        {
            // We sometimes see two RTs at the same shader clock, even at a stable profiling power state.
            rt_back = (rt_back + new_realtime) / 2;
            sh_back = (sh_back + globaltime) / 2;
            return;
        }
    }

    rocprofiler_thread_trace_decoder_realtime_t rt{};
    rt.shader_clock = globaltime;
    rt.realtime_clock = new_realtime;
    realtime.emplace_back(rt);
}
