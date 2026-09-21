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

#include "gfx11token.h"
#include "gfx11parser.h"
#include "trace_parser.hpp"

typedef gfx10::Token Token;

namespace gfx11
{

TokenLookupTable::TokenLookupTable()
{
    // GFX11 overrides on top of the inherited gfx10 table.
    // TIMESTAMP shrinks from 64 to 48 bits; UTIL_COUNTER uses the gfx11 layout.
    // clang-format off
    //                  type             pattern  plen  toklen  time_begin  time_end
    AddEncoding({MISC_GFX10,         0b1010001, 7, 24,  7, 16});
    AddEncoding({UTIL_COUNTER_GFX11, 0b0110001, 7, 48,  7,  9});
    AddEncoding({TIMESTAMP,          0b0000001, 7, 48, 12, 48});
    // clang-format on
}

gfx10::Token TokenGenerator::next()
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
        globaltime = lookupbits.getTime(info, current, globaltime, packetlost, real);

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
        globaltime = lookupbits.getTime(info, current, globaltime, packetlost, real);

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

TokenGenerator::TokenGenerator(const uint8_t* _buffer, size_t size, int64_t _globaltime, int64_t _base_time) :
NaviTokenGenerator(_buffer, size, _globaltime, _base_time)
{
    if (!_buffer || !size) throw std::exception();
}

} // namespace gfx11
