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

#include "gfx12token.h"
#include "gfx12parser.h"
#include "trace_parser.hpp"

typedef gfx10::Token Token;

namespace gfx12
{

TokenLookupTable::TokenLookupTable()
{
    // GFX12 overrides on top of the inherited gfx11 table.
    // Adds new token types (EXEC_POPCOUNT*, NEW_PC_GFX12) and changes bit lengths for several
    // existing types (INST time field shifts, SHADER_DATA grows, WAVE_START_EXT shrinks, etc.).
    // clang-format off
    //                  type             pattern     plen  toklen  time_begin  time_end
    AddEncoding({INST,              0b010,      3, 20,  3,  6});
    AddEncoding({SHADER_DATA,       0b0000110,  7, 56,  7, 10});
    AddEncoding({SHADER_DATA_SHORT, 0b1000110,  7, 32,  7, 10});
    AddEncoding({EXEC_POPCOUNT1,    0b1100110,  7, 24,  7, 10});
    AddEncoding({EXEC_POPCOUNT3,    0b010110,   6, 48,  6,  9});
    AddEncoding({NEW_PC_GFX12,      0b0100001,  7, 72,  8, 11});
    AddEncoding({WAVE_START_EXT,    0b11100,    5, 40,  5,  7});
    AddEncoding({WAVE_ALLOC,        0b00101,    5, 24,  5,  8});
    AddEncoding({TIMESTAMP,         0b0000001,  7, 64, 12, 64});
    // clang-format on
}

gfx10::Token TokenGenerator::next()
{
    // Unsafe reads when the buffer is sufficiently padded.
    while (bufferPadded())
    {
        readOne_unsafe();

        if (bIsExt && (current & 1)) // Handle wave_start_ext
        {
            bits_toread = 8; // WG is 40 bits, but VGPR and LDS ext are 48 bits
            continue;
        }

        auto& info = lookupbits.lookup(current);
        RdnaType type = (RdnaType) info.type;
        if (type == RdnaType::NOP)
        {
            bits_toread = 4;
            continue;
        }

        bits_toread = info.length;
        bIsExt = type == WAVE_START_EXT;

        int64_t real = 0;
        globaltime = lookupbits.getTime(info, current, globaltime, packetlost, real);

        if (type == RdnaType::TIMESTAMP || type == RdnaType::TIME)
        {
            if (real != 0) addRealtime(real);
            continue;
        }

        auto token = Token{globaltime, current, type};

        // Read last 8 bits of INST_PC
        if (bits_toread == 72 && bufferPadded())
        {
            advance4Bit(getBuffer());
            advance4Bit(getBuffer());
            token.contents = current;
        }

        return token;
    }

    // Safe reads when the buffer is not padded. TODO: Avoid duplication.
    while (bufferValid_unsafe())
    {
        readOne_safe();

        if (bIsExt && (current & 1)) // Handle wave_start_ext
        {
            bits_toread = 8; // WG is 40 bits, but VGPR and LDS ext are 48 bits
            continue;
        }

        auto& info = lookupbits.lookup(current);
        RdnaType type = (RdnaType) info.type;
        bits_toread = info.length;
        if (bit_ptr + bits_toread > 8 * BUFFER_SIZE + 64) break;

        if (type == RdnaType::NOP)
        {
            bits_toread = 4;
            continue;
        }

        bIsExt = type == WAVE_START_EXT;

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

        // Read last 8 bits of INST_PC
        if (bits_toread == 72 && bufferPadded())
        {
            advance4Bit(getBuffer());
            advance4Bit(getBuffer());
            token.contents = current;
        }

        return token;
    }

    current = 0;
    bit_ptr = 8 * BUFFER_SIZE;
    return Token{0, 0, RdnaType::TIMESTAMP};
}

TokenGenerator::TokenGenerator(const uint8_t* _buffer, size_t size, int64_t _globaltime, int64_t _base_time) :
NaviTokenGenerator(_buffer, size, _globaltime, _base_time)
{
    if (!_buffer || !size) throw std::exception();
}

} // namespace gfx12
