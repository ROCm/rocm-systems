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

#include <array>
#include <cstring>

#include "mi400parser.h"
#include "mi400token.h"
#include "trace_parser.hpp"

typedef mi400::Token Token;

namespace mi400
{

TokenLookupTable::TokenLookupTable()
{
    // MI400 overrides on top of the inherited gfx12 table.
    // Adds LDS_CONFIG and MEDIUM_TIME, and re-points INST/VALU_INST/IMM_ONE/WAVE_END/MISC_GFX10/TIME/NOP
    // to the bit lengths and time-field positions used by MI4xx (gfx12.5) traces.
    // clang-format off
    //                  type             pattern      plen  toklen  time_begin  time_end
    AddEncoding({INST,         0b0010,      4, 24, 4,  6});
    AddEncoding({VALU_INST,    0b011,       3,  8, 4,  8});
    AddEncoding({NOP,          0b0000,      4,  8, 0,  0});
    AddEncoding({IMM_ONE,      0b1101,      4, 16, 7, 10});
    AddEncoding({WAVE_END,     0b1000001,   7, 24, 7, 10});
    AddEncoding({LDS_CONFIG,   0b00100110,  8, 24, 8, 10});
    AddEncoding({MISC_GFX10,   0b1010001,   7, 24, 7, 10});
    AddEncoding({TIME,         0b1110,      4,  8, 4,  8});
    AddEncoding({MEDIUM_TIME,  0b10100110,  8, 16, 8, 16});
    // clang-format on
}

std::array<int, 16> TM_DELTA_TABLE = {1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 1, 2};

gfx10::Token TokenGenerator::next()
{
    // size of newpc: safe
    while (byte_ptr + 9 < BUFFER_SIZE)
    {
        readOne_unsafe400();

        if (bIsExt && (current & 1)) // Handle wave_start_ext
        {
            bits_toread = 8; // LDS is 48 bits, WG is 40 bits
            continue;
        }

        auto& info = lookupbits.lookup(current);
        RdnaType type = (RdnaType) info.type;
        bits_toread = info.length;
        if (type == RdnaType::NOP || bits_toread == 0)
        {
            bits_toread = 8;
            continue;
        }

        bIsExt = type == WAVE_START_EXT;

        int64_t real = 0;
        if (type == VALU_INST)
            globaltime += TM_DELTA_TABLE[valu_inst_type{.raw = current}.wavetm];
        else
            globaltime = lookupbits.getTime(info, current, globaltime, packetlost, real);

        if (type == RdnaType::TIMESTAMP || type == RdnaType::TIME)
        {
            if (real != 0) addRealtime(real);
            continue;
        }

        auto token = Token{globaltime, current, type};

        if (type == VALU_INST)
        {
            ::valu_inst_type valu12 = {};
            valu12.wid = get_valu_inst_mi400();
            token.contents = valu12.raw;
        }
        else if (type == INST)
        {
            auto inst = gfx12::inst_type{.raw = current};
            if (inst.inst >= 10 && inst.inst <= 14) update_fifo(inst.wid);
        }

        // Read last 8 bits of INST_PC
        if (bits_toread > 64 && byte_ptr < BUFFER_SIZE)
        {
            advanceByte(getBuffer400());
            token.contents = current;
        }

        bit_ptr = byte_ptr * 8;
        return token;
    }

    // Duplicated for performance reasons. Avoiding duplication leads to worse performance.
    while (byte_ptr < BUFFER_SIZE || current)
    {
        readOne_safe400();

        if (bIsExt && (current & 1)) // Handle wave_start_ext
        {
            bits_toread = 8; // LDS is 48 bits, WG is 40 bits
            continue;
        }

        auto& info = lookupbits.lookup(current);
        RdnaType type = (RdnaType) info.type;
        bits_toread = info.length;
        if (type == RdnaType::NOP || bits_toread == 0)
        {
            bits_toread = 8;
            continue;
        }

        bIsExt = type == WAVE_START_EXT;

        int64_t real = 0;
        if (type == VALU_INST)
            globaltime += TM_DELTA_TABLE[valu_inst_type{.raw = current}.wavetm];
        else
            globaltime = lookupbits.getTime(info, current, globaltime, packetlost, real);

        if (type == RdnaType::TIMESTAMP || type == RdnaType::TIME)
        {
            if (real != 0) addRealtime(real);
            continue;
        }

        auto token = Token{globaltime, current, type};

        if (type == VALU_INST)
        {
            ::valu_inst_type valu12 = {};
            valu12.wid = get_valu_inst_mi400();
            token.contents = valu12.raw;
        }
        else if (type == INST)
        {
            auto inst = gfx12::inst_type{.raw = current};
            if (inst.inst >= 10 && inst.inst <= 14) update_fifo(inst.wid);
        }

        // Read last 8 bits of INST_PC
        if (bits_toread > 64 && byte_ptr < BUFFER_SIZE)
        {
            advanceByte(getBuffer400());
            token.contents = current;
        }

        bit_ptr = byte_ptr * 8;
        return token;
    }

    bit_ptr = byte_ptr * 8;
    return Token{0, 0, RdnaType::TIMESTAMP};
}

TokenGenerator::TokenGenerator(const uint8_t* _buffer, size_t size, int64_t _globaltime, int64_t _base_time) :
NaviTokenGenerator(_buffer, size, _globaltime, _base_time)
{
    if (!_buffer || !size) throw std::exception();
}

void TokenGenerator::update_fifo(int wave)
{
    if (FIFO[0] == wave) return;

    int pos = 5;
    for (int i = 1; i < 5; i++)
        if (FIFO[i] == wave) pos = i;

    for (int j = pos; j >= 1; j--) FIFO[j] = FIFO[j - 1];

    FIFO[0] = wave;
}

std::array<int, 16> WAVE_ID_TM_TABLE = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4};

int TokenGenerator::get_valu_inst_mi400()
{
    int wave = FIFO.at(WAVE_ID_TM_TABLE.at(valu_inst_type{.raw = current}.wavetm));
    update_fifo(wave);
    return wave;
}

} // namespace mi400
