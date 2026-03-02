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
#include <cassert>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#include "gfx12parser.h"
#include "gfx12token.h"
#include "trace_parser.hpp"

typedef gfx10::Token Token;

namespace gfx12
{

const std::array<std::pair<int, int>, NAVI_TYPE_LAST> TokenLookupTable::time_bits = []()
{
    std::array<std::pair<int, int>, NAVI_TYPE_LAST> ret{};

    ret.at(RdnaType::INST) = {3, 6};
    ret.at(RdnaType::INST) = {3, 6};
    ret.at(RdnaType::INST) = {3, 6};
    ret.at(RdnaType::VALU_INST) = {3, 6};
    ret.at(RdnaType::IMM_ONE) = {4, 7};
    ret.at(RdnaType::IMMEDIATE) = {5, 8};
    ret.at(RdnaType::WAVE_READY) = {5, 8};
    ret.at(RdnaType::NEW_PC_GFX12) = {8, 11};
    ret.at(RdnaType::WAVE_START) = {5, 7};
    ret.at(RdnaType::WAVE_START_EXT) = {5, 7};
    ret.at(RdnaType::WAVE_ALLOC) = {5, 8};
    ret.at(RdnaType::WAVE_END) = {5, 8};
    ret.at(RdnaType::SHADER_DATA) = {7, 10};
    ret.at(RdnaType::SHADER_DATA_SHORT) = {7, 10};
    ret.at(RdnaType::UTIL_COUNTER_GFX10) = {7, 9};
    ret.at(RdnaType::UTIL_COUNTER_GFX11) = {7, 9};
    ret.at(RdnaType::TIME) = {4, 8};
    ret.at(RdnaType::NOP) = {0, 0};
    ret.at(RdnaType::MISC_GFX10) = {7, 16};
    ret.at(RdnaType::EVENT) = {8, 11};
    ret.at(RdnaType::EVENT_SYNC) = {8, 11};
    ret.at(RdnaType::REG) = {4, 7};
    ret.at(RdnaType::REG_INIT) = {7, 10};
    ret.at(RdnaType::TIMESTAMP) = {12, 64};
    ret.at(RdnaType::HEADER) = {0, 0};
    ret.at(RdnaType::EXEC_POPCOUNT1) = {7, 10};
    ret.at(RdnaType::EXEC_POPCOUNT3) = {6, 9};

    return ret;
}();

static const std::array<uint8_t, 32> TOKEN_LEN = {
    /*UNKNOWN*/ 8,
    /*VALU_INST*/ 12,
    /*IMM_ONE*/ 12,
    /*IMMEDIATE*/ 24,
    /*WAVE_READY*/ 24,
    /*NEW_PC*/ 72,
    /*WAVE_END*/ 20,
    /*WAVE_START*/ 32,
    /*WAVE_START_EXT*/ 40,
    /*WAVE_ALLOC*/ 24,
    /*SHADER_DATA*/ 56,
    /*SHADER_DATA_SHORT*/ 32,
    /*UTIL_COUNTER*/ 48,
    /*TIME*/ 8,
    /*NOP*/ 4,
    /*MISC_GFX10*/ 24,
    /*EVENT*/ 24,
    /*EVENT_SYNC*/ 32,
    /*REG*/ 64,
    /*REG_INIT*/ 64,
    /*TIMESTAMP*/ 64,
    /*HEADER*/ 64,
    /*INST*/ 20,
    /*UTIL_COUNTER*/ 48,
    /*EXEC_POPCOUNT1*/ 24,
    /*EXEC_POPCOUNT3*/ 48,
    /*NEW_PC_GFX12*/ 72,
};

gfx10::Token TokenGenerator::next()
{
    while (bufferPadded())
    {
        readOne_unsafe();

        if (bIsExt && (current & 1)) // Handle wave_start_ext
        {
            bits_toread = 8; // PBB/Base are 48 bits, WG is 40 bits
            continue;
        }

        RdnaType type = (RdnaType) lookupbits.lookup(current);
        if (type == RdnaType::NOP)
        {
            bits_toread = 4;
            continue;
        }

        bits_toread = TOKEN_LEN[type & 0x1F];
        bIsExt = type == WAVE_START_EXT;

        int64_t real = 0;
        globaltime = lookupbits.getTime(type, current, globaltime, packetlost, real);

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

    // Duplicated for performance reasons. Avoiding duplication leads to worse performance.
    while (bufferValid_unsafe())
    {
        readOne_safe();

        if (bIsExt && (current & 1)) // Handle wave_start_ext
        {
            bits_toread = 8; // PBB/Base are 48 bits, WG is 40 bits
            continue;
        }

        RdnaType type = (RdnaType) lookupbits.lookup(current);
        if (type == RdnaType::NOP)
        {
            bits_toread = 4;
            continue;
        }

        bits_toread = TOKEN_LEN[type & 0x1F];
        bIsExt = type == WAVE_START_EXT;

        int64_t real = 0;
        globaltime = lookupbits.getTime(type, current, globaltime, packetlost, real);

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

    return Token{0, 0, RdnaType::TIMESTAMP};
}

TokenGenerator::TokenGenerator(const uint8_t* _buffer, size_t size, int64_t _globaltime, int64_t _base_time) :
NaviTokenGenerator(_buffer, size, _globaltime, _base_time)
{
    if (!_buffer || !size) throw std::exception();
}

} // namespace gfx12
