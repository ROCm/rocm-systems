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

#include <sys/stat.h>
#include <array>
#include <cassert>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#include "mi400parser.h"
#include "mi400token.h"
#include "trace_parser.hpp"

typedef mi400::Token Token;

namespace mi400
{

std::array<std::pair<int, int>, NAVI_TYPE_LAST> TokenLookupTable::time_bits = []()
{
    std::unordered_map<int, std::pair<int, int>> map{};
    map[UNKNOWN] = {0, 0};
    map[INST] = {4, 6};
    map[VALU_INST] = {4, 8};
    map[WAVE_READY] = {5, 8};
    map[IMMEDIATE] = {5, 8};
    map[IMM_ONE] = {7, 10};
    map[NEW_PC_GFX12] = {8, 11};
    map[EXEC_POPCOUNT1] = {7, 10};
    map[EXEC_POPCOUNT3] = {6, 9};
    map[WAVE_START] = {5, 7};
    map[WAVE_START_EXT] = {5, 7};
    map[WAVE_ALLOC] = {5, 8};
    map[WAVE_END] = {7, 10};
    map[SHADER_DATA] = {7, 10};
    map[SHADER_DATA_SHORT] = {7, 10};
    map[LDS_CONFIG] = {8, 10};
    map[UTIL_COUNTER_GFX11] = {7, 9};
    map[TIME] = {4, 8};
    map[MISC_GFX10] = {7, 10};
    map[EVENT] = {8, 11};
    map[EVENT_SYNC] = {8, 11};
    map[REG] = {4, 7};
    map[REG_INIT] = {7, 10};
    map[TIMESTAMP] = {12, 64};
    map[HEADER] = {0, 0};
    map[MEDIUM_TIME] = {8, 16};
    map[NOP] = {0, 0};

    // Unused
    map[NEW_PC_GFX10] = {0, 0};
    map[UTIL_COUNTER_GFX10] = {0, 0};

    std::array<std::pair<int, int>, NAVI_TYPE_LAST> ret{};

    for (size_t i = 0; i < NAVI_TYPE_LAST; i++)
        if (map.find(i) == map.end()) abort();

    for (size_t i = 0; i < NAVI_TYPE_LAST; i++) ret[i] = map.at(i);

    return ret;
}();

std::array<uint8_t, 64> TokenGenerator::TOKEN_LEN = []()
{
    static_assert(NAVI_TYPE_LAST <= 64);
    std::array<uint8_t, 64> map{};
    for (auto& v : map) v = 8;

    map[UNKNOWN] = 8;
    map[INST] = 24;
    map[VALU_INST] = 8;
    map[IMM_ONE] = 16;
    map[IMMEDIATE] = 24;
    map[WAVE_READY] = 24;
    map[NEW_PC_GFX12] = 72;
    map[EXEC_POPCOUNT1] = 24;
    map[EXEC_POPCOUNT3] = 48;
    map[WAVE_START] = 32;
    map[WAVE_START_EXT] = 40;
    map[WAVE_ALLOC] = 24;
    map[WAVE_END] = 24;
    map[SHADER_DATA] = 56;
    map[SHADER_DATA_SHORT] = 32;
    map[UTIL_COUNTER_GFX11] = 48;
    map[LDS_CONFIG] = 24;
    map[MISC_GFX10] = 24;
    map[EVENT] = 24;
    map[EVENT_SYNC] = 32;
    map[REG] = 64;
    map[REG_INIT] = 64;
    map[TIME] = 8;
    map[MEDIUM_TIME] = 16;
    map[TIMESTAMP] = 64;
    map[HEADER] = 64;
    map[NOP] = 8;

    // Unused
    map[NEW_PC_GFX10] = 8;
    map[UTIL_COUNTER_GFX10] = 8;

    return map;
}();

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

        RdnaType type = (RdnaType) lookupbits.lookup(current);
        if (type == RdnaType::NOP)
        {
            bits_toread = 8;
            continue;
        }

        bits_toread = TOKEN_LEN.at(type);
        bIsExt = type == WAVE_START_EXT;

        int64_t real = 0;
        if (type == VALU_INST)
            globaltime += TM_DELTA_TABLE[valu_inst_type{.raw = current}.wavetm];
        else
            globaltime = lookupbits.getTime(type, current, globaltime, packetlost, real);

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

        RdnaType type = (RdnaType) lookupbits.lookup(current);
        if (type == RdnaType::NOP)
        {
            bits_toread = 8;
            continue;
        }

        bits_toread = TOKEN_LEN.at(type);
        bIsExt = type == WAVE_START_EXT;

        int64_t real = 0;
        if (type == VALU_INST)
            globaltime += TM_DELTA_TABLE[valu_inst_type{.raw = current}.wavetm];
        else
            globaltime = lookupbits.getTime(type, current, globaltime, packetlost, real);

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
