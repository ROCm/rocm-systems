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

#pragma once
#include <unordered_map>
#include "gfx12/gfx12parser.h"
#include "mi400token.h"

namespace mi400
{

class TokenLookupTable : public gfx12::TokenLookupTable
{
public:
    TokenLookupTable()
    {
        AddEncoding({
            RdnaType::INST, {0, 1, 0, 0}
        });
        AddEncoding({
            RdnaType::VALU_INST, {1, 1, 0}
        });
        AddEncoding({
            RdnaType::NOP, {0, 0, 0, 0}
        });
        AddEncoding({
            RdnaType::IMM_ONE, {1, 0, 1, 1}
        });
        AddEncoding({
            RdnaType::WAVE_END, {1, 0, 0, 0, 0, 0, 1}
        });
        AddEncoding({
            RdnaType::LDS_CONFIG, {0, 1, 1, 0, 0, 1, 0, 0}
        });
        AddEncoding({
            RdnaType::MISC_GFX10, {1, 0, 0, 0, 1, 0, 1}
        });
        AddEncoding({
            RdnaType::TIME, {0, 1, 1, 1}
        });
        AddEncoding({
            RdnaType::MEDIUM_TIME, {0, 1, 1, 0, 0, 1, 0, 1}
        });
    }

    int64_t getTime(RdnaType type, uint64_t contents, int64_t cur_time, bool& PL, int64_t& rt)
    {
        if (type == RdnaType::TIMESTAMP)
        {
            gfx12::timestamp_type stamp{.raw = contents};
            PL |= bool(stamp.pl && !stamp.rt);
            if (stamp.rt == 0) return stamp.time + cur_time;

            if (stamp.pl == 0) rt = stamp.time;
            return cur_time;
        }
        else if (type == RdnaType::TIME) { cur_time += 1; }
        return getDelta(type, contents) + cur_time;
    };

private:
    int64_t getDelta(RdnaType type, uint64_t contents)
    {
        auto res = time_bits[type];
        uint64_t beg = res.first;
        uint64_t mask = (1ull << (res.second - beg)) - 1;
        return ((contents >> beg) & mask);
    };

    static std::array<std::pair<int, int>, NAVI_TYPE_LAST> time_bits;
};

class TokenGenerator : public NaviTokenGenerator
{
public:
    TokenGenerator(const uint8_t* _buffer, size_t size, int64_t _globaltime, int64_t _base_time);
    gfx10::Token next() override;

    inline uint64_t getBuffer400() { return buffer[byte_ptr]; };

    inline void advanceByte(uint64_t value)
    {
        current = (current >> 8) | (value << 56);
        byte_ptr++;
        bits_toread -= 8;
    }

    inline void readOne_unsafe400()
    {
        while (bits_toread > 0) advanceByte(getBuffer400());
    }

    inline void readOne_safe400()
    {
        while (bits_toread > 0) advanceByte(byte_ptr < BUFFER_SIZE ? getBuffer400() : 0);
    }

    void update_fifo(int wave);
    int get_valu_inst_mi400();

protected:
    std::array<int, 6> FIFO = {-1, -1, -1, -1, -1, -1};

private:
    size_t byte_ptr = 0;
    bool bIsExt = false;
    TokenLookupTable lookupbits{};
    static std::array<uint8_t, 64> TOKEN_LEN;
};

} // namespace mi400
