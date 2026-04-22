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
#include "gfx11token.h"

namespace gfx11
{
class TokenLookupTable : public gfx10::TokenLookupTable
{
public:
    TokenLookupTable()
    {
        AddEncoding({
            RdnaType::MISC_GFX10, {1, 0, 0, 0, 1, 0, 1}
        });
        AddEncoding({
            RdnaType::UTIL_COUNTER_GFX11, {1, 0, 0, 0, 1, 1, 0}
        });
    }

    int64_t getTime(RdnaType type, uint64_t contents, int64_t cur_time, bool& PL, int64_t& rt)
    {
        if (type == RdnaType::TIMESTAMP)
        {
            timestamp_type stamp{.raw = contents};
            PL |= stamp.pl == 1 && stamp.rt == 0;
            if (stamp.rt == 0) return stamp.time + cur_time;

            if (stamp.pl == 0) rt = stamp.time;
            return cur_time;
        }
        return getDelta(type, contents) + cur_time;
    };

private:
    int64_t getDelta(RdnaType type, uint64_t contents)
    {
        auto res = time_bits[type];
        uint64_t beg = res.first;
        uint64_t mask = (1ull << (res.second - beg)) - 1;
        return ((contents >> beg) & mask) + 4 * (type == RdnaType::TIME);
    };

    static const std::array<std::pair<int, int>, NAVI_TYPE_LAST> time_bits;
};

class TokenGenerator : public NaviTokenGenerator
{
public:
    TokenGenerator(const uint8_t* _buffer, size_t size, int64_t _globaltime, int64_t _base_time);
    gfx10::Token next() override;

private:
    TokenLookupTable lookupbits{};
};
} // namespace gfx11
