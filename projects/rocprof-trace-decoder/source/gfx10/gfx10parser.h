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
#include <vector>
#include "gfx10token.h"

struct encoding_t
{
    RdnaType type;
    std::vector<uint8_t> bits;
};

namespace gfx10
{
class TokenLookupTable : public std::array<uint8_t, 256>
{
public:
    TokenLookupTable();
    void AddEncoding(const encoding_t& encoding);

    uint8_t lookup(uint8_t u) const { return data()[u]; };

    int64_t getTime(RdnaType type, uint64_t contents, int64_t cur_time, int64_t& rt)
    {
        if (type == RdnaType::TIMESTAMP)
        {
            timestamp_type stamp{.raw = contents};
            if (stamp.type == 1) return stamp.time + cur_time;

            if (stamp.type == 2) rt = stamp.time;
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
    Token next() override;

private:
    TokenLookupTable lookupbits;
};
} // namespace gfx10
