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
#include "gfx10token.h"

// Combined token lookup entry. TokenLookupTable is a 256-entry array of these, indexed by the
// low 8 bits of the current token window. A single lookup yields the token type, its bit length,
// and the bit range of the embedded time-delta field — replacing three separate table lookups.
struct token_info_t
{
    uint8_t type;       // RdnaType
    uint8_t length;     // total token bit length
    uint8_t time_begin; // first bit of the time-delta field within the token
    uint8_t time_end;   // one-past-last bit; delta mask = (1 << (time_end - time_begin)) - 1
};

// Describes how to populate TokenLookupTable entries for a given token type.
//
//   pattern     — the fixed low bits that identify this token, packed LSB-first.
//                 e.g. the SQTT encoding {0,1,0} means bit0=0, bit1=1, bit2=0 → 0b010.
//   pattern_len — how many low bits are significant (the prefix length).
//
// AddEncoding() fills every table slot whose low pattern_len bits match pattern,
// so shorter prefixes cover more slots (e.g. 3-bit prefix → 32 slots).
//
// To add a new token type for a new GPU generation:
//   1. Add the RdnaType to token_types.h.
//   2. Call AddEncoding() in the new generation's TokenLookupTable constructor with:
//        {TYPE, pattern, pattern_len, total_bit_length, time_begin, time_end}
//      This will override any inherited entries that share the same bit prefix.
struct encoding_t
{
    uint8_t type;        // RdnaType
    uint8_t pattern;     // bit pattern identifying this token (LSB-first packed)
    uint8_t pattern_len; // number of significant bits in pattern
    uint8_t length;      // total token bit length
    uint8_t time_begin;  // first bit of time-delta field
    uint8_t time_end;    // one-past-last bit of time-delta field
};

namespace gfx10
{
class TokenLookupTable : public std::array<token_info_t, 256>
{
public:
    TokenLookupTable();
    void AddEncoding(const encoding_t& encoding);

    const token_info_t& lookup(uint8_t u) const { return data()[u]; };

    int64_t getTime(const token_info_t& info, uint64_t contents, int64_t cur_time, int64_t& rt)
    {
        if (info.type == RdnaType::TIMESTAMP)
        {
            timestamp_type stamp{.raw = contents};
            if (stamp.type == 1) return stamp.time + cur_time;

            if (stamp.type == 2) rt = stamp.time;
            return cur_time;
        }
        return getDelta(info, contents) + cur_time;
    };

private:
    static int64_t getDelta(const token_info_t& info, uint64_t contents)
    {
        uint64_t mask = (1ull << (info.time_end - info.time_begin)) - 1;
        return ((contents >> info.time_begin) & mask) + 4 * (info.type == RdnaType::TIME);
    };
};

class TokenGenerator : public NaviTokenGenerator
{
public:
    TokenGenerator(const uint8_t* _buffer, size_t size, int64_t _globaltime, int64_t _base_time);
    Token next() final;

private:
    TokenLookupTable lookupbits;
};
} // namespace gfx10
