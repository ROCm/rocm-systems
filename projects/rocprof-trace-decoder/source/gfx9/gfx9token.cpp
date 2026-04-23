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

#include "gfx9token.h"
#include <cassert>
#include <stdexcept>

namespace gfx9
{
std::array<int, 16> token_len_dict = {16, 64, 64, 32, 16, 48, 16, 16, 16, 16, 16, 64, 48, 32, 64, 48};

namespace
{

uint64_t unpack16(const uint8_t*& data)
{
    uint64_t value = (uint64_t(data[1]) << 8) | uint64_t(data[0]);
    data += sizeof(uint16_t);
    return value;
}

uint64_t unpack32(const uint8_t*& data)
{
    auto high = unpack16(data);
    auto low = unpack16(data);
    return (low << 16) | high;
}

Token parseOne(const uint8_t*& buffer, size_t& curr_len, size_t buf_size)
{
    size_t rem = buf_size - curr_len;
    if (rem < 8)
    {
        if (rem < size_t(token_len_dict.at(buffer[0] & 15) / 8))
        {
            curr_len = buf_size;
            return Token(TOKEN_TIME, 0);
        }
    }

    uint64_t curr = unpack16(buffer);
    uint64_t type = curr & 15;
    int msg_len = token_len_dict.at(type);
    curr_len += msg_len / 8;

    if (msg_len == 16) return Token(type, curr);
    if (msg_len == 32) return Token(type, (unpack16(buffer) << 16) + curr);
    if (msg_len == 48) return Token(type, (unpack32(buffer) << 16) + curr);
    if (msg_len == 64)
    {
        uint64_t mid = unpack16(buffer);
        uint64_t high = unpack32(buffer);
        return Token(type, (high << 32) + (mid << 16) + curr);
    }
    assert(false && "invalid token size");
    throw std::runtime_error("invalid token size");
}

} // anonymous namespace

Token MITokenGenerator::next()
{
    if (!valid()) throw std::exception();

    while (valid())
    {
        Token token = !lookahead.empty() ? lookahead.at(0) : parseOne(buffer, cur_len, BUFFER_SIZE);
        if (!lookahead.empty()) lookahead.pop_front();

        globaltime += token.delta * 4;
        token.time = globaltime;

        if (token.type == 0 && token.fields.misc.misc_type == 1) // This is time reset
            patch_time();
        else if (token.type != 1)
            return token;
    }

    return Token{1, 0};
}

void MITokenGenerator::patch_time()
{
    auto update = [this](int64_t token_time)
    {
        if (base_time == 0) base_time = token_time - globaltime + 4;
        if (token_time > base_time) globaltime = (token_time - base_time) & ~3ll;
    };

    for (size_t i = 0; i < lookahead.size(); i++)
    {
        auto& token = lookahead.at(i);
        if (token.type == 1)
        {
            update(token.time);
            lookahead.erase(lookahead.begin() + i);
            return;
        }
        else if (token.type == 0 && token.fields.misc.misc_type == 1)
            return;
    }

    while (cur_len < BUFFER_SIZE)
    {
        Token token = parseOne(buffer, cur_len, BUFFER_SIZE);
        if (token.type == 1)
        {
            update(token.time);
            return;
        }
        else if (token.type == 0 && token.fields.misc.misc_type == 1)
            return;
        lookahead.push_back(token);
    }
}

} // namespace gfx9