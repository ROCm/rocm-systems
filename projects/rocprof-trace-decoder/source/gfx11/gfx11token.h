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
#include <cstdint>
#include <iostream>
#include <sstream>
#include <vector>
#include "common.hpp"
#include "gfx10/gfx10parser.h"
#include "gfx10/gfx10token.h"

namespace gfx11
{
union timestamp_type
{
    struct
    {
        uint64_t header   : 7;
        uint64_t _unused1 : 1;
        uint64_t pl       : 1;
        uint64_t rt       : 1;
        uint64_t _unused2 : 2;
        uint64_t time     : 36;
    };
    uint64_t raw;

#ifdef SQTT_LOGGING
    std::stringstream print() const
    {
        std::stringstream ss;
        ss << "time:" << time << " pl:" << pl << " rt:" << rt;
        return ss;
    }
    const char* typestr() const { return "TIMESTAMP"; };
#endif
};

union util_ctr_type
{
    struct
    {
        uint64_t header   : 7;
        uint64_t tm       : 2;
        uint64_t cID      : 2;
        uint64_t spi_busy : 4;
        uint64_t vdata0   : 4;
        uint64_t vdata1   : 4;
        uint64_t sdata0   : 4;
        uint64_t sdata1   : 4;
        uint64_t lds0     : 4;
        uint64_t lds1     : 4;
        uint64_t exp0     : 4;
        uint64_t exp1     : 4;
        uint64_t SA       : 1;
    };
    uint64_t raw;

    static const int ctr_size = 4;

#ifdef SQTT_LOGGING
    std::stringstream print() const { return std::stringstream{}; }
    const char* typestr() const { return "UTILCTR"; };
#endif
};

union shader_data_type
{
    struct
    {
        uint64_t header : 8;
        uint64_t sa     : 1;
        uint64_t simd   : 2;
        uint64_t wgp    : 3;
        uint64_t exec   : 1;
        uint64_t wave   : 5;
        uint64_t data   : 32;
    };
    uint64_t raw;

    shader_data_common_type get()
    {
        shader_data_common_type common{*this};
        common.data = data;
        common.invalid = exec;
        return common;
    }

#ifdef SQTT_LOGGING
    std::stringstream print() const
    {
        std::stringstream ss;
        ss << "wgp:" << wgp << " simd:" << simd << " wave:" << wave << " data:0x" << std::hex << data << std::dec;
        return ss;
    }
    const char* typestr() const { return "SHADER_DATA"; };
#endif
};

union shader_data_short_type
{
    struct
    {
        uint64_t header : 8;
        uint64_t sa     : 1;
        uint64_t simd   : 2;
        uint64_t wgp    : 3;
        uint64_t exec   : 1;
        uint64_t wave   : 5;
        uint64_t data   : 8;
    };
    uint64_t raw;

    shader_data_common_type get()
    {
        shader_data_common_type common{*this};
        common.isshort = 1;
        common.data = data;
        common.invalid = exec;
        return common;
    }

#ifdef SQTT_LOGGING
    std::stringstream print() const
    {
        std::stringstream ss;
        ss << "wgp:" << wgp << " simd:" << simd << " wave:" << wave << " data:0x" << std::hex << data << std::dec;
        return ss;
    }
    const char* typestr() const { return "SHADER_DATA_SHORT"; };
#endif
};

class Token : public gfx10::Token
{
public:
    Token() = default;
    Token(int64_t globaltime, uint64_t _contents, RdnaType _type) : gfx10::Token(globaltime, _contents, _type) {}
};
} // namespace gfx11
