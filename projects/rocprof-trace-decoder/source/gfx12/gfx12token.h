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
#include "gfx11/gfx11parser.h"
#include "gfx11/gfx11token.h"

namespace gfx12
{
union timestamp_type
{
    struct
    {
        uint64_t header : 7;
        uint64_t rt     : 1;
        uint64_t pl     : 1;
        uint64_t tl     : 1;
        uint64_t unused : 2;
        uint64_t time   : 52;
    };
    uint64_t raw;

#ifdef SQTT_LOGGING
    std::stringstream print() const
    {
        std::stringstream ss;
        ss << "time:" << time << " pl:" << pl << " rt:" << rt << " tl:" << tl;
        return ss;
    }
    const char* typestr() const { return "TIMESTAMP"; };
#endif
};

union inst_type
{
    struct
    {
        uint64_t header : 3;
        uint64_t tm     : 3;
        uint64_t w64h   : 1;
        uint64_t wid    : 5;
        uint64_t inst   : 8;
    };
    uint64_t raw;

    inst_type_common get() const { return inst_type_common{.tm = tm, .w64h = w64h, .wid = wid, .inst = inst}; }

#ifdef SQTT_LOGGING
    std::stringstream print() const
    {
        std::stringstream ss;
        ss << "wid:" << wid << " inst:" << inst << " w64:" << (bool) w64h;
        return ss;
    }
    const char* typestr() const { return "INST"; };
#endif
};

union wend_type
{
    struct
    {
        uint64_t header : 5;
        uint64_t tm     : 3;
        uint64_t sa     : 1;
        uint64_t simd   : 2;
        uint64_t wgp    : 4;
        uint64_t wid    : 5;
    };
    uint64_t raw;

    wend_type_common get() const { return wend_type_common{.tm = tm, .sa = sa, .simd = simd, .wgp = wgp, .wid = wid}; }

#ifdef SQTT_LOGGING
    std::stringstream print() const
    {
        std::stringstream ss;
        ss << "wgp:" << wgp << " simd:" << simd << " wid:" << wid << " sa:" << sa;
        return ss;
    }
    const char* typestr() const { return "WAVE_END"; };
#endif
};

union wstart_type
{
    struct
    {
        uint64_t header     : 4;
        uint64_t isExt      : 1;
        uint64_t tm         : 2;
        uint64_t sa         : 1;
        uint64_t simd       : 2;
        uint64_t wgp        : 4;
        uint64_t reserved   : 1;
        uint64_t wid        : 5;
        uint64_t dispatcher : 5;
        uint64_t count      : 7;
        uint64_t zero       : 1;
        uint64_t wgid       : 5;
        uint64_t last       : 1;
        uint64_t dvg        : 1;
    };
    uint64_t raw;

    wstart_type_common get() const
    {
        return wstart_type_common{
            .tm = tm,
            .sa = sa,
            .simd = simd,
            .wgp = wgp,
            .wid = wid,
            .pipe = dispatcher & 0x3ul,
            .me = (dispatcher >> 2ul) & 1ul,
            .count = count,
            .isExt = isExt,
            .wgid = wgid,
            .last = last,
            .dynvgpr = dvg};
    }

#ifdef SQTT_LOGGING
    std::stringstream print() const
    {
        std::stringstream ss;
        ss << "wgp:" << wgp << " simd:" << simd << " wid:" << wid << " sa:" << sa << " dispatcher:" << dispatcher
           << " count:" << count;
        return ss;
    }
    const char* typestr() const { return "WAVE_START"; };
#endif
};

union new_pc_type
{
    struct
    {
        uint64_t tm   : 3;
        uint64_t wave : 5;
        uint64_t pc   : 55;
        uint64_t err  : 1;
    };
    uint64_t raw;

#ifdef SQTT_LOGGING
    std::stringstream print() const
    {
        std::stringstream ss;
        ss << "w:" << wave << " 0x" << std::hex << pc << " 0x" << (pc << 2) << std::dec;
        return ss;
    }
    const char* typestr() const { return "NEW_PC"; };
#endif
};

union shader_data_type
{
    struct
    {
        uint64_t header : 10;
        uint64_t sa     : 1;
        uint64_t simd   : 2;
        uint64_t wgp    : 4;
        uint64_t wave   : 5;
        uint64_t data   : 32;
        uint64_t sdt    : 2;
    };
    uint64_t raw;

    shader_data_common_type get()
    {
        shader_data_common_type common{*this};
        common.data = data;
        common.priv = sdt != 0;
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
        uint64_t header : 10;
        uint64_t sa     : 1;
        uint64_t simd   : 2;
        uint64_t wgp    : 4;
        uint64_t wave   : 5;
        uint64_t data   : 8;
        uint64_t sdt    : 2;
    };
    uint64_t raw;

    shader_data_common_type get()
    {
        shader_data_common_type common{*this};
        common.isshort = 1;
        common.data = data;
        common.priv = sdt != 0;
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

class Token : public gfx11::Token
{
public:
    Token() = default;
    Token(int64_t globaltime, uint64_t _contents, RdnaType _type) : gfx11::Token(globaltime, _contents, _type) {}
};
} // namespace gfx12
