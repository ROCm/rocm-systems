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
#include "gfx12/gfx12parser.h"
#include "gfx12/gfx12token.h"
#include "rocprof_trace_decoder/cxx/common.hpp"

namespace mi400
{

union inst_type
{
    struct
    {
        uint64_t header : 3;
        uint64_t tm     : 3;
        uint64_t tp     : 1;
        uint64_t wid    : 5;
        uint64_t inst   : 9;
    };
    uint64_t raw;

    inst_type_common get() const { return inst_type_common{.tm = tm, .w64h = 0, .wid = wid, .inst = inst}; }
};

union wend_type
{
    struct
    {
        uint64_t header : 7;
        uint64_t tm     : 3;
        uint64_t sa     : 1;
        uint64_t simd   : 2;
        uint64_t wgp    : 4;
        uint64_t wid    : 5;
    };
    uint64_t raw;

    wend_type_common get() const { return wend_type_common{.tm = tm, .sa = sa, .simd = simd, .wgp = wgp, .wid = wid}; }
};

union valu_inst_type
{
    struct
    {
        uint64_t header : 3;
        uint64_t tp     : 1;
        uint64_t wavetm : 4;
    };
    uint64_t raw;
};

union immed_one_type
{
    struct
    {
        uint64_t header : 7;
        uint64_t tm     : 3;
        uint64_t wid    : 5;
    };
    uint64_t raw;

#ifdef SQTT_LOGGING
    std::stringstream print() const
    {
        std::stringstream ss;
        ss << "wid:" << wid;
        return ss;
    }
    const char* typestr() const { return "IMMEDONE"; };
#endif
};

union lds_config_type
{
    struct
    {
        uint64_t header : 8;
        uint64_t tm     : 2;
        uint64_t wgp0   : 4;
        uint64_t split0 : 3;
        uint64_t wgp1   : 4;
        uint64_t split1 : 3;
    };
    uint64_t raw;
};

union misc_type
{
    struct
    {
        uint64_t header           : 7;
        uint64_t tm               : 3;
        uint64_t spm              : 1;
        uint64_t gc_rinse         : 1;
        uint64_t save_context     : 1;
        uint64_t tt_stall_start   : 1;
        uint64_t tt_stall_end     : 1;
        uint64_t DIDT_stall_start : 1;
        uint64_t DIDT_stall_end   : 1;
        uint64_t CLF              : 1;
        uint64_t CLL              : 1;
        uint64_t CLID             : 4;
    };
    uint64_t raw;

#ifdef SQTT_LOGGING
    std::stringstream print() const
    {
        std::stringstream ss;
        ss << "raw:0x" << std::hex << raw << std::dec;
        return ss;
    }
    const char* typestr() const { return "MISC"; };
#endif
};

union header_type
{
    struct
    {
        uint64_t header   : 7;
        uint64_t version  : 6;
        uint64_t DSIMD    : 2;
        uint64_t DWGP     : 4;
        uint64_t DSA      : 1;
        uint64_t NSA      : 1;
        uint64_t _unused2 : 4;
        uint64_t WSM      : 2;
        uint64_t UCF      : 1;
        uint64_t DPRate   : 3;
        uint64_t _unused3 : 1;
        uint64_t wv20     : 1;
        uint64_t _unused4 : 2;
        uint64_t mode     : 2;
        uint64_t tokexc   : 4;
        uint64_t exoth    : 1;
        uint64_t exec     : 1;
        uint64_t dbl      : 1;
        uint64_t hiwat    : 3;
        uint64_t stall    : 1;
        uint64_t spistl   : 1;
        uint64_t sqstl    : 1;
        uint64_t cntm     : 1;
        uint64_t cntd     : 1;
        uint64_t drev     : 1;
        uint64_t afpd     : 1;
        uint64_t afm      : 1;
        uint64_t vgprs    : 2;
        uint64_t wsdet    : 1;
        uint64_t aldet    : 1;
        uint64_t exbarw   : 1;
        uint64_t trans2   : 1;
    };
    uint64_t raw;

#ifdef SQTT_LOGGING
    std::stringstream print() const
    {
        return std::stringstream{} << "TT Version:" << version << "DWGP:" << DWGP << " DSIMD:" << DSIMD
                                   << " DSA:" << DSA << " NSA:" << NSA << " UCF:" << UCF << "DPRate:" << DPRate
                                   << " WSM:" << WSM;
    }
    const char* typestr() const { return "HEADER"; };
#endif
};

class Token : public gfx12::Token
{
public:
    Token() = default;
    Token(int64_t globaltime, uint64_t _contents, RdnaType _type) : gfx12::Token(globaltime, _contents, _type) {}
};

} // namespace mi400
