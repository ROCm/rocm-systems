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

#include <deque>
#include <iostream>
#include <sstream>
#include <vector>
#include "common.hpp"
#include "token_types.h"
#include "trace_parser.hpp"

namespace gfx10
{
union wstart_type
{
    struct
    {
        uint64_t header     : 5;
        uint64_t tm         : 2;
        uint64_t sa         : 1;
        uint64_t simd       : 2;
        uint64_t wgp        : 3;
        uint64_t wid        : 5;
        uint64_t queue      : 3;
        uint64_t pipe       : 2;
        uint64_t me         : 1;
        uint64_t dispatcher : 1;
        uint64_t count      : 7;
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
            .pipe = pipe,
            .me = me,
            .count = count,
            .isExt = 0};
    }
};
} // namespace gfx10

struct wend_type_common
{
    uint64_t tm   : 3;
    uint64_t sa   : 1;
    uint64_t simd : 2;
    uint64_t wgp  : 4;
    uint64_t wid  : 5;

    uint64_t SACU() const { return get_sa_wgp(sa, wgp); }
    uint64_t getGPULocation() const { return (SACU() << 7) | (simd << 5) | wid; };

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

namespace gfx10
{
union wend_type
{
    struct
    {
        uint64_t header  : 5;
        uint64_t tm      : 3;
        uint64_t sa      : 1;
        uint64_t simd    : 2;
        uint64_t wgp     : 3;
        uint64_t _unused : 1;
        uint64_t wid     : 5;
    };
    uint64_t raw;

    wend_type_common get() const { return wend_type_common{.tm = tm, .sa = sa, .simd = simd, .wgp = wgp, .wid = wid}; }
};
} // namespace gfx10

union header_type
{
    struct
    {
        uint64_t header    : 7;
        uint64_t version   : 6;
        uint64_t DSIMD     : 2;
        uint64_t DWGP      : 3;
        uint64_t _unused   : 1;
        uint64_t DSA       : 1;
        uint64_t NSA       : 1;
        uint64_t NWGP      : 3;
        uint64_t _unused2  : 1;
        uint64_t WSM       : 2;
        uint64_t UCF       : 1;
        uint64_t DPRate    : 4;
        uint64_t wv20      : 1;
        uint64_t dp_derate : 3;
    };
    uint64_t raw;

#ifdef SQTT_LOGGING
    std::stringstream print() const
    {
        std::stringstream ss;
        ss << "TT Version:" << version << " NWGP:" << NWGP << "DWGP:" << DWGP << " DSIMD:" << DSIMD << " DSA:" << DSA
           << " NSA:" << NSA << " UCF:" << UCF << "DPRate:" << DPRate << " WSM:" << WSM;
        return ss;
    }
    const char* typestr() const { return "HEADER"; };
#endif
};

struct inst_type_common
{
    uint64_t tm   : 4;
    uint64_t w64h : 1;
    uint64_t wid  : 6;
    uint64_t inst : 10;

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

namespace gfx10
{
union inst_type
{
    struct
    {
        uint64_t header  : 3;
        uint64_t _unused : 1;
        uint64_t tm      : 3;
        uint64_t w64h    : 1;
        uint64_t wid     : 5;
        uint64_t inst    : 7;
    };
    uint64_t raw;

    inst_type_common get() const { return inst_type_common{.tm = tm, .w64h = w64h, .wid = wid, .inst = inst}; }
};
} // namespace gfx10

union valu_inst_type
{
    struct
    {
        uint64_t header : 3;
        uint64_t tm     : 3;
        uint64_t w64h   : 1;
        uint64_t wid    : 5;
    };
    uint64_t raw;

#ifdef SQTT_LOGGING
    std::stringstream print() const
    {
        std::stringstream ss;
        ss << "wid:" << wid << " w64:" << (bool) w64h;
        return ss;
    }
    const char* typestr() const { return "VALU"; };
#endif
};

union immed_one_type
{
    struct
    {
        uint64_t header : 4;
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

union immediate_type
{
    struct
    {
        uint64_t header : 5;
        uint64_t tm     : 3;
        uint64_t waves  : 16;
    };
    uint64_t raw;

#ifdef SQTT_LOGGING
    std::stringstream print() const
    {
        std::stringstream ss;
        ss << "mask:0x" << std::hex << waves << std::dec;
        return ss;
    }
    const char* typestr() const { return "IMMED"; };
#endif
};

union wave_ready_type
{
    struct
    {
        uint64_t header : 5;
        uint64_t tm     : 3;
        uint64_t waves  : 16;
    };
    uint64_t raw;

#ifdef SQTT_LOGGING
    std::stringstream print() const
    {
        std::stringstream ss;
        ss << "mask:0x" << std::hex << waves << std::dec;
        return ss;
    }
    const char* typestr() const { return "WAVERDY"; };
#endif
};

namespace gfx10
{
union misc_type
{
    struct
    {
        uint64_t header           : 7;
        uint64_t tm               : 9;
        uint64_t spm_or_pl        : 1; // PL on gfx10, SPM on gfx11+
        uint64_t gc_rinse         : 1;
        uint64_t reserved         : 1;
        uint64_t save_context     : 1;
        uint64_t tt_stall_start   : 1;
        uint64_t tt_stall_end     : 1;
        uint64_t DIDT_stall_start : 1;
        uint64_t DIDT_stall_end   : 1;
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

union timestamp_type
{
    struct
    {
        uint64_t header  : 7;
        uint64_t _unused : 7;
        uint64_t type    : 2;
        uint64_t time    : 48;
    };
    uint64_t raw;

#ifdef SQTT_LOGGING
    std::stringstream print() const
    {
        std::stringstream ss;
        ss << "type:" << type << " time:" << time;
        return ss;
    }
    const char* typestr() const { return "TIMESTAMP"; };
#endif
};

union util_ctr_type
{
    struct
    {
        uint64_t header           : 7;
        uint64_t tm               : 2;
        uint64_t spi_busy_or_lds1 : 8;
        uint64_t vdata0           : 8;
        uint64_t vdata1           : 8;
        uint64_t sdata0           : 8;
        uint64_t sdata1           : 8;
        uint64_t lds0             : 8;
        uint64_t cID              : 6;
        uint64_t SA               : 1;
    };
    uint64_t raw;

    static const int ctr_size = 8;

#ifdef SQTT_LOGGING
    std::stringstream print() const { return std::stringstream{}; };
    const char* typestr() const { return "UTILCTR"; };
#endif
};

union new_pc_type
{
    struct
    {
        uint64_t header : 8;
        uint64_t tm     : 3;
        uint64_t wave   : 5;
        uint64_t pc     : 46;
        uint64_t err    : 1;
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
} // namespace gfx10

union reg_write_type
{
    struct
    {
        uint64_t header  : 4;
        uint64_t tm      : 3;
        uint64_t pipe    : 2;
        uint64_t me      : 2;
        uint64_t RDP     : 1;
        uint64_t context : 3;
        uint64_t CS      : 1;
        uint64_t regaddr : 16;
        uint64_t regdata : 32;
    };
    uint64_t raw;

#ifdef SQTT_LOGGING
    std::stringstream print() const
    {
        std::stringstream ss;
        ss << "pipe:" << pipe << std::hex << " addr:0x" << regaddr << " data:0x" << regdata << std::dec;
        return ss;
    }
    const char* typestr() const { return "REG_WRITE"; };
#endif
};

union reg_init_type
{
    struct
    {
        uint64_t header  : 7;
        uint64_t tm      : 3;
        uint64_t vmid    : 4;
        uint64_t pipe    : 2;
        uint64_t me      : 2;
        uint64_t type    : 2;
        uint64_t data    : 24;
        uint64_t context : 3;
        uint64_t data2   : 5;
        uint64_t sync_id : 10;
        uint64_t rsvd    : 2;
    };
    uint64_t raw;

#ifdef SQTT_LOGGING
    std::stringstream print() const
    {
        std::stringstream ss;
        ss << "pipe:" << pipe << " type:" << type << " context:" << context << std::hex << " data:0x" << data
           << " data2:0x" << data2 << std::dec;
        return ss;
    }
    const char* typestr() const { return "REG_INIT"; };
#endif
};

namespace gfx10
{
class Token
{
public:
    Token() = default;
    Token(int64_t globaltime, uint64_t _contents, RdnaType _type) : time(globaltime), contents(_contents), type(_type)
    {}
    int64_t time;
    uint64_t contents;
    RdnaType type;
};
} // namespace gfx10

struct shader_data_common_type
{
    uint64_t data    : 32;
    uint64_t cu      : 8;
    uint64_t simd    : 3;
    uint64_t wave    : 6;
    uint64_t isshort : 1;
    uint64_t priv    : 1;
    uint64_t invalid : 1;

    shader_data_common_type() = default;

    template <typename T> shader_data_common_type(const T& other)
    {
        simd = other.simd;
        cu = other.wgp + (other.sa << ROCPROFILER_TRACE_DECODER_CU_SA_SHIFT);
        wave = other.wave;
        data = isshort = priv = invalid = 0;
    };

#ifdef SQTT_LOGGING
    std::stringstream print() const
    {
        std::stringstream ss;
        ss << "cu:" << cu << " simd:" << simd << " wave:" << wave << " data:0x" << std::hex << data << std::dec;
        return ss;
    }
    const char* typestr() const { return "SHADER_DATA"; };
#endif
};

class NaviTokenGenerator : public TokenGenerator
{
public:
    NaviTokenGenerator(const uint8_t* _buffer, size_t size, int64_t _globaltime, int64_t _base_time) :
    TokenGenerator(_buffer, size, _globaltime, _base_time)
    {}

    virtual gfx10::Token next() = 0;
    bool bufferPadded() { return bit_ptr + 64 < 8 * BUFFER_SIZE; }
    bool nextValid() { return bufferPadded() || !lookahead.empty() || current; }
    bool bufferValid_unsafe() { return bit_ptr < 8 * BUFFER_SIZE || current; }

    std::vector<att_decoder_realtime_t> realtime{};

    bool packetlost = false;

protected:
    inline uint64_t getBuffer() { return buffer[bit_ptr / 8]; };

    inline void advance4Bit(uint64_t value)
    {
        uint64_t bmask = value >> (bit_ptr & 0x4);
        current = (current >> 4) | ((bmask << 60) & ~0xFull);
        bit_ptr += 4;
        bits_toread -= 4;
    }

    inline void readOne_unsafe()
    {
        if ((bit_ptr & 0x7) == 0)
            while (bits_toread >= 8)
            {
                current = (current >> 8) | ((getBuffer() << 56) & ~0xFFull);
                bit_ptr += 8;
                bits_toread -= 8;
            }

        while (bits_toread > 0) advance4Bit(getBuffer());
    }

    inline void readOne_safe()
    {
        while (bits_toread > 0) advance4Bit(bit_ptr < 8 * BUFFER_SIZE ? getBuffer() : 0);
    }

    void addRealtime(uint64_t rt_ts);

    std::deque<gfx10::Token> lookahead{};

    int64_t globaltime = 0;
    uint64_t current = 0;
    int bits_toread = 64;
    size_t bit_ptr = 0;
};
