// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "build_standalone.h"

#include <cstring>

#include "gfx9token.h"
#include "rocprof_trace_decoder/trace_decoder_instrument.h"
#include "trace_parser.hpp"

namespace gfx9::build_standalone
{
namespace
{

constexpr uint16_t USERDATA2_ADDR = 0xC342;

inline StatusToken encode_reg_cs(uint8_t me, uint8_t pipe, uint8_t regaddr, uint32_t regdata)
{
    const uint64_t enc_me = (1u - (me & 1u)) & 1u;
    uint64_t v = static_cast<uint64_t>(gfx9::TOKEN_REG_CS) | (static_cast<uint64_t>(pipe & 0x3) << 5) | (enc_me << 7) |
                 (static_cast<uint64_t>(regaddr & 0x7F) << 9) | (static_cast<uint64_t>(regdata) << 16);
    return {v, 6};
}

inline StatusToken encode_reg(uint8_t me, uint8_t pipe, uint16_t regaddr, uint32_t regdata)
{
    const uint64_t enc_me = (1u - (me & 1u)) & 1u;
    uint64_t v = static_cast<uint64_t>(gfx9::TOKEN_REG) | (static_cast<uint64_t>(pipe & 0x3) << 5) | (enc_me << 7) |
                 (uint64_t{1} << 15) | (uint64_t{gfx9::Reg::REG_TYPE_USERDATA} << 10) |
                 (static_cast<uint64_t>(regaddr) << 16) | (static_cast<uint64_t>(regdata) << 32);
    return {v, 8};
}

inline void emit_codeobj_field(std::vector<StatusToken>& out, uint32_t marker_type, uint32_t payload)
{
    rocprof_trace_decoder_packet_header_t hdr{};
    hdr.opcode = ROCPROF_TRACE_DECODER_PACKET_OPCODE_CODEOBJ;
    hdr.type = marker_type;
    hdr.data20 = 0;
    out.push_back(encode_reg(0, 0, USERDATA2_ADDR, hdr.u32All));
    out.push_back(encode_reg(0, 0, USERDATA2_ADDR, payload));
}

} // namespace

std::vector<StatusToken> build_status_tokens(const CSRegisterHandler& reg)
{
    std::vector<StatusToken> out;
    out.reserve(32 + 6 + 1 + reg.active_codeobjs.read().size() * 14);

    for (uint8_t me = 0; me < 2; ++me)
    {
        for (uint8_t pipe = 0; pipe < 4; ++pipe)
        {
            uint64_t pgm = reg.wave_start_addr.at(me).at(pipe);
            if (pgm != 0)
            {
                out.push_back(encode_reg_cs(me, pipe, COMPUTE_PGM_LO, static_cast<uint32_t>(pgm)));
                out.push_back(encode_reg_cs(me, pipe, COMPUTE_PGM_HI, static_cast<uint32_t>(pgm >> 32)));
            }
            uint64_t dpkt = reg.dispatch_pkt_addr.at(me).at(pipe);
            if (dpkt != 0)
            {
                out.push_back(encode_reg_cs(me, pipe, COMPUTE_DISPATCH_PKT_LO, static_cast<uint32_t>(dpkt)));
                out.push_back(encode_reg_cs(me, pipe, COMPUTE_DISPATCH_PKT_HI, static_cast<uint32_t>(dpkt >> 32)));
            }
        }
    }

    out.push_back(encode_reg_cs(0, 0, COMPUTE_NUM_THREAD_X, reg.num_thread_x));
    out.push_back(encode_reg_cs(0, 0, COMPUTE_NUM_THREAD_Y, reg.num_thread_y));
    out.push_back(encode_reg_cs(0, 0, COMPUTE_NUM_THREAD_Z, reg.num_thread_z));
    out.push_back(encode_reg_cs(0, 0, COMPUTE_PGM_RSRC1, reg.rsrc1));
    out.push_back(encode_reg_cs(0, 0, COMPUTE_PGM_RSRC2, reg.rsrc2));
    out.push_back(encode_reg_cs(0, 0, COMPUTE_PGM_RSRC3, reg.rsrc3));

    if (reg.bIsROCMFormat)
    {
        rocprof_trace_decoder_instrument_enable_t enable{};
        enable.char1 = '\0';
        enable.char2 = 'R';
        enable.char3 = 'O';
        enable.char4 = 'C';
        out.push_back(encode_reg(0, 0, USERDATA2_ADDR, enable.u32All));

        for (const auto& co : reg.active_codeobjs.read())
        {
            uint64_t id = co.id;
            uint64_t addr = co.addr;
            uint64_t size = co.size;

            emit_codeobj_field(out, ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ID_LO, static_cast<uint32_t>(id));
            emit_codeobj_field(out, ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ID_HI, static_cast<uint32_t>(id >> 32));
            emit_codeobj_field(out, ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_SIZE_LO, static_cast<uint32_t>(size));
            emit_codeobj_field(
                out, ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_SIZE_HI, static_cast<uint32_t>(size >> 32)
            );
            emit_codeobj_field(out, ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ADDR_LO, static_cast<uint32_t>(addr));
            emit_codeobj_field(
                out, ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ADDR_HI, static_cast<uint32_t>(addr >> 32)
            );

            rocprof_trace_decoder_codeobj_marker_tail_t tail{};
            tail.isUnload = 0;
            tail.bFromStart = 1;
            tail.legacy_id = 0;
            emit_codeobj_field(out, ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_TAIL, tail.raw);
        }
    }

    return out;
}

size_t write_tokens(uint8_t* out, const std::vector<StatusToken>& tokens)
{
    size_t off = 0;
    for (const auto& t : tokens)
    {
        std::memcpy(out + off, &t.bits, t.bytes);
        off += t.bytes;
    }
    return off;
}

} // namespace gfx9::build_standalone
