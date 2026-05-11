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

#include <cstddef>
#include <cstdint>
#include <vector>

#include "rocprof_trace_decoder/rocprof_trace_decoder.h"
#include "rocprof_trace_decoder/trace_decoder_instrument.h"

#include "gfx9/quick_scan.h"
#include "trace_parser.hpp" // CSRegisterHandler, sqtt_token_reg_t, sqtt_event_type_t

namespace
{

// gfx9 sqtt_token_type_t nibbles for the rare cluster captured by the scanner.
constexpr uint32_t TOKEN_REG = 2;
constexpr uint32_t TOKEN_REG_CS = 5;
constexpr uint32_t TOKEN_REG_CS_PRIV = 15;

// Duck-typed payload matching the interface CSRegisterHandler::UpdateRegCS /
// UpdateRegNoCS expect (.me, .pipe, .regaddr, .regdata, .disable). Lets us
// reuse the same register-tracking logic as gfx9wave.cpp without pulling in
// the full gfx9 token parser.
struct QuickReg
{
    int8_t me;
    int8_t pipe;
    uint32_t regaddr;
    uint32_t regdata;
    int8_t disable;
};

// Decode a TOKEN_REG_CS / TOKEN_REG_CS_PRIV (see gfx9::RegCs in
// source/gfx9/gfx9token.h:85): pipe at bits 5-6, me at bits 7-8 (post-fixup
// +1 mod 2), regaddr at bits 9-15, regdata at bits 16-47.
inline QuickReg decode_regcs(uint64_t val)
{
    QuickReg r{};
    r.pipe = static_cast<int8_t>((val >> 5) & 0x3);
    r.me = static_cast<int8_t>(((val >> 7) & 0x3) + 1) & 0x1;
    r.regaddr = static_cast<uint32_t>((val >> 9) & 0x7F);
    r.regdata = static_cast<uint32_t>((val >> 16) & 0xFFFFFFFFu);
    r.disable = 0;
    return r;
}

// Decode a TOKEN_REG (see gfx9::Reg in source/gfx9/gfx9token.h:73): pipe at
// bits 5-6, me at bits 7-8 (post-fixup +1 mod 2), regaddr at bits 16-31,
// regdata at bits 32-63, disable = !bit15.
inline QuickReg decode_reg(uint64_t val)
{
    QuickReg r{};
    r.pipe = static_cast<int8_t>((val >> 5) & 0x3);
    r.me = static_cast<int8_t>(((val >> 7) & 0x3) + 1) & 0x1;
    r.regaddr = static_cast<uint32_t>((val >> 16) & 0xFFFF);
    r.regdata = static_cast<uint32_t>((val >> 32) & 0xFFFFFFFFu);
    r.disable = static_cast<int8_t>(!((val >> 15) & 1));
    return r;
}

inline bool is_gfx9_header(const rocprof_trace_decoder_gfx9_header_t& h)
{
    return (h.legacy_version == 0 || h.legacy_version == 0x11) && (h.gfx9_version2 >= 4 && h.gfx9_version2 <= 6);
}

rocprofiler_thread_trace_decoder_status_t quick_scan_gfx9(
    const uint8_t* tokens, uint64_t tokens_size, rocprof_trace_decoder_trace_callback_t trace_callback, void* userdata
)
{
    // Cap matches the 32k generously-sized hint in the mi400 scanner — most
    // traces produce ~50 rare tokens per ~5 MB; oversizing avoids a second
    // pass and the scanner silently drops overflow.
    constexpr size_t kCap = 1u << 15;
    std::vector<gfx9::quick_scan::QuickToken> raw;
    raw.resize(kCap);
    size_t n = gfx9::quick_scan::scan_gfx9(tokens, tokens_size, raw.data(), raw.size());

    if (n == 0) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;

    // Mirror the register-tracking path in gfx9wave.cpp: feed each REG /
    // REG_CS / REG_CS_PRIV write through CSRegisterHandler so that when we
    // see COMPUTE_DISPATCH_INITIATOR with the launch bit set, we have all
    // the previously-latched dispatch state (entry point, thread dims, lds
    // size, dispatch packet addr) ready to publish.
    CSRegisterHandler csregister;

    for (size_t i = 0; i < n; ++i)
    {
        const auto& tok = raw[i];

        if (tok.type == TOKEN_REG_CS || tok.type == TOKEN_REG_CS_PRIV)
        {
            QuickReg r = decode_regcs(tok.contents);
            csregister.UpdateRegCS(r);

            if (r.regaddr == COMPUTE_DISPATCH_INITIATOR && (r.regdata & 1) != 0)
            {
                // Time is unavailable from quick_scan; PopulateDispatch will
                // record 0, which matches the contract documented in
                // rocprof_trace_decoder.h.
                rocprofiler_thread_trace_decoder_dispatch_t dispatch = csregister.PopulateDispatch(0, r.me, r.pipe);
                auto status = trace_callback(ROCPROFILER_THREAD_TRACE_DECODER_RECORD_DISPATCH, &dispatch, 1, userdata);
                if (status != ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS) return status;
            }
            else if (r.regaddr == COMPUTE_NOWHERE && r.regdata == EVENT_CS_PARTIAL_FLUSH)
            {
                rocprofiler_thread_trace_decoder_event_t ev{};
                ev.size = sizeof(ev);
                ev.time = 0;
                ev.type = ROCPROF_TRACE_DECODER_EVENT_CS_PARTIAL_FLUSH;
                ev.me_id = static_cast<uint8_t>(r.me);
                ev.pipe_id = static_cast<uint8_t>(r.pipe);
                ev.reserved = 0;
                ev.payload = 0;
                auto status = trace_callback(ROCPROFILER_THREAD_TRACE_DECODER_RECORD_EVENT, &ev, 1, userdata);
                if (status != ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS) return status;
            }
        }
        else if (tok.type == TOKEN_REG)
        {
            QuickReg r = decode_reg(tok.contents);
            if (r.disable) continue;
            // Only userdata2 writes update register state we care about for
            // dispatch attribution (codeobj load/unload markers); other REG
            // tokens are ignored, mirroring gfx9wave.cpp's TOKEN_REG branch.
            csregister.UpdateRegNoCS(r);
        }
    }

    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

} // namespace

extern "C" __attribute__((visibility("default"))) rocprofiler_thread_trace_decoder_status_t
rocprof_trace_decoder_quick_scan(
    rocprof_trace_decoder_gfx9_header_t header,
    const void* data,
    uint64_t data_size,
    rocprof_trace_decoder_trace_callback_t trace_callback,
    void* userdata,
    int flags
)
{
    if (!data || data_size == 0 || !trace_callback)
        return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

    const uint8_t* buf = static_cast<const uint8_t*>(data);

    if (is_gfx9_header(header)) return quick_scan_gfx9(buf, data_size, trace_callback, userdata);

    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_NOT_IMPLEMENTED;
}
