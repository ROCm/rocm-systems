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
#include <cstring>
#include <vector>

#include "rocprof_trace_decoder/rocprof_trace_decoder.h"
#include "rocprof_trace_decoder/trace_decoder_instrument.h"

#include "gfx9/quick_scan.h"
#include "trace_parser.hpp" // sqtt_token_reg_t (COMPUTE_DISPATCH_INITIATOR / COMPUTE_PGM_*)

namespace
{

// gfx9 sqtt_token_type_t nibbles for the rare cluster captured by the scanner.
constexpr uint32_t TOKEN_REG         = 2;
constexpr uint32_t TOKEN_REG_CS      = 5;
constexpr uint32_t TOKEN_REG_CS_PRIV = 15;

// Bit layout of TOKEN_REG_CS / TOKEN_REG_CS_PRIV (see source/gfx9/gfx9token.h
// `RegCs`): pipe at bits 5-6, me at bits 7-8 (post-fixup +1 mod 2),
// regaddr at bits 9-15, regdata at bits 16-47.
inline rocprofiler_thread_trace_decoder_event_type_t map_regcs_to_event(uint16_t regaddr)
{
    switch (regaddr)
    {
        case COMPUTE_DISPATCH_INITIATOR: return ROCPROF_TRACE_DECODER_EVENT_DISPATCH_BEGIN;
        case COMPUTE_PGM_LO: return ROCPROF_TRACE_DECODER_EVENT_REG_PGM_LO;
        case COMPUTE_PGM_HI: return ROCPROF_TRACE_DECODER_EVENT_REG_PGM_HI;
        case COMPUTE_PGM_RSRC1: return ROCPROF_TRACE_DECODER_EVENT_REG_PGM_RSRC1;
        case COMPUTE_PGM_RSRC2: return ROCPROF_TRACE_DECODER_EVENT_REG_PGM_RSRC2;
        case COMPUTE_PGM_RSRC3: return ROCPROF_TRACE_DECODER_EVENT_REG_PGM_RSRC3;
        case COMPUTE_NOWHERE: return ROCPROF_TRACE_DECODER_EVENT_REG_NOWHERE;
        default: return ROCPROF_TRACE_DECODER_EVENT_NONE;
    }
}

inline bool is_gfx9_header(const rocprof_trace_decoder_gfx9_header_t& h)
{
    return (h.legacy_version == 0 || h.legacy_version == 0x11) && (h.gfx9_version2 >= 4 && h.gfx9_version2 <= 6);
}

rocprofiler_thread_trace_decoder_status_t quick_scan_gfx9(
    const uint8_t* tokens,
    uint64_t tokens_size,
    rocprof_trace_decoder_trace_callback_t trace_callback,
    void* userdata
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

    std::vector<rocprof_trace_decoder_event_t> events;
    events.reserve(n);

    for (size_t i = 0; i < n; ++i)
    {
        const auto& tok = raw[i];
        if (tok.type != TOKEN_REG_CS && tok.type != TOKEN_REG_CS_PRIV) continue;

        const uint64_t val = tok.contents;
        const uint8_t pipe = static_cast<uint8_t>((val >> 5) & 0x3);
        const uint8_t me = static_cast<uint8_t>((val >> 7) & 0x3);
        const uint16_t regaddr = static_cast<uint16_t>((val >> 9) & 0x7F);
        const uint32_t regdata = static_cast<uint32_t>((val >> 16) & 0xFFFFFFFFu);

        auto type = map_regcs_to_event(regaddr);
        if (type == ROCPROF_TRACE_DECODER_EVENT_NONE) continue;

        rocprof_trace_decoder_event_t ev{};
        ev.size = sizeof(ev);
        ev.time = 0; // Timestamp not available from quick scan.
        ev.type = type;
        ev.me_id = me;
        ev.pipe_id = pipe;
        ev.reserved = 0;
        ev.payload = regdata;
        events.push_back(ev);
    }

    if (events.empty()) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;

    return trace_callback(
        ROCPROFILER_THREAD_TRACE_DECODER_RECORD_EVENT, events.data(), events.size(), userdata
    );
}

} // namespace

extern "C" rocprofiler_thread_trace_decoder_status_t rocprof_trace_decoder_quick_scan(
    rocprof_trace_decoder_gfx9_header_t header,
    const void* data,
    uint64_t data_size,
    rocprof_trace_decoder_trace_callback_t trace_callback,
    void* userdata
)
{
    if (!data || data_size == 0 || !trace_callback)
        return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

    const uint8_t* buf = static_cast<const uint8_t*>(data);

    if (is_gfx9_header(header))
        return quick_scan_gfx9(buf, data_size, trace_callback, userdata);

    // gfx10+ quick-scan dispatch is not yet wired into the public API. The
    // gfx12/mi400 scanners exist (source/gfx12/quick_scan.h, source/mi400/
    // quick_scan.h) but their event-type mapping has not been validated.
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_NOT_IMPLEMENTED;
}
