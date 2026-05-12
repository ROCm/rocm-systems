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

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "rocprof_trace_decoder/rocprof_trace_decoder.h"
#include "rocprof_trace_decoder/trace_decoder_instrument.h"

#include "gfx9/quick_scan.h"
#include "handle.hpp"
#include "trace_parser.hpp" // CSRegisterHandler, sqtt_token_reg_t, sqtt_event_type_t

#define PUBLIC_API extern "C" __attribute__((visibility("default")))

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

inline bool is_gfx9_header(uint64_t header)
{
    rocprof_trace_decoder_gfx9_header_t h{.raw = header};
    return (h.legacy_version == 0 || h.legacy_version == 0x11) && (h.gfx9_version2 >= 4 && h.gfx9_version2 <= 6);
}

rocprofiler_thread_trace_decoder_status_t quick_scan_gfx9(
    CSRegisterHandler& csregister,
    const std::vector<gfx9::quick_scan::QuickToken>& raw,
    int n,
    rocprof_trace_decoder_trace_callback_t trace_callback,
    void* userdata
)
{
    if (n == 0) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;

    // Mirror the register-tracking path in gfx9wave.cpp: feed each REG /
    // REG_CS / REG_CS_PRIV write through CSRegisterHandler so that when we
    // see COMPUTE_DISPATCH_INITIATOR with the launch bit set, we have all
    // the previously-latched dispatch state (entry point, thread dims, lds
    // size, dispatch packet addr) ready to publish.

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

PUBLIC_API rocprofiler_thread_trace_decoder_status_t rocprof_trace_decoder_quick_scan(
    rocprof_trace_decoder_handle_t handle,
    uint64_t chunk_index,
    const void* data,
    uint64_t data_size,
    rocprof_trace_decoder_trace_callback_t trace_callback,
    void* userdata
)
{
    if (!data || data_size < 8 || !trace_callback)
        return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

    int gfxip = 0;
    auto csregister = std::shared_ptr<CSRegisterHandler>{nullptr};

    const uint8_t* buf = static_cast<const uint8_t*>(data);

    if (chunk_index == 0)
    {
        gfxip = is_gfx9_header(static_cast<const uint64_t*>(data)[0]) ? 9 : 0;

        auto decoder = HandleData::get_write_handle(handle);
        if (!decoder.valid()) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

        csregister = std::make_shared<CSRegisterHandler>();
        decoder->gfxip = gfxip;

        if (gfxip == 9)
        {
            data_size -= 8;
            buf += 8;
        }

        if (data_size == 0)
        {
            decoder->pipestate[chunk_index + 1] = std::move(csregister);
            decoder->cv.notify_all();
            return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
        }
    }
    else
    {
        auto decoder = HandleData::get_read_handle(handle);
        if (!decoder.valid()) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

        // Todo: Ensure it has been written
        gfxip = decoder->gfxip;
    }

    thread_local std::vector<gfx9::quick_scan::QuickToken> raw{1u << 20};

    size_t ntokens = gfx9::quick_scan::scan_gfx9(buf, data_size, raw.data(), raw.size());
    while (ntokens == raw.size())
    {
        raw.resize(raw.size() * 2);
        ntokens = gfx9::quick_scan::scan_gfx9(buf, data_size, raw.data(), raw.size());
    }

    if (csregister == nullptr)
    {
        auto decoder = HandleData::get_read_handle(handle);
        if (!decoder.valid()) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

        bool ready = decoder->cv.wait_for(
            decoder.lk,
            std::chrono::milliseconds(100),
            [&]() { return decoder->pipestate.find(chunk_index) != decoder->pipestate.end(); }
        );
        if (!ready) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES;

        csregister = decoder->pipestate.at(chunk_index);
    }

    if (csregister == nullptr) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES;
    csregister = std::make_shared<CSRegisterHandler>(*csregister); // Make a copy

    rocprofiler_thread_trace_decoder_status_t status;
    if (gfxip == 9)
        status = quick_scan_gfx9(*csregister, raw, ntokens, trace_callback, userdata);
    else
        return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_NOT_IMPLEMENTED;

    auto decoder = HandleData::get_write_handle(handle);
    if (!decoder.valid()) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

    decoder->pipestate[chunk_index + 1] = std::move(csregister);
    decoder->cv.notify_all();
    return status;
}

PUBLIC_API rocprofiler_thread_trace_decoder_status_t
rocprof_trace_decoder_flush_chunk(rocprof_trace_decoder_handle_t handle, uint64_t chunk_index)
{
    auto hd = HandleData::get_write_handle(handle);
    if (!hd.valid()) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

    if (hd->pipestate.erase(chunk_index + 1) == 0)
        return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

PUBLIC_API rocprofiler_thread_trace_decoder_status_t rocprof_trace_decoder_build_standalone(
    rocprof_trace_decoder_handle_t handle,
    uint64_t chunk_index,
    const void* data,
    uint64_t data_size,
    uint64_t offset_begin,
    uint64_t offset_end,
    void* data_out,
    uint64_t* size_out
)
{
    if (!data || data_size < 8 || !data_out || offset_begin >= offset_end || offset_end < data_size)
        return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

    /*int gfxip = 0;
    auto csregister = std::shared_ptr<CSRegisterHandler>{nullptr};

    {
        auto decoder = HandleData::get_read_handle(handle);
        if (!decoder.valid()) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

        gfxip = decoder->gfxip;

        const auto& pipe = decoder->pipestate;
        if (pipe.find(chunk_index) != pipe.end()) csregister = decoder->pipestate.at(chunk_index);
    }

    if (csregister == nullptr) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

    CSRegisterHandler temp = *csregister;

    if (gfxip != 9) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_NOT_IMPLEMENTED;

    const uint8_t* buf = static_cast<const uint8_t*>(data);

    // Advance temp to that byte
    auto status = quick_scan_gfx9(temp, buf, offset_begin, trace_callback, userdata);

    // Get tokens to reconstruct regs
    auto tokens = BuildStatusTokens(temp);

    if (*size_out < tokens.size()*8 + offset_end - offset_begin)
    {
        *size_out = tokens.size()*8 + 64 + offset_end - offset_begin;
        return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES;
    }

    auto* buf_out = static_cast<uint8_t*>(data_out);
    // Write beginning.
    size_t size_used = WriteTokens(buf_out, tokens);

    std::memcpy(buf_out + size_used, buf + offset_begin, offset_end - offset_begin);
    size_used += offset_end - offset_begin;

    std::memset(buf_out + size_used, 64, 0);
    *size_out = size_used;*/

    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}