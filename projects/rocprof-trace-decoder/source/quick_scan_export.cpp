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
#include <cstring>
#include <memory>
#include <sstream>
#include <vector>

#include "gfx12/gfx12token.h"
#include "gfx9/build_standalone.h"
#include "gfx9/gfx9token.h" // gfx9::Reg, gfx9::RegCs
#include "mi400/mi400token.h"
#include "quick_scan_export.hpp"

//#define GET_TIMING

#ifdef GET_TIMING
#    define TIMING(x)   auto x = std::chrono::steady_clock::now();
#    define DELTA(x, y) " t" #    x << " " << (t##x - t##y).count() / 10000 / 100.0f <<
#else
#    define TIMING(x)
#    define DELTA(x)
#endif

namespace quick_scan
{
bool avx512_available()
{
#if ROCPROF_TRACE_DECODER_QUICK_SCAN_HAS_SIMD
    static const bool ok = []
    {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx512vbmi") && __builtin_cpu_supports("avx512bw") &&
               __builtin_cpu_supports("avx512f");
    }();
    return ok;
#else
    return false;
#endif
}
} // namespace quick_scan

namespace
{
using quick_scan::QuickToken;
using StatusToken = gfx9::build_standalone::StatusToken;

size_t scan_none(const uint8_t* buf, size_t size, QuickToken* out, size_t out_cap) { return 0; };

uint64_t load_header_word(const void* data)
{
    uint64_t header = 0;
    std::memcpy(&header, data, sizeof(header));
    return header;
}

rocprofiler_thread_trace_decoder_status_t process_events_none(
    CSRegisterHandler& csregister,
    const std::vector<QuickToken>& raw,
    size_t n,
    uint64_t header_skip,
    rocprof_trace_decoder_trace_callback_t trace_callback,
    void* userdata
)
{
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
};

// Status-token reconstruction (build_standalone). gfx9 keeps its builder in
// source/gfx9/build_standalone.{h,cpp}; RDNA status context is emitted here
// because gfx12 and gfx1250 share the same REG token shape for the registers
// that CSRegisterHandler tracks.

constexpr uint16_t RDNA_USERDATA2_ADDR = 0xC342;

inline StatusToken encode_rdna_reg(uint8_t me, uint8_t pipe, uint16_t regaddr, uint32_t regdata, bool cs)
{
    gfx10::reg_write_type r{};
    r.header = 0b1001;
    r.tm = 0;
    r.pipe = pipe & 0x3;
    r.me = me & 0x3;
    r.RDP = 0;
    r.context = 0;
    r.CS = cs ? 1 : 0;
    r.regaddr = regaddr;
    r.regdata = regdata;
    return {r.raw, sizeof(r.raw)};
}

inline void emit_rdna_codeobj_field(std::vector<StatusToken>& out, uint32_t marker_type, uint32_t payload)
{
    rocprof_trace_decoder_packet_header_t hdr{};
    hdr.opcode = ROCPROF_TRACE_DECODER_PACKET_OPCODE_CODEOBJ;
    hdr.type = marker_type;
    hdr.data20 = 0;
    out.push_back(encode_rdna_reg(0, 0, RDNA_USERDATA2_ADDR, hdr.u32All, false));
    out.push_back(encode_rdna_reg(0, 0, RDNA_USERDATA2_ADDR, payload, false));
}

std::vector<StatusToken> build_rdna_status_tokens(const CSRegisterHandler& reg)
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
                out.push_back(encode_rdna_reg(me, pipe, COMPUTE_PGM_LO, static_cast<uint32_t>(pgm), true));
                out.push_back(encode_rdna_reg(me, pipe, COMPUTE_PGM_HI, static_cast<uint32_t>(pgm >> 32), true));
            }
            uint64_t dpkt = reg.dispatch_pkt_addr.at(me).at(pipe);
            if (dpkt != 0)
            {
                out.push_back(encode_rdna_reg(me, pipe, COMPUTE_DISPATCH_PKT_LO, static_cast<uint32_t>(dpkt), true));
                out.push_back(
                    encode_rdna_reg(me, pipe, COMPUTE_DISPATCH_PKT_HI, static_cast<uint32_t>(dpkt >> 32), true)
                );
            }
        }
    }

    out.push_back(encode_rdna_reg(0, 0, COMPUTE_NUM_THREAD_X, reg.num_thread_x, true));
    out.push_back(encode_rdna_reg(0, 0, COMPUTE_NUM_THREAD_Y, reg.num_thread_y, true));
    out.push_back(encode_rdna_reg(0, 0, COMPUTE_NUM_THREAD_Z, reg.num_thread_z, true));
    out.push_back(encode_rdna_reg(0, 0, COMPUTE_PGM_RSRC1, reg.rsrc1, true));
    out.push_back(encode_rdna_reg(0, 0, COMPUTE_PGM_RSRC2, reg.rsrc2, true));
    out.push_back(encode_rdna_reg(0, 0, COMPUTE_PGM_RSRC3, reg.rsrc3, true));

    if (reg.bIsROCMFormat)
    {
        rocprof_trace_decoder_instrument_enable_t enable{};
        enable.char1 = '\0';
        enable.char2 = 'R';
        enable.char3 = 'O';
        enable.char4 = 'C';
        out.push_back(encode_rdna_reg(0, 0, RDNA_USERDATA2_ADDR, enable.u32All, false));

        for (const auto& co : reg.active_codeobjs.read())
        {
            uint64_t id = co.id;
            uint64_t addr = co.addr;
            uint64_t size = co.size;

            emit_rdna_codeobj_field(out, ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ID_LO, static_cast<uint32_t>(id));
            emit_rdna_codeobj_field(
                out, ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ID_HI, static_cast<uint32_t>(id >> 32)
            );
            emit_rdna_codeobj_field(
                out, ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_SIZE_LO, static_cast<uint32_t>(size)
            );
            emit_rdna_codeobj_field(
                out, ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_SIZE_HI, static_cast<uint32_t>(size >> 32)
            );
            emit_rdna_codeobj_field(
                out, ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ADDR_LO, static_cast<uint32_t>(addr)
            );
            emit_rdna_codeobj_field(
                out, ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_ADDR_HI, static_cast<uint32_t>(addr >> 32)
            );

            rocprof_trace_decoder_codeobj_marker_tail_t tail{};
            tail.isUnload = 0;
            tail.bFromStart = 1;
            tail.legacy_id = 0;
            emit_rdna_codeobj_field(out, ROCPROF_TRACE_DECODER_CODEOBJ_MARKER_TYPE_TAIL, tail.raw);
        }
    }

    return out;
}

std::vector<StatusToken> build_status_tokens(int gfxip, const CSRegisterHandler& reg)
{
    if (gfxip == 9) return gfx9::build_standalone::build_status_tokens(reg);
    if (gfxip == 12 || gfxip == 1250) return build_rdna_status_tokens(reg);
    return {};
}

size_t write_status_tokens(uint8_t* out, const std::vector<StatusToken>& tokens)
{
    size_t off = 0;
    for (const auto& t : tokens)
    {
        std::memcpy(out + off, &t.bits, t.bytes);
        off += t.bytes;
    }
    return off;
}

inline int extract_gfxip(uint64_t header)
{
    {
        rocprof_trace_decoder_gfx9_header_t h{.raw = header};
        if ((h.legacy_version == 0 || h.legacy_version == 0x11) && (h.gfx9_version2 >= 4 && h.gfx9_version2 <= 6))
            return 9;
    }

    auto hw_header = mi400::header_type{.raw = header};

    if (hw_header.version == 4) return 12;
    if (hw_header.version == 5) return 1250;
    return 0;
}

// Templated on whether to emit dispatch/event records. The state-update path
// (UpdateRegCS / UpdateRegNoCS) runs unconditionally; build_standalone
// instantiates with EmitEvents=false to advance a CSRegisterHandler over the
// chunk prefix without surfacing events to a caller that didn't ask for them.
template <bool EmitEvents> rocprofiler_thread_trace_decoder_status_t process_events_gfx9(
    CSRegisterHandler& csregister,
    const std::vector<quick_scan::QuickToken>& raw,
    size_t n,
    uint64_t header_skip,
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
    //
    // `header_skip` is the byte distance between the caller's `data` and the
    // buffer the scanner saw (8 for gfx9 chunk 0, 0 otherwise). Adding it to
    // tok.offset yields a byte_offset relative to the caller's `data`.

    for (size_t i = 0; i < n; ++i)
    {
        const auto& tok = raw[i];

        if (tok.type == gfx9::TOKEN_REG_CS || tok.type == gfx9::TOKEN_REG_CS_PRIV)
        {
            gfx9::RegCs r{tok.contents};
            csregister.UpdateRegCS(r);

            if constexpr (EmitEvents)
            {
                if (r.regaddr == COMPUTE_DISPATCH_INITIATOR && (r.regdata & 1) != 0)
                {
                    // Time is unavailable from quick_scan; PopulateDispatch will
                    // record 0, which matches the contract documented in
                    // rocprof_trace_decoder.h.
                    rocprofiler_thread_trace_decoder_dispatch_t dispatch = csregister.PopulateDispatch(0, r.me, r.pipe);
                    dispatch.byte_offset = static_cast<uint64_t>(tok.offset) + header_skip;
                    auto status =
                        trace_callback(ROCPROFILER_THREAD_TRACE_DECODER_RECORD_DISPATCH, &dispatch, 1, userdata);
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
                    ev.flags = ROCPROF_TRACE_DECODER_EVENT_FLAGS_NONE;
                    ev.payload.raw = 0;
                    ev.byte_offset = static_cast<uint64_t>(tok.offset) + header_skip;
                    auto status = trace_callback(ROCPROFILER_THREAD_TRACE_DECODER_RECORD_EVENT, &ev, 1, userdata);
                    if (status != ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS) return status;
                }
            }
        }
        else if (tok.type == gfx9::TOKEN_REG)
        {
            gfx9::Reg r{tok.contents};
            if (r.disable) continue;
            // Only userdata2 writes update register state we care about for
            // dispatch attribution (codeobj load/unload markers); other REG
            // tokens are ignored, mirroring gfx9wave.cpp's TOKEN_REG branch.
            csregister.UpdateRegNoCS(r);
        }
    }

    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

// Templated on whether to emit dispatch/event records. The state-update path
// (UpdateRegCS / UpdateRegNoCS) runs unconditionally; build_standalone
// instantiates with EmitEvents=false to advance a CSRegisterHandler over the
// chunk prefix without surfacing events to a caller that didn't ask for them.
template <bool EmitEvents> rocprofiler_thread_trace_decoder_status_t process_events_gfx12(
    CSRegisterHandler& csregister,
    const std::vector<quick_scan::QuickToken>& raw,
    size_t n,
    uint64_t header_skip,
    rocprof_trace_decoder_trace_callback_t trace_callback,
    void* userdata
)
{
    if (n == 0) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;

    for (size_t i = 0; i < n; ++i)
    {
        const auto& tok = raw[i];

        if (tok.type == RdnaType::REG)
        {
            gfx10::reg_write_type reg{.raw = tok.contents};

            if (reg.CS)
                csregister.UpdateRegCS(reg);
            else
                csregister.UpdateRegNoCS(reg);
        }

        if (!EmitEvents) continue;

        if (tok.type == RdnaType::REG_INIT)
        {
            gfx10::reg_init_type reg{.raw = tok.contents};
            if (reg.type != 2 || (reg.data & 1) == 0) continue;

            rocprofiler_thread_trace_decoder_dispatch_t dispatch = csregister.PopulateDispatch(0, reg.me, reg.pipe);
            dispatch.byte_offset = static_cast<uint64_t>(tok.offset);
            auto status = trace_callback(ROCPROFILER_THREAD_TRACE_DECODER_RECORD_DISPATCH, &dispatch, 1, userdata);
            if (status != ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS) return status;
        }
        else if (tok.type == RdnaType::EVENT)
        {
            gfx10::event_type event{.raw = tok.contents};

            rocprofiler_thread_trace_decoder_event_type_t type{};

            switch (event.id)
            {
                case EVENT_CS_PARTIAL_FLUSH: type = ROCPROF_TRACE_DECODER_EVENT_CS_PARTIAL_FLUSH; break;
                case EVENT_CACHE_FLUSH:
                case EVENT_CACHE_FLUSH_WR:
                case EVENT_CACHE_FLUSH_INV:
                case EVENT_CACHE_FLUSH_INV_WR: type = ROCPROF_TRACE_DECODER_EVENT_CACHE_FLUSH; break;
                case EVENT_BOTTOM_OF_PIPE_WR: type = ROCPROF_TRACE_DECODER_EVENT_BOTTOM_OF_PIPE_TS; break;
                case EVENT_TT_FLUSH: type = ROCPROF_TRACE_DECODER_EVENT_TT_FLUSH; break;
                default: break;
            }

            if (type == ROCPROF_TRACE_DECODER_EVENT_NONE) continue;

            rocprofiler_thread_trace_decoder_event_t ev{};
            ev.size = sizeof(ev);
            ev.time = 0;
            ev.type = type;
            ev.me_id = static_cast<uint8_t>(event.me & 1);
            ev.pipe_id = static_cast<uint8_t>(event.pipe);
            ev.flags = ROCPROF_TRACE_DECODER_EVENT_FLAGS_PER_PIPE;
            if (event.bop) ev.flags |= ROCPROF_TRACE_DECODER_EVENT_FLAGS_BOP;
            ev.payload.raw = 0;
            ev.byte_offset = static_cast<uint64_t>(tok.offset);
            auto status = trace_callback(ROCPROFILER_THREAD_TRACE_DECODER_RECORD_EVENT, &ev, 1, userdata);
            if (status != ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS) return status;
        }
    }

    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

} // namespace

extern "C"
{
ROCPROF_TRACE_DECODER_API rocprofiler_thread_trace_decoder_status_t rocprof_trace_decoder_quick_scan(
    rocprof_trace_decoder_handle_t handle,
    uint64_t chunk_index,
    const void* data,
    uint64_t data_size,
    rocprof_trace_decoder_trace_callback_t trace_callback,
    void* userdata
)
{
    thread_local std::vector<quick_scan::QuickToken> raw{1u << 18};

    if (!quick_scan::avx512_available()) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_NOT_IMPLEMENTED;

    if (!data || data_size == 0)
    {
        for (auto& r : raw) r = quick_scan::QuickToken{};
        return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
    }

    // Every path below dereferences an 8-byte header word (load_header_word) and
    // gfx9 chunk 0 subtracts an 8-byte header_skip; a chunk smaller than that
    // word can't hold a header or a full token. Reject it before any 8-byte read.
    if (data_size < sizeof(uint64_t)) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

    TIMING(t0);

    int gfxip = 0;
    CSRegisterHandler local;

    const uint8_t* buf = static_cast<const uint8_t*>(data);
    uint64_t header_skip = 0;

    if (chunk_index == 0)
    {
        uint64_t header_word = load_header_word(data);
        gfxip = extract_gfxip(header_word);
        if (gfxip > 9) local.tt_version = mi400::header_type{.raw = header_word}.version;

        auto decoder = HandleData::get_write_handle(handle);
        if (!decoder.valid()) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

        decoder->gfxip = gfxip;
        if (gfxip != 0) decoder->trace_header = header_word;
        if (gfxip == 9) header_skip = 8;

        if (data_size == header_skip)
        {
            decoder->pipestate.put(chunk_index + 1, std::move(local));
            decoder->cv.notify_all();
            return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
        }
    }
    else
    {
        auto decoder = HandleData::get_read_handle(handle);
        if (!decoder.valid()) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

        if (decoder->gfxip == 0) decoder->gfxip = extract_gfxip(load_header_word(data));

        gfxip = decoder->gfxip;
    }

    TIMING(t1);

    data_size -= header_skip;
    buf += header_skip;

    size_t ntokens = 0;
#if ROCPROF_TRACE_DECODER_QUICK_SCAN_HAS_SIMD
    try
    {
        auto scanner = (gfxip == 9) ? &quick_scan::scan_gfx9 : &scan_none;
        if (gfxip == 12) scanner = &quick_scan::scan_gfx12;
        if (gfxip == 1250) scanner = &quick_scan::scan_mi400;

        ntokens = scanner(buf, data_size, raw.data(), raw.size());
        while (ntokens == raw.size())
        {
            raw.resize(raw.size() * 2);
            ntokens = scanner(buf, data_size, raw.data(), raw.size());
        }
    }
    catch (std::exception&)
#endif
    {
        return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_NOT_IMPLEMENTED;
    }

    TIMING(t2);

    if (chunk_index != 0)
    {
        auto decoder = HandleData::get_read_handle(handle);
        if (!decoder.valid()) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

        bool ready = false;
        int count = 0;

        // wait up to a second
        while (!ready && count < 100000 / 40)
        {
            count++;
            ready = decoder->cv.wait_for(
                decoder.lk,
                std::chrono::microseconds(40),
                [&]()
                {
                    const CSRegisterHandler* p = decoder->pipestate.get(chunk_index);
                    if (!p) return false;
                    // Copy under the read lock; the borrowed pointer is invalid
                    // once we release it, but `local` now owns its own state
                    // (CowPtr refcount bumps on the shared codeobj vector + tables).
                    local = *p;
                    return true;
                }
            );
        }

        if (!ready) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES;
    }

    TIMING(t3);

    if (!trace_callback) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

    rocprofiler_thread_trace_decoder_status_t status;
    if (gfxip == 9)
        status = process_events_gfx9<true>(local, raw, ntokens, header_skip, trace_callback, userdata);
    else if (gfxip != 0)
        status = process_events_gfx12<true>(local, raw, ntokens, header_skip, trace_callback, userdata);
    else
        status = process_events_none(local, raw, ntokens, header_skip, trace_callback, userdata);

    TIMING(t4);

    std::condition_variable_any* cv = nullptr;
    {
        auto decoder = HandleData::get_write_handle(handle);
        if (!decoder.valid()) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

        decoder->pipestate.put(chunk_index + 1, std::move(local));
        cv = &decoder->cv;
    }
    cv->notify_all();

    TIMING(t5);

#ifdef GET_TIMING
    if ((t3 - t2).count() > data_size / 16 && (t3 - t2).count() > 1000)
    {
        std::cout << chunk_index << " - " << DELTA(3, 2) DELTA(1, 0) DELTA(2, 1) DELTA(4, 3) DELTA(5, 4) std::endl;
    }
#endif

    return status;
}

ROCPROF_TRACE_DECODER_API rocprofiler_thread_trace_decoder_status_t rocprof_trace_decoder_build_standalone(
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
    if (!data || data_size <= 8 || !data_out || !size_out || offset_begin + 8 >= offset_end || offset_end > data_size)
        return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

    const uint8_t* buf = static_cast<const uint8_t*>(data);

    // Snapshot the gfxip, the saved arch header, and a copy of the
    // entry-state CSRegisterHandler under a single read lock. The local copy
    // owns its own state (CowPtr refcount bumps on the codeobj vector and
    // tables) so we can release the lock before doing the per-call work and
    // the in-place state advance below doesn't mutate the cached snapshot.
    int gfxip = 0;
    uint64_t header_word = 0;
    CSRegisterHandler temp;

    {
        auto decoder = HandleData::get_read_handle(handle);
        if (!decoder.valid()) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

        gfxip = decoder->gfxip;
        header_word = decoder->trace_header;

        if (header_word == 0)
        {
            header_word = load_header_word(data);
            if (gfxip == 0) gfxip = extract_gfxip(header_word);
        }

        if (chunk_index == 0)
            temp = CSRegisterHandler{};
        else if (const CSRegisterHandler* p = decoder->pipestate.get(chunk_index))
            temp = *p;
        else
            return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;
    }

    if (gfxip != 9 && gfxip != 12 && gfxip != 1250)
        return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_NOT_IMPLEMENTED;

    // Walk the prefix [0, offset_begin) through the private CSRegisterHandler
    // copy taken above. EmitEvents=false: this is a state-advance pass,
    // no callback is invoked.

    thread_local std::vector<quick_scan::QuickToken> raw{1u << 16};

    if (!quick_scan::avx512_available()) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_NOT_IMPLEMENTED;

    size_t ntokens = 0;
#if ROCPROF_TRACE_DECODER_QUICK_SCAN_HAS_SIMD
    auto scanner = (gfxip == 9) ? &quick_scan::scan_gfx9 : &scan_none;
    if (gfxip == 12) scanner = &quick_scan::scan_gfx12;
    if (gfxip == 1250) scanner = &quick_scan::scan_mi400;

    ntokens = scanner(buf, offset_begin, raw.data(), raw.size());
    while (ntokens == raw.size())
    {
        raw.resize(raw.size() * 2);
        ntokens = scanner(buf, offset_begin, raw.data(), raw.size());
    }
#else
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_NOT_IMPLEMENTED;
#endif

    if (gfxip == 9)
        process_events_gfx9<false>(temp, raw, static_cast<int>(ntokens), 0, nullptr, nullptr);
    else if (gfxip != 0)
        process_events_gfx12<false>(temp, raw, static_cast<int>(ntokens), 0, nullptr, nullptr);

    // Materialise the synthetic context as SQTT tokens. Wrapped in a
    // gfxip-keyed dispatch so additional architectures can plug their own
    // BuildStatusTokens_* in later without touching this control flow.
    std::vector<StatusToken> status_tokens = build_status_tokens(gfxip, temp);

    // Output is always self-describing: [8B arch header] [synthetic context
    // tokens] [verbatim data slice]. The header is unconditional so every
    // built buffer can be replayed standalone.
    constexpr uint64_t header_bytes = 8;
    size_t status_bytes = 0;
    for (const auto& t : status_tokens) status_bytes += t.bytes;

    uint64_t data_slice_bytes = offset_end - offset_begin;
    uint64_t total = header_bytes + status_bytes + data_slice_bytes;

    if (*size_out < total)
    {
        *size_out = total;
        return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_OUT_OF_RESOURCES;
    }

    uint8_t* buf_out = static_cast<uint8_t*>(data_out);
    std::memcpy(buf_out, &header_word, header_bytes);
    size_t used = header_bytes;
    used += write_status_tokens(buf_out + used, status_tokens);
    std::memcpy(buf_out + used, buf + offset_begin, data_slice_bytes);
    used += data_slice_bytes;

    *size_out = used;
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

} // extern "C"
