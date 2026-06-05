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

size_t scan_none(const uint8_t* buf, size_t size, QuickToken* out, size_t out_cap) { return 0; };

rocprofiler_thread_trace_decoder_status_t process_events_none(
    CSRegisterHandler& csregister,
    const std::vector<QuickToken>& raw,
    int n,
    uint64_t header_skip,
    rocprof_trace_decoder_trace_callback_t trace_callback,
    void* userdata
)
{
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
};

// Status-token reconstruction (build_standalone) — see
// source/gfx9/build_standalone.{h,cpp}. Per-arch builders live next to
// the per-arch scanner; this TU only orchestrates.

inline int extract_gfxip(uint64_t header)
{
    {
        rocprof_trace_decoder_gfx9_header_t h{.raw = header};
        if ((h.legacy_version == 0 || h.legacy_version == 0x11) && (h.gfx9_version2 >= 4 && h.gfx9_version2 <= 6))
            return 9;
    }

    auto hw_header = mi400::header_type{.raw = header};

    if (hw_header.version == 4) return 12;
    return 0;
}

// Templated on whether to emit dispatch/event records. The state-update path
// (UpdateRegCS / UpdateRegNoCS) runs unconditionally; build_standalone
// instantiates with EmitEvents=false to advance a CSRegisterHandler over the
// chunk prefix without surfacing events to a caller that didn't ask for them.
template <bool EmitEvents> rocprofiler_thread_trace_decoder_status_t process_events_gfx9(
    CSRegisterHandler& csregister,
    const std::vector<quick_scan::QuickToken>& raw,
    int n,
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
                    ev.reserved = 0;
                    ev.payload = 0;
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
    if (!quick_scan::avx512_available()) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_NOT_IMPLEMENTED;

    thread_local std::vector<quick_scan::QuickToken> raw{1u << 18};

    if (!data || data_size == 0) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
    if (!trace_callback) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

    TIMING(t0);

    int gfxip = 0;
    // Local working copy of the entry-state register tracker. For chunk 0
    // this stays default-constructed; for later chunks we copy the
    // previous chunk's pipestate slot into it under the read lock below.
    // CSRegisterHandler move/copy is cheap (scalars + std::array + CowPtr
    // refcount bumps) so neither path allocates per-chunk.
    CSRegisterHandler local;

    const uint8_t* buf = static_cast<const uint8_t*>(data);
    // Only chunk 0 carries the 8-byte arch header at the start of the
    // buffer; later chunks are pure token streams. `header_skip` is the byte
    // distance between the caller's `data` and the buffer the scanner sees,
    // and is added back to tok.offset when reporting byte_offsets so they
    // stay relative to the caller's `data`.
    uint64_t header_skip = 0;

    if (chunk_index == 0)
    {
        uint64_t header_word = static_cast<const uint64_t*>(data)[0];
        gfxip = extract_gfxip(header_word);

        auto decoder = HandleData::get_write_handle(handle);
        if (!decoder.valid()) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;

        decoder->gfxip = gfxip;
        // Snapshot the raw chunk-0 header so build_standalone can prepend it
        // to the rebuilt buffer (it isn't reachable any other way once the
        // caller's data buffer is gone).
        if (gfxip == 9)
        {
            decoder->gfx9_header = header_word;
            header_skip = 8;
        }

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

        if (decoder->gfxip == 0) decoder->gfxip = extract_gfxip(static_cast<const uint64_t*>(data)[0]);

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

    rocprofiler_thread_trace_decoder_status_t status;
    if (gfxip == 9)
        status = process_events_gfx9<true>(local, raw, ntokens, header_skip, trace_callback, userdata);
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
        header_word = decoder->gfx9_header;

        if (const CSRegisterHandler* p = decoder->pipestate.get(chunk_index))
            temp = *p;
        else
            return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_ARGUMENT;
    }

    if (gfxip != 9) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_NOT_IMPLEMENTED;

    const uint8_t* buf = static_cast<const uint8_t*>(data);

    // Walk the prefix [0, offset_begin) through the private CSRegisterHandler
    // copy taken above. EmitEvents=false: this is a state-advance pass,
    // no callback is invoked.

    thread_local std::vector<quick_scan::QuickToken> raw{1u << 16};

    if (!quick_scan::avx512_available()) return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_NOT_IMPLEMENTED;

    size_t ntokens = 0;
#if ROCPROF_TRACE_DECODER_QUICK_SCAN_HAS_SIMD
    ntokens = quick_scan::scan_gfx9(buf, offset_begin, raw.data(), raw.size());
    while (ntokens == raw.size())
    {
        raw.resize(raw.size() * 2);
        ntokens = quick_scan::scan_gfx9(buf, offset_begin, raw.data(), raw.size());
    }
#else
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_NOT_IMPLEMENTED;
#endif

    process_events_gfx9</*EmitEvents=*/false>(temp, raw, static_cast<int>(ntokens), 0, nullptr, nullptr);

    // Materialise the synthetic context as SQTT tokens. Wrapped in a
    // gfxip-keyed dispatch so additional architectures can plug their own
    // BuildStatusTokens_* in later without touching this control flow.
    std::vector<gfx9::build_standalone::StatusToken> status_tokens;
    if (gfxip == 9)
        status_tokens = gfx9::build_standalone::build_status_tokens(temp);
    else
        return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_NOT_IMPLEMENTED;

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
    used += gfx9::build_standalone::write_tokens(buf_out + used, status_tokens);
    std::memcpy(buf_out + used, buf + offset_begin, data_slice_bytes);
    used += data_slice_bytes;

    *size_out = used;
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

} // extern "C"
