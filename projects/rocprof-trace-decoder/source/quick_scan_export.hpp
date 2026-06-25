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

#include "rocprof_trace_decoder/rocprof_trace_decoder.h"
#include "rocprof_trace_decoder/trace_decoder_instrument.h"

#include "handle.hpp"
#include "trace_parser.hpp" // CSRegisterHandler, sqtt_token_reg_t, sqtt_event_type_t

//#define GET_TIMING

#ifndef ROCPROF_TRACE_DECODER_QUICK_SCAN_HAS_SIMD
#    if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#        define ROCPROF_TRACE_DECODER_QUICK_SCAN_HAS_SIMD 1
#    else
#        define ROCPROF_TRACE_DECODER_QUICK_SCAN_HAS_SIMD 0
#    endif
#endif

namespace quick_scan
{
// Captured rare-token entry. `contents` is the full 64-bit window starting
// at the token's first byte (always safe — every gfx9 rare type is <= 8 bytes).
// Only the bits within the type's encoded field width are meaningful;
// downstream consumers should mask per-type (see source/gfx9/gfx9token.h).
//
// `type` is a value of gfx9::sqtt_token_type_t (the low-nibble of the first
// byte). Currently captured rare set: REG(2), REG_CS(5), EVENT(7),
// EVENT_CS(8), REG_CS_PRIV(15).
//
// `offset` is the token's first-byte position within the buffer passed to
// scan_* (for nibble-packed gfx12 streams, this is the containing byte).
// gfx9 scans chunk 0 post-header, so callers that need an offset relative to
// their original `data` buffer must add the header skip; RDNA/MI400 buffers are
// scanned without a header skip.
// Packed as a 16:48 bitfield with `type` to keep sizeof at 16B. 48 bits of
// offset covers chunks up to 256 TiB; placing `type` in the low bits lets
// the consumer's hot dispatch (cmp tok.type, ...) fold the load+mask into
// a single 16-bit compare against memory, saving a shift per access.
struct QuickToken
{
    uint64_t contents;
    uint64_t type   : 16;
    uint64_t offset : 48;
};

// The scanners are SIMD-only and built only on GCC/clang x86-64. On other
// compilers (MSVC), other architectures (aarch64, non-x86), or x86 CPUs at
// runtime that lack AVX-512, no implementation is available. Availability
// is probed at the export entry point (rocprof_trace_decoder_quick_scan
// with data=nullptr), which returns ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_NOT_IMPLEMENTED
// when the scanner cannot run. Callers must NOT invoke these scan_*
// functions directly without first confirming availability via that probe.
//
// Purpose-built fast scanners that walk an SQTT token stream and capture
// only the rare-token cluster (REG / REG_CS / EVENT / EVENT_CS /
// REG_CS_PRIV for gfx9) with full 64-bit contents. Skips everything else
// with a small length LUT — no Token{} construction, no patch_time()
// lookahead, no globaltime tracking.
//
// The buffer must point AFTER the gfx9 8-byte header
// (rocprof_trace_decoder_gfx9_header_t). Size is the post-header byte
// count. See iterate_tokens.hpp:75-87 for the convention.
//
// Writes up to `out_cap` entries into `out` in stream order and returns
// the number written. Single-threaded, no exceptions.
#if ROCPROF_TRACE_DECODER_QUICK_SCAN_HAS_SIMD
size_t scan_gfx9(const uint8_t* buf, size_t size, QuickToken* __restrict__ out, size_t out_cap);
size_t scan_gfx12(const uint8_t* buf, size_t size, QuickToken* __restrict__ out, size_t out_cap);
size_t scan_mi400(const uint8_t* buf, size_t size, QuickToken* __restrict__ out, size_t out_cap);
#endif

// Returns true iff the running CPU supports AVX-512 (vbmi+bw+f) and the
// TU was built for x86 with a compiler that has the SIMD paths. False on
// MSVC, on non-x86 targets, or on x86 CPUs lacking AVX-512. Used by the
// export to decide between dispatching the scanner and returning
// ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_NOT_IMPLEMENTED.
bool avx512_available();
}; // namespace quick_scan
