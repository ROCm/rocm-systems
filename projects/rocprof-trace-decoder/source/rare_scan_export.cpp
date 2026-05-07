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
//
// Quick-and-dirty C-ABI exports for the per-arch rare-token scanners. Returned
// data is a packed array of 96-bit entries: { uint64_t contents; uint32_t type; }.
// Callers must duplicate that struct definition (intentional for early use; a
// proper header lands when the wider rare-scan API is finalized).

#include <cstddef>
#include <cstdint>

#include "gfx9/rare_scan.h"

#define RARE_SCAN_EXPORT __attribute__((visibility("default")))

extern "C"
{
// Scan a gfx9 SQTT token stream for the rare-token cluster (REG / REG_CS /
// EVENT / EVENT_CS / REG_CS_PRIV).
//
// The input must be the post-header byte range — i.e. a gfx9 token stream
// with the 8-byte rocprof_trace_decoder_gfx9_header_t already stripped. The
// header is only present once at the very start of a trace; in triple-buffer
// mode rocprofiler-sdk delivers it as a standalone 8-byte shader-data chunk
// before the first payload chunk, so subsequent chunks are already
// header-free. Callers that hold the full single-buffer trace should advance
// their pointer past the 8-byte header before invoking this function.
//
// Returns the number of RareToken entries written to `out` (capped at out_cap).
RARE_SCAN_EXPORT size_t
rocprof_trace_decoder_rare_scan_gfx9(const uint8_t* buf,
                                     size_t         size,
                                     void*          out,
                                     size_t         out_cap)
{
    if (!buf || size == 0 || !out || out_cap == 0) return 0;
    return gfx9::rare_scan::scan_gfx9(
        buf, size, static_cast<gfx9::rare_scan::RareToken*>(out), out_cap);
}
}
