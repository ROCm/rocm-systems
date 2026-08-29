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
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#pragma once

// Pure exact-geometry validation for a KFD dispatch-log stream, factored out of
// kfd_reader.cpp so it can be unit-tested without the vendored UAPI header (which
// is translation-unit-exclusive) or a live stream. The validator takes plain
// scalars pulled from kfd_dlog_stream_info and the requested ring size, and checks
// them against the kernel's canonical BO layout:
//   records[buffer_size] | wptr[num_regions] | rptr[num_regions] | page-pad
// Every offset, the derived per-region record count, and the page-aligned mmap
// size are fixed by the ABI, so anything else is a contract violation.

#include "lib/rocprofiler-sdk/kfd/dlog_drain.hpp"  // kFwRecBytes, kMaxRegions

#include <cstddef>
#include <cstdint>

namespace rocprofiler
{
namespace kfd
{
// Round `x` up to a multiple of `page` (page must be a power of two, > 0).
inline uint64_t
round_up_to_page(uint64_t x, uint64_t page)
{
    return (x + page - 1) & ~(page - 1);
}

// The geometry fields the validator checks, all straight from kfd_dlog_stream_info.
struct stream_geometry
{
    uint32_t num_regions         = 0;
    uint32_t region_record_count = 0;
    uint64_t buffer_size         = 0;
    uint64_t mmap_size           = 0;
    uint64_t records_offset      = 0;
    uint64_t wptr_offset         = 0;
    uint64_t rptr_offset         = 0;
};

// Which validation check rejected the geometry. Kept as a returned value rather
// than a log line so the validator stays pure and unit-testable; the caller
// (which has the singletons) turns it into a warning.
enum class geometry_reason
{
    ok,                    // accepted
    buffer_size_mismatch,  // kernel buffer_size != the size we requested
    bad_region_layout,     // num_regions out of range, or region_record_count not a power of two
    layout_mismatch,       // an offset, the record count, the mmap size, or region disjointness
};

inline const char*
geometry_reason_name(geometry_reason r)
{
    switch(r)
    {
        case geometry_reason::buffer_size_mismatch: return "buffer-size-mismatch";
        case geometry_reason::bad_region_layout: return "bad-region-layout";
        case geometry_reason::layout_mismatch: return "layout-mismatch";
        case geometry_reason::ok: break;
    }
    return "ok";
}

// Result of validation: on success, `mmap_len` is the exact page-aligned mapping
// length the caller must mmap. On failure, `ok` is false, `mmap_len` is 0, and
// `reason` says which check rejected it.
struct stream_geometry_result
{
    bool            ok       = false;
    uint64_t        mmap_len = 0;
    geometry_reason reason   = geometry_reason::ok;
};

// Validate `g` (the kernel-returned geometry) against `requested_buffer_size` and
// `page_size`. Rejects an unsupported region layout copy_pipes() could not drain,
// then requires every offset / count / mmap size to match the canonical layout
// exactly, and requires the three regions to be disjoint. All arithmetic is in
// u64; num_regions is bounded by kMaxRegions and buffer_size is a bounded u32, so
// nothing wraps.
inline stream_geometry_result
validate_stream_geometry(const stream_geometry& g,
                         uint64_t               requested_buffer_size,
                         uint64_t               page_size)
{
    if(g.buffer_size != requested_buffer_size)
        return {false, 0, geometry_reason::buffer_size_mismatch};

    const uint32_t rrc = g.region_record_count;
    if(g.num_regions == 0 || g.num_regions > kMaxRegions || rrc == 0 || (rrc & (rrc - 1)) != 0)
        return {false, 0, geometry_reason::bad_region_layout};

    const uint64_t nr        = g.num_regions;
    const uint64_t ptr_bytes = nr * sizeof(uint64_t);
    const uint64_t exp_wptr  = g.buffer_size;
    const uint64_t exp_rptr  = g.buffer_size + ptr_bytes;
    const uint64_t exp_mmap  = round_up_to_page(exp_rptr + ptr_bytes, page_size);
    const uint64_t exp_rrc   = g.buffer_size / (nr * kFwRecBytes);

    // Disjoint by construction once the offsets are exact, but asserted directly:
    // records ends at wptr start, wptr ends at rptr start, rptr ends within mmap.
    const bool disjoint =
        g.records_offset < exp_wptr && exp_wptr < exp_rptr && exp_rptr + ptr_bytes <= exp_mmap;

    if(g.records_offset != 0 || g.wptr_offset != exp_wptr || g.rptr_offset != exp_rptr ||
       g.mmap_size != exp_mmap || rrc != exp_rrc || !disjoint)
        return {false, 0, geometry_reason::layout_mismatch};

    return stream_geometry_result{true, exp_mmap, geometry_reason::ok};
}
}  // namespace kfd
}  // namespace rocprofiler
