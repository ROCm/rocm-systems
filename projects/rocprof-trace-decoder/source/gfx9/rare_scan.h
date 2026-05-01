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

#include <cstddef>
#include <cstdint>

namespace gfx9::rare_scan
{

// Captured rare-token entry. `contents` is the full 64-bit window starting
// at the token's first byte (always safe — every gfx9 rare type is <= 8 bytes).
// Only the bits within the type's encoded field width are meaningful;
// downstream consumers should mask per-type (see source/gfx9/gfx9token.h).
//
// `type` is a value of gfx9::sqtt_token_type_t (the low-nibble of the first
// byte). Currently captured rare set: REG(2), REG_CS(5), EVENT(7),
// EVENT_CS(8), REG_CS_PRIV(15).
struct RareToken
{
    uint64_t contents;
    uint32_t type;
};

// Purpose-built fast scanner that walks a gfx9 SQTT token stream and
// captures only the rare-token cluster (REG / REG_CS / EVENT / EVENT_CS /
// REG_CS_PRIV) with full 64-bit contents. Skips everything else with a
// 16-entry nibble-keyed length LUT — no Token{} construction, no
// patch_time() lookahead (which only reorders TIME tokens, none of which
// are in the rare cluster), no globaltime tracking.
//
// The buffer must point AFTER the gfx9 8-byte header
// (rocprof_trace_decoder_gfx9_header_t). Size is the post-header byte
// count. See iterate_tokens.hpp:75-87 for the convention.
//
// Writes up to `out_cap` entries into `out` in stream order; returns the
// number written. Single-threaded, no exceptions.
size_t scan_gfx9(const uint8_t* buf, size_t size, RareToken* out, size_t out_cap);

} // namespace gfx9::rare_scan
