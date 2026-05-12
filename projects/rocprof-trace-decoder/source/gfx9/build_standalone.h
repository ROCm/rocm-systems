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

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

class CSRegisterHandler;

namespace gfx9::build_standalone
{

// One synthesised SQTT token. `bits` holds the raw 64-bit token payload
// (low-bit first, matching the on-wire layout the gfx9 scanner expects);
// `bytes` is how many of those bytes are meaningful and should land in the
// output buffer (REG = 8, REG_CS / REG_CS_PRIV = 6).
struct StatusToken
{
    uint64_t bits;
    uint8_t  bytes;
};

// Produce the stream of synthetic SQTT tokens that, when re-fed through a
// fresh CSRegisterHandler, recreates the state captured in `reg`.
//
// Emits, in stream order:
//   1. Per-(me,pipe) REG_CS writes for COMPUTE_PGM_LO/HI and
//      COMPUTE_DISPATCH_PKT_LO/HI (latched 64-bit values stored in
//      reg.wave_start_addr / reg.dispatch_pkt_addr).
//   2. Scalar REG_CS writes on (me=0,pipe=0) for COMPUTE_NUM_THREAD_X/Y/Z
//      and COMPUTE_PGM_RSRC1/2/3 (last-write-wins latches).
//   3. If reg.bIsROCMFormat is set: the "\0ROC" USERDATA2 enable header,
//      followed by a 14-write CODEOBJ marker sequence per active code
//      object (ID_LO/HI, SIZE_LO/HI, ADDR_LO/HI, TAIL with bFromStart=1).
//      Sizes are recovered from reg.table.find_codeobj_in_range(addr).
//
// Callers should pass the result to write_tokens() to serialise into the
// output buffer.
std::vector<StatusToken> build_status_tokens(const CSRegisterHandler& reg);

// Stream a token vector into a byte buffer, copying each token's low
// `bytes` bytes little-endian (matches the layout the gfx9 scanner reads).
// Returns the number of bytes written. The caller is responsible for
// ensuring `out` has room for sum(t.bytes for t in tokens).
size_t write_tokens(uint8_t* out, const std::vector<StatusToken>& tokens);

} // namespace gfx9::build_standalone
