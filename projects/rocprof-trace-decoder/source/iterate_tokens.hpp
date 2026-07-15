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

// Templated token-iteration helper. Pulled out of trace_parser.hpp so the
// per-arch token generator headers (which are heavy) only land in the TUs
// that actually want to walk tokens — the bench, primarily. Kept header-only
// so the visitor inlines into the inner loop instead of going through a
// std::function indirect call.

#pragma once

#include <cstdint>
#include <exception>
#include <vector>

#include "gfx10/gfx10parser.h"
#include "gfx11/gfx11parser.h"
#include "gfx12/gfx12parser.h"
#include "gfx9/gfx9token.h" // gfx9 has no separate parser.h; MITokenGenerator lives here
#include "mi400/mi400parser.h"
#include "trace_parser.hpp"

// Walks every SQTT token in `buffer` and invokes `on_token(int token_type)`
// once per token, without any wave-state reconstruction or stitching. The
// integer passed to on_token belongs to gfx9::sqtt_token_type_t for
// TraceArch::GFX9, otherwise RdnaType (see source/gfx10/token_types.h).
// Returns the architecture detected from the header, or TraceArch::UNKNOWN
// if dispatch fails.
//
// Optional `rare_out`: if non-null, the mi400 fast path captures the full
// 64-bit contents of REG / REG_INIT / EVENT / EVENT_SYNC tokens into this
// vector (preallocate via .reserve() — these types are uncommon but may
// number in the tens of thousands on long traces).
//
// Templated on the visitor so the callback inlines into the loop body.
template <class F> TraceArch IterateTokens_internal(
    const uint8_t* buffer,
    uint64_t buffer_size,
    F&& on_token,
    std::vector<mi400::TokenGenerator::QuickToken>* rare_out = nullptr
)
{
    auto arch = DetectArch_internal(buffer, buffer_size);

    auto run_navi = [&](auto generator)
    {
        try
        {
            while (generator.nextValid()) on_token(int(generator.next().type));
        }
        catch (const std::exception&)
        {}
    };

    switch (arch)
    {
        case TraceArch::GFX9:
        {
            const uint8_t* tokens = buffer + sizeof(rocprof_trace_decoder_gfx9_header_t);
            uint64_t tokens_size = buffer_size - sizeof(rocprof_trace_decoder_gfx9_header_t);
            try
            {
                gfx9::MITokenGenerator gen(tokens, tokens_size, 0, 0);
                while (gen.valid()) on_token(int(gen.next().type));
            }
            catch (const std::exception&)
            {}
            break;
        }
        case TraceArch::MI400:
        {
            // mi400 has a dedicated type-only fast scanner (windowed multi-token
            // decode). Forwards the optional rare-token capture sink so callers
            // get full contents for REG / REG_INIT / EVENT / EVENT_SYNC.
            try
            {
                mi400::TokenGenerator gen(buffer, buffer_size, 0, 0);
                if (rare_out)
                    gen.scan_types(on_token, *rare_out);
                else
                    gen.scan_types(on_token);
            }
            catch (const std::exception&)
            {}
            break;
        }
        case TraceArch::GFX12: run_navi(gfx12::TokenGenerator(buffer, buffer_size, 0, 0)); break;
        case TraceArch::GFX11: run_navi(gfx11::TokenGenerator(buffer, buffer_size, 0, 0)); break;
        case TraceArch::GFX10: run_navi(gfx10::TokenGenerator(buffer, buffer_size, 0, 0)); break;
        case TraceArch::UNKNOWN: break;
    }
    return arch;
}
