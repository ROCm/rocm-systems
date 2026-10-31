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

#include "../quick_scan_export.hpp"

#include <cstring>

#include "gfx10/token_types.h"
#include "mi400parser.h"

// SIMD-only scanner — no scalar fallback. The AVX-512 path uses GCC
// function multiversioning (__attribute__((target("…")))) which MSVC does
// not implement, so MSVC and non-x86 targets compile to a stub that
// returns SCAN_NOT_IMPLEMENTED. x86 CPUs without AVX-512 also resolve to
// the stub at runtime.
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#    include <immintrin.h>
#    define MI400_RARE_SCAN_HAS_X86 1
#else
#    define MI400_RARE_SCAN_HAS_X86 0
#endif

#if MI400_RARE_SCAN_HAS_X86

namespace mi400::quick_scan
{
namespace
{
using ::quick_scan::QuickToken;
using ::quick_scan::ScanState;

// Single combined byte per first-byte slot, derived once from the canonical
// mi400::TokenLookupTable at TU init. Any future AddEncoding edit in
// mi400token.cpp / gfx12 / gfx11 / gfx10 propagates automatically.
//
// Per-byte info encoding, used by the scalar walks:
//   bits 0-3 : token byte length (max 9)
//   bit  4   : is_rare (first byte of REG/REG_INIT/EVENT/EVENT_SYNC/UNKNOWN)
//   bits 5-7 : unused
//
// MI400 tokens are whole bytes; sub-byte entries inherited from the gfx12
// table are never emitted and are clamped to one byte so they cannot look like
// rare tokens (jump 0). WAVE_START_EXT is an ordinary fixed-length (40-bit)
// token; it is not followed by extension bytes.
//
// `jump` and `len` are the vector LUTs: jump[b] is 0 for rare bytes and the
// token length otherwise, so that index + jump[b] is the jump function g
// described below; len[b] is the token length for every byte.
struct Tables
{
    uint8_t info[256];
    uint8_t rare_type[256]; // RdnaType id (1..4); valid only when info[b] & 0x10
    alignas(64) uint8_t jump[256];
    alignas(64) uint8_t len[256];
};

constexpr uint8_t INFO_RARE = 0x10;
constexpr uint8_t INFO_LEN = 0x0F;
constexpr uint8_t MAX_TOKEN_BYTES = 9;

const Tables& tables()
{
    static const Tables t = []
    {
        Tables out{};
        mi400::TokenLookupTable lk{}; // populates all 256 entries
        for (int b = 0; b < 256; ++b)
        {
            const auto& info = lk.lookup(static_cast<uint8_t>(b));
            const unsigned len = static_cast<unsigned>(info.length) >> 3;
            const unsigned type = info.type;
            if (len > MAX_TOKEN_BYTES) __builtin_trap();
            uint8_t v = static_cast<uint8_t>((len ? len : 1) & INFO_LEN);
            if (type < mi400::TokenGenerator::RARE_END)
            {
                v |= INFO_RARE;
                out.rare_type[b] = static_cast<uint8_t>(type);
            }
            out.info[b] = v;
            out.len[b] = v & INFO_LEN;
            out.jump[b] = (v & INFO_RARE) ? 0 : (v & INFO_LEN);
        }
        return out;
    }();
    return t;
}

#    define MI400_SCAN_TARGET __attribute__((target("avx512vbmi,avx512bw,avx512f,bmi2")))

MI400_SCAN_TARGET inline unsigned lane0(__m512i v)
{ return static_cast<uint8_t>(_mm_cvtsi128_si32(_mm512_castsi512_si128(v))); }

// 256-entry byte LUT held in four zmm registers: vpermi2b covers 128 entries
// using the low 7 index bits, so the two halves are blended on bit 7.
MI400_SCAN_TARGET inline __m512i lookup256(const __m512i (&lut)[4], __m512i bytes_v, __mmask64 hi_mask)
{
    const __m512i lo = _mm512_permutex2var_epi8(lut[0], bytes_v, lut[1]);
    const __m512i hi = _mm512_permutex2var_epi8(lut[2], bytes_v, lut[3]);
    return _mm512_mask_blend_epi8(hi_mask, lo, hi);
}

// AVX-512_VBMI path. Per 64-byte chunk:
//   1. Look up jump[byte] and form g[i] = i if a rare token starts at byte i,
//      else i + len(i). Exit lanes (>= 64) are fixed points through
//      exit_table_v.
//   2. Square g six times. Tokens are at least one byte long, so g^64[i] is
//      the first rare token on the token-start orbit from i, or
//      64 + the next chunk's entry if there is none.
//   3. Fast path: g^64[entry] leaves the chunk.
//   4. Slow path: capture the rare token, then jump straight to the next one
//      through g^64 sampled just past it.
//
// State carried across chunks: the entry (byte offset within the next chunk
// where the next token starts; 0..MAX_TOKEN_BYTES-1).
//
// With `state`, the scan starts at state->pos and, when `out` fills, stops
// right after the last captured token so a later call can resume.
MI400_SCAN_TARGET size_t scan_mi400_avx512(
    const uint8_t* buf, size_t size, QuickToken* __restrict__ out, size_t out_cap, ScanState* state
)
{
    if (!buf || !out || out_cap == 0 || size == 0)
    {
        if (state && (!buf || size == 0)) state->done = true;
        return 0;
    }

    const Tables& T = tables();

    constexpr size_t CHUNK = 64;
    constexpr size_t TAIL_GUARD = 16; // longest token = 9B + safety; covers the 8-byte contents reads
    static_assert((CHUNK - 1) + MAX_TOKEN_BYTES <= 127, "vpermi2b successor indices must not wrap");

    alignas(64) static constexpr uint8_t index_bytes[CHUNK] = {
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
        44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};

    __m512i jump_lut[4];
    __m512i len_lut[4];
    for (int i = 0; i < 4; ++i)
    {
        jump_lut[i] = _mm512_load_si512(reinterpret_cast<const __m512i*>(T.jump + 64 * i));
        len_lut[i] = _mm512_load_si512(reinterpret_cast<const __m512i*>(T.len + 64 * i));
    }
    const __m512i index_v = _mm512_load_si512(reinterpret_cast<const __m512i*>(index_bytes));
    const __m512i chunk_v = _mm512_set1_epi8(static_cast<char>(CHUNK));
    const __m512i exit_table_v = _mm512_add_epi8(index_v, chunk_v);

    size_t bp = state ? state->pos : 0;
    __m512i entry_v = _mm512_setzero_si512();
    size_t n_out = 0;

    // `out` is full: record the resume point just past the captured token.
    const auto stop_after = [&](size_t next_byte)
    {
        if (state) *state = ScanState{next_byte, false};
        return n_out;
    };

    while (bp + CHUNK + TAIL_GUARD <= size)
    {
        const uint8_t* chunk = buf + bp;
        const __m512i bytes_v = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(chunk));
        const __mmask64 hi_mask = _mm512_movepi8_mask(bytes_v);

        __m512i g = _mm512_add_epi8(index_v, lookup256(jump_lut, bytes_v, hi_mask));
        g = _mm512_permutex2var_epi8(g, g, exit_table_v); // g^2
        g = _mm512_permutex2var_epi8(g, g, exit_table_v); // g^4
        g = _mm512_permutex2var_epi8(g, g, exit_table_v); // g^8
        g = _mm512_permutex2var_epi8(g, g, exit_table_v); // g^16
        g = _mm512_permutex2var_epi8(g, g, exit_table_v); // g^32
        g = _mm512_permutex2var_epi8(g, g, exit_table_v); // g^64

        // The entry is carried as a broadcast vector so that the loop-carried
        // chain is a single vpermb + vpsubb. The scalar copy only feeds the
        // (predicted) fast-path branch, off the critical path.
        const __m512i pos_v = _mm512_permutexvar_epi8(entry_v, g);
        size_t pos = lane0(pos_v);

        if (__builtin_expect(pos >= CHUNK, 1))
        {
            entry_v = _mm512_sub_epi8(pos_v, chunk_v);
            bp += CHUNK;
            continue;
        }

        // ============= Slow path =============
        // after_rare[r] is the next rare token (or exit) following the rare
        // token at r: g^64 sampled just past that token.
        const __m512i len_v = lookup256(len_lut, bytes_v, hi_mask);
        const __m512i after = _mm512_permutex2var_epi8(g, _mm512_add_epi8(index_v, len_v), exit_table_v);

        alignas(64) uint8_t after_rare[CHUNK];
        _mm512_store_si512(reinterpret_cast<__m512i*>(after_rare), after);

        // pos is always a rare token start here.
        while (pos < CHUNK)
        {
            const uint8_t b = chunk[pos];
            uint64_t contents;
            std::memcpy(&contents, chunk + pos, 8);
            out[n_out++] = QuickToken{contents, T.rare_type[b], bp + pos};
            if (n_out == out_cap) return stop_after(bp + pos + (T.info[b] & INFO_LEN));
            pos = after_rare[pos];
        }

        entry_v = _mm512_set1_epi8(static_cast<char>(pos - CHUNK));
        bp += CHUNK;
    }

    bp += lane0(entry_v);

    // Scalar tail.
    while (bp < size)
    {
        const uint8_t b = buf[bp];
        const uint8_t v = T.info[b];

        if (__builtin_expect(v & INFO_RARE, 0))
        {
            uint64_t contents = 0;
            const size_t avail = size - bp;
            const size_t to_copy = avail < 8 ? avail : 8;
            std::memcpy(&contents, buf + bp, to_copy);
            out[n_out++] = QuickToken{contents, T.rare_type[b], bp};
            if (n_out == out_cap) return stop_after(bp + (v & INFO_LEN));
        }

        bp += v & INFO_LEN;
    }

    if (state) *state = ScanState{bp, true};
    return n_out;
}

#    undef MI400_SCAN_TARGET

using ScanFn = size_t (*)(const uint8_t*, size_t, QuickToken*, size_t, ScanState*);

// Returns nullptr if the x86 CPU lacks AVX-512.
ScanFn select_scanner()
{
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512vbmi") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512f"))
        return &scan_mi400_avx512;
    return nullptr;
}

} // namespace
} // namespace mi400::quick_scan

namespace quick_scan
{
size_t scan_mi400(const uint8_t* buf, size_t size, QuickToken* __restrict__ out, size_t out_cap, ScanState* state)
{
    static const mi400::quick_scan::ScanFn fn = mi400::quick_scan::select_scanner();
    if (!fn) throw std::exception();

    return fn(buf, size, out, out_cap, state);
}
} // namespace quick_scan

#endif // MI400_RARE_SCAN_HAS_X86
