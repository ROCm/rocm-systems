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

#include "quick_scan.h"

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

// Single combined byte per first-byte slot, derived once from the canonical
// mi400::TokenLookupTable at TU init. Any future AddEncoding edit in
// mi400token.cpp / gfx12 / gfx11 / gfx10 propagates automatically.
//
// Per-byte info encoding (one cache load per inner iter, vs three in the
// previous design):
//   bits 0-3 : token byte length (max 9 = 0b1001 fits in 4 bits)
//   bit  4   : is_rare       (first byte of REG/REG_INIT/EVENT/EVENT_SYNC/UNKNOWN)
//   bit  5   : is_ext_start  (first byte of WAVE_START_EXT)
//   bits 6-7 : unused
struct Tables
{
    uint8_t info[256];
    uint8_t rare_type[256]; // RdnaType id (1..4); valid only when info[b] & 0x10
};

constexpr uint8_t INFO_RARE = 0x10;
constexpr uint8_t INFO_EXT = 0x20;
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
            uint8_t v = static_cast<uint8_t>(len & INFO_LEN);
            if (type < mi400::TokenGenerator::RARE_END)
            {
                v |= INFO_RARE;
                out.rare_type[b] = static_cast<uint8_t>(type);
            }
            if (type == RdnaType::WAVE_START_EXT) v |= INFO_EXT;
            out.info[b] = v;
        }
        return out;
    }();
    return t;
}

// AVX-512_VBMI path. Per 64-byte chunk:
//   1. Load 64 bytes.
//   2. Look up info[byte] for each lane via two `vpermi2b` over the 256-byte
//      info table, blended on bit 7 of the byte. ~6c latency.
//   3. Build a normal-state successor function and use vpermi2b doubling to
//      compute both the next entry and whether rare/ext attention lies on the
//      actual token-start orbit for each possible entry.
//   4. Fast path: when ext is false and the current entry has no reachable
//      attention, advance without any scalar walk.
//   5. Slow path: store info_vec to a 64B stack buffer and scalar-walk with
//      full rare/ext handling.
//
// State carried across chunks: `entry` (byte offset within next chunk where
// the next token starts; 0..max_token_len-1) and `ext` (WAVE_START_EXT
// in-progress flag). Same semantics as the scalar.

__attribute__((target("avx512vbmi,avx512bw,avx512f,bmi2"))) size_t scan_mi400_avx512(
    const uint8_t* buf, size_t size, TokenGenerator::QuickToken* __restrict__ out, size_t out_cap
)
{
    if (!buf || !out || out_cap == 0 || size == 0) return 0;

    const Tables& T = tables();

    // 256-byte info LUT held in 4 zmm registers.
    const __m512i lut0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(T.info));
    const __m512i lut1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(T.info + 64));
    const __m512i lut2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(T.info + 128));
    const __m512i lut3 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(T.info + 192));

    constexpr size_t CHUNK = 64;
    constexpr size_t TAIL_GUARD = 16; // longest token = 9B + safety
    constexpr uint8_t SUCCESSOR_CAP = (CHUNK - 1) + MAX_TOKEN_BYTES;
    static_assert(SUCCESSOR_CAP <= 127, "vpermi2b successor indices must not wrap");

    alignas(64) static constexpr uint8_t index_bytes[CHUNK] = {
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
        44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};

    size_t bp = 0;
    size_t entry = 0;
    bool ext = false;
    size_t n_out = 0;

    // Splat for the fast-skip test: any reachable token start has rare-or-ext set?
    const __m512i rare_or_ext_v = _mm512_set1_epi8(static_cast<char>(INFO_RARE | INFO_EXT));
    const __m512i len_mask_v = _mm512_set1_epi8(static_cast<char>(INFO_LEN));
    const __m512i index_v = _mm512_load_si512(index_bytes);
    const __m512i exit_table_v = _mm512_add_epi8(index_v, _mm512_set1_epi8(64));
    const __m512i len_one_v = _mm512_set1_epi8(1);
    const __m512i succ_cap_v = _mm512_set1_epi8(static_cast<char>(SUCCESSOR_CAP));
    const __m512i zero_v = _mm512_setzero_si512();

    while (bp + CHUNK + TAIL_GUARD <= size && n_out < out_cap)
    {
        const __m512i bytes_v = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(buf + bp));

        // 256-byte LUT lookup: vpermi2b uses low 7 bits of each index lane.
        // First permute covers info[0..127], second covers info[128..255];
        // blend on bit 7 of the byte (extracted via movepi8_mask).
        const __m512i lo_lookup = _mm512_permutex2var_epi8(lut0, bytes_v, lut1);
        const __m512i hi_lookup = _mm512_permutex2var_epi8(lut2, bytes_v, lut3);
        const __mmask64 hi_mask = _mm512_movepi8_mask(bytes_v);
        const __m512i info_v = _mm512_mask_blend_epi8(hi_mask, lo_lookup, hi_lookup);

        // Build succ_1[i] = min(i + max(len[i], 1), 72), then compose the
        // successor function 6 times. In parallel, propagate rare/ext bits
        // along the same orbit so payload-byte aliases no longer force the
        // slow path unless they are reachable token starts.
        __m512i len_v = _mm512_and_si512(info_v, len_mask_v);
        const __mmask64 zero_len_mask = _mm512_cmpeq_epi8_mask(len_v, zero_v);
        len_v = _mm512_mask_mov_epi8(len_v, zero_len_mask, len_one_v);

        __m512i succ = _mm512_add_epi8(index_v, len_v);
        succ = _mm512_min_epu8(succ, succ_cap_v);

        __m512i attention_path = _mm512_and_si512(info_v, rare_or_ext_v);

        attention_path = _mm512_or_si512(attention_path, _mm512_permutex2var_epi8(attention_path, succ, zero_v));
        succ = _mm512_permutex2var_epi8(succ, succ, exit_table_v); // succ_2
        attention_path = _mm512_or_si512(attention_path, _mm512_permutex2var_epi8(attention_path, succ, zero_v));
        succ = _mm512_permutex2var_epi8(succ, succ, exit_table_v); // succ_4
        attention_path = _mm512_or_si512(attention_path, _mm512_permutex2var_epi8(attention_path, succ, zero_v));
        succ = _mm512_permutex2var_epi8(succ, succ, exit_table_v); // succ_8
        attention_path = _mm512_or_si512(attention_path, _mm512_permutex2var_epi8(attention_path, succ, zero_v));
        succ = _mm512_permutex2var_epi8(succ, succ, exit_table_v); // succ_16
        attention_path = _mm512_or_si512(attention_path, _mm512_permutex2var_epi8(attention_path, succ, zero_v));
        succ = _mm512_permutex2var_epi8(succ, succ, exit_table_v); // succ_32
        attention_path = _mm512_or_si512(attention_path, _mm512_permutex2var_epi8(attention_path, succ, zero_v));
        succ = _mm512_permutex2var_epi8(succ, succ, exit_table_v); // succ_64

        const __mmask64 attention_entry_mask = _mm512_test_epi8_mask(attention_path, rare_or_ext_v);
        const __m128i succ_lo_v = _mm512_castsi512_si128(succ);
        const uint64_t succ_lo = static_cast<uint64_t>(_mm_cvtsi128_si64(succ_lo_v));
        const uint64_t succ_hi = static_cast<uint64_t>(_mm_cvtsi128_si64(_mm_srli_si128(succ_lo_v, 8)));
        const uint64_t succ_word = entry < 8 ? succ_lo : succ_hi;
        size_t pos = static_cast<size_t>((succ_word >> ((entry & 7u) * 8)) & 0xFFu);

        if (__builtin_expect(!ext && ((attention_entry_mask >> entry) & 1u) == 0, 1))
        {
            entry = pos - CHUNK;
            bp += CHUNK;
            continue;
        }

        // ============= Slow path =============
        // Either ext is currently active (in-flight WSE) OR this chunk has a
        // reachable rare/ext-start token. Walk byte-by-byte with full state
        // tracking and capture.
        {
            alignas(64) uint8_t info_buf[CHUNK];
            _mm512_store_si512(reinterpret_cast<__m512i*>(info_buf), info_v);

            pos = entry;
            while (pos < CHUNK)
            {
                const uint8_t b = buf[bp + pos];

                if (__builtin_expect(ext && (b & 1u), 0))
                {
                    pos += 1;
                    continue;
                }

                const uint8_t v = info_buf[pos];
                const unsigned bytes = v & INFO_LEN;

                if (__builtin_expect(v & INFO_RARE, 0))
                {
                    uint64_t contents;
                    std::memcpy(&contents, buf + bp + pos, 8);
                    out[n_out++] = TokenGenerator::QuickToken{contents, T.rare_type[b]};
                    if (n_out >= out_cap) goto done;
                }

                ext = (v & INFO_EXT) != 0;
                pos += bytes ? bytes : 1;
            }
        }

        entry = pos - CHUNK;
        bp += CHUNK;
    }

    bp += entry;

done:
    // Scalar tail: identical to scan_mi400_scalar's tail.
    while (bp < size && n_out < out_cap)
    {
        uint8_t b = buf[bp];

        if (ext && (b & 1u))
        {
            bp += 1;
            continue;
        }

        const uint8_t v = T.info[b];
        const unsigned bytes = v & INFO_LEN;

        if (__builtin_expect(v & INFO_RARE, 0))
        {
            uint64_t contents = 0;
            const size_t avail = size - bp;
            const size_t to_copy = avail < 8 ? avail : 8;
            std::memcpy(&contents, buf + bp, to_copy);
            out[n_out++] = TokenGenerator::QuickToken{contents, T.rare_type[b]};
            if (n_out >= out_cap) break;
        }

        ext = (v & INFO_EXT) != 0;
        bp += bytes ? bytes : 1;
    }

    return n_out;
}

using ScanFn = size_t (*)(const uint8_t*, size_t, TokenGenerator::QuickToken*, size_t);

// Returns nullptr if the x86 CPU lacks AVX-512.
ScanFn select_scanner()
{
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512vbmi") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512f"))
        return &scan_mi400_avx512;
    return nullptr;
}

} // namespace

size_t scan_mi400(const uint8_t* buf, size_t size, TokenGenerator::QuickToken* __restrict__ out, size_t out_cap)
{
    static const ScanFn fn = select_scanner();
    if (!fn) throw std::exception();

    return fn(buf, size, out, out_cap);
}

} // namespace mi400::quick_scan

#endif // MI400_RARE_SCAN_HAS_X86
