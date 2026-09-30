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

#include <array>
#include <cstring>

#include "gfx9token.h" // gfx9::token_len_dict, sqtt_token_type_t

// SIMD-only scanner — no scalar fallback. The AVX paths use GCC function
// multiversioning (__attribute__((target("…")))) which MSVC does not
// implement, so MSVC and non-x86 targets compile to a stub that returns
// quick_scan::SCAN_NOT_IMPLEMENTED.
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#    include <immintrin.h>
#    define GFX9_QUICK_SCAN_HAS_X86 1
#else
#    define GFX9_QUICK_SCAN_HAS_X86 0
#endif

namespace gfx9::quick_scan
{
namespace
{
using ::quick_scan::QuickToken;

// gfx9 token format (see gfx9token.h:181-199, gfx9token.cpp:29):
//   - The low nibble of the first byte is the type id (0..15).
//   - Length (in bytes) is determined entirely by the nibble — see
//     token_len_dict (lengths in bits, divided by 8 here).
//   - There is no extension-byte state machine (mi400's WAVE_START_EXT
//     has no analog in gfx9).
//   - patch_time() lookahead in gfx9token.cpp only reorders TOKEN_TIME
//     events; none of the rare types we capture are TIME, so output
//     ordering is preserved by a strict left-to-right walk.

constexpr uint16_t RARE_MASK = (1 << TOKEN_REG_CS) | (1 << TOKEN_REG) | (1 << TOKEN_REG_CS_PRIV) |
                               (1 << TOKEN_EVENT_CS) |
                               (1 << TOKEN_EVENT); // | (1 << TOKEN_WAVE_START) | (1 << TOKEN_WAVE_END);

// Per-nibble combined info byte:
//   bits 0-3 : token byte length (max 8 fits in 4 bits)
//   bit  4   : is_rare (set when nibble ∈ {2, 5, 7, 8, 15})
//   bits 5-7 : unused
constexpr uint8_t INFO_LEN = 0x0F;
constexpr uint8_t INFO_RARE = 0x10;

struct Tables
{
    // 16-entry length LUT for vpshufb (broadcast). bytes-per-token by nibble.
    alignas(16) uint8_t length_lut[16];
    // 16-entry combined info LUT (length + rare bit).
    alignas(16) uint8_t info_lut[16];
};

// Local copy of gfx9::token_len_dict (defined in gfx9token.cpp:29 but not
// extern-declared in the header). Bit-lengths per nibble. Keep in sync —
// the comment in gfx9token.cpp is the source of truth.
constexpr int LEN_BITS[16] = {16, 64, 64, 32, 16, 48, 16, 16, 16, 16, 16, 64, 48, 32, 64, 48};

const Tables& tables()
{
    thread_local const Tables t = []
    {
        Tables out{};
        for (int n = 0; n < 16; ++n)
        {
            const unsigned bytes = static_cast<unsigned>(LEN_BITS[n]) / 8u;
            out.length_lut[n] = static_cast<uint8_t>(bytes & INFO_LEN);
            uint8_t v = static_cast<uint8_t>(bytes & INFO_LEN);
            if ((RARE_MASK >> n) & 1u) v |= INFO_RARE;
            out.info_lut[n] = v;
        }
        return out;
    }();
    return t;
}

// Sanity check that our rare set still matches the enum values.
static_assert(
    TOKEN_REG == 2 && TOKEN_REG_CS == 5 && TOKEN_EVENT == 7 && TOKEN_EVENT_CS == 8 && TOKEN_REG_CS_PRIV == 15,
    "gfx9 rare-token cluster nibble ids changed — update RARE_MASK"
);

#if GFX9_QUICK_SCAN_HAS_X86

// AVX-512_VBMI path. Per 64-byte chunk:
//   1. Decode per-byte nibble info with vpshufb.
//   2. Build succ_1[i] = min(i + len[i], 71) for every byte position.
//   3. Use vpermi2b doubling to compute both succ_32 and whether any rare
//      token lies on the actual token-start orbit from each possible entry.
//   4. If the current entry has no reachable rare token, update entry from
//      succ_32 without a scalar walk. Otherwise scalar-walk only that chunk
//      to emit captures.
//
// The old chunk-wide rare test was conservative over all 64 bytes, so random
// payload nibbles frequently forced the scalar path. This path tests only
// reachable token starts.
__attribute__((target("avx512vbmi,avx512bw,avx512f,bmi2"))) size_t scan_gfx9_avx512vbmi(
    const uint8_t* buf, size_t size, QuickToken* __restrict__ out, size_t out_cap
)
{
    if (!buf || !out || out_cap == 0 || size == 0) return 0;

    const Tables& T = tables();

    const __m512i info_lut_v = _mm512_broadcast_i32x4(_mm_load_si128(reinterpret_cast<const __m128i*>(T.info_lut)));
    const __m512i nibble_mask_v = _mm512_set1_epi8(0x0F);
    const __m512i len_mask_v = _mm512_set1_epi8(static_cast<char>(INFO_LEN));
    const __m512i rare_mask_v = _mm512_set1_epi8(static_cast<char>(INFO_RARE));

    constexpr size_t CHUNK = 64;
    constexpr size_t TAIL_GUARD = 16;
    constexpr uint8_t SUCCESSOR_CAP = (CHUNK - 1) + 8;
    static_assert(SUCCESSOR_CAP <= 127, "vpermi2b successor indices must not wrap");

    alignas(64) thread_local constexpr uint8_t index_bytes[CHUNK] = {
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
        44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};

    const __m512i index_v = _mm512_load_si512(index_bytes);
    const __m512i exit_table_v = _mm512_add_epi8(index_v, _mm512_set1_epi8(64));
    const __m512i len_two_v = _mm512_set1_epi8(2);
    const __m512i succ_cap_v = _mm512_set1_epi8(static_cast<char>(SUCCESSOR_CAP));
    const __m512i zero_v = _mm512_setzero_si512();

    size_t bp = 0;
    size_t entry = 0;
    size_t n_out = 0;

    while (bp + CHUNK + TAIL_GUARD <= size && n_out < out_cap)
    {
        const __m512i bytes_v = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(buf + bp));
        const __m512i nibbles_v = _mm512_and_si512(bytes_v, nibble_mask_v);
        const __m512i info_v = _mm512_shuffle_epi8(info_lut_v, nibbles_v);

        __m512i len_v = _mm512_and_si512(info_v, len_mask_v);
        const __mmask64 zero_len_mask = _mm512_cmpeq_epi8_mask(len_v, zero_v);
        len_v = _mm512_mask_mov_epi8(len_v, zero_len_mask, len_two_v);

        __m512i succ = _mm512_add_epi8(index_v, len_v);
        succ = _mm512_min_epu8(succ, succ_cap_v);

        __m512i rare_path = _mm512_and_si512(info_v, rare_mask_v);

        rare_path = _mm512_or_si512(rare_path, _mm512_permutex2var_epi8(rare_path, succ, zero_v));
        succ = _mm512_permutex2var_epi8(succ, succ, exit_table_v); // succ_2
        rare_path = _mm512_or_si512(rare_path, _mm512_permutex2var_epi8(rare_path, succ, zero_v));
        succ = _mm512_permutex2var_epi8(succ, succ, exit_table_v); // succ_4
        rare_path = _mm512_or_si512(rare_path, _mm512_permutex2var_epi8(rare_path, succ, zero_v));
        succ = _mm512_permutex2var_epi8(succ, succ, exit_table_v); // succ_8
        rare_path = _mm512_or_si512(rare_path, _mm512_permutex2var_epi8(rare_path, succ, zero_v));
        succ = _mm512_permutex2var_epi8(succ, succ, exit_table_v); // succ_16
        rare_path = _mm512_or_si512(rare_path, _mm512_permutex2var_epi8(rare_path, succ, zero_v));
        succ = _mm512_permutex2var_epi8(succ, succ, exit_table_v); // succ_32

        const __mmask64 rare_entry_mask = _mm512_test_epi8_mask(rare_path, rare_mask_v);
        const uint64_t succ_low = static_cast<uint64_t>(_mm_cvtsi128_si64(_mm512_castsi512_si128(succ)));
        size_t pos = static_cast<size_t>((succ_low >> (entry * 8)) & 0xFFu);

        if (__builtin_expect(((rare_entry_mask >> entry) & 1u) == 0, 1))
        {
            entry = pos - CHUNK;
            bp += CHUNK;
            continue;
        }

        alignas(64) uint8_t info_buf[CHUNK];
        _mm512_store_si512(reinterpret_cast<__m512i*>(info_buf), info_v);

        pos = entry;
        while (pos < CHUNK)
        {
            const uint8_t v = info_buf[pos];
            const unsigned bytes = v & INFO_LEN;

            if (__builtin_expect(v & INFO_RARE, 0))
            {
                uint64_t contents;
                std::memcpy(&contents, buf + bp + pos, 8);
                if (__builtin_expect(n_out < out_cap, 1))
                    out[n_out++] = QuickToken{contents, static_cast<uint64_t>(buf[bp + pos] & 0x0F), bp + pos};
            }

            pos += bytes ? bytes : 2;
        }

        entry = pos - CHUNK;
        bp += CHUNK;
    }

    bp += entry;

    while (bp < size && n_out < out_cap)
    {
        const uint8_t nibble = buf[bp] & 0x0F;
        const uint8_t v = T.info_lut[nibble];
        const unsigned bytes = v & INFO_LEN;

        if (__builtin_expect(v & INFO_RARE, 0))
        {
            uint64_t contents = 0;
            const size_t avail = size - bp;
            const size_t to_copy = avail < 8 ? avail : 8;
            std::memcpy(&contents, buf + bp, to_copy);
            out[n_out++] = QuickToken{contents, nibble, bp};
            if (n_out >= out_cap) break;
        }

        bp += bytes ? bytes : 2;
    }

    return n_out;
}

// AVX2 fallback for CPUs without AVX-512. Per 16-byte chunk:
//   1. Load 16 bytes.
//   2. Decode low-nibble token info with vpshufb against a duplicated 16B LUT.
//   3. Build a lane-local successor function and compose it to succ_8.
//   4. Propagate rare bits only along token-start successor orbits.
//   5. If the current entry cannot reach a rare token, advance without any
//      scalar walk. Otherwise scalar-walk the chunk to emit captures.
//
// A 16-byte chunk is small enough for SSSE3/AVX2 vpshufb's 128-bit lane-local
// semantics: max token length is 8 bytes and min length is 2 bytes, so after at
// most 8 token starts the path exits the chunk. Exit successors are kept as
// 16..23 values for entry update and masked to zero only for shuffle lookups.
__attribute__((target("avx2"))) size_t scan_gfx9_avx2(
    const uint8_t* buf, size_t size, QuickToken* __restrict__ out, size_t out_cap
)
{
    if (!buf || !out || out_cap == 0 || size == 0) return 0;

    const Tables& T = tables();

    const __m128i info_lut_v = _mm_load_si128(reinterpret_cast<const __m128i*>(T.info_lut));
    const __m128i nibble_mask_v = _mm_set1_epi8(0x0F);
    const __m128i len_mask_v = _mm_set1_epi8(static_cast<char>(INFO_LEN));
    const __m128i rare_mask_v = _mm_set1_epi8(static_cast<char>(INFO_RARE));
    const __m128i fifteen_v = _mm_set1_epi8(15);
    const __m128i high_bit_v = _mm_set1_epi8(static_cast<char>(0x80));
    const __m128i zero_v = _mm_setzero_si128();

    constexpr size_t CHUNK = 16;
    constexpr size_t TAIL_GUARD = 16;

    alignas(16) thread_local constexpr uint8_t index_bytes[CHUNK] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    const __m128i index_v = _mm_load_si128(reinterpret_cast<const __m128i*>(index_bytes));

    size_t bp = 0;
    size_t entry = 0;
    size_t n_out = 0;

    while (bp + CHUNK + TAIL_GUARD <= size && n_out < out_cap)
    {
        const __m128i bytes_v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(buf + bp));
        const __m128i nibbles_v = _mm_and_si128(bytes_v, nibble_mask_v);
        const __m128i info_v = _mm_shuffle_epi8(info_lut_v, nibbles_v);

        const __m128i len_v = _mm_and_si128(info_v, len_mask_v);
        __m128i succ = _mm_add_epi8(index_v, len_v);
        __m128i rare_path = _mm_and_si128(info_v, rare_mask_v);

#    define GFX9_AVX2_COMPOSE_SUCCESSOR()                                                                              \
        do {                                                                                                           \
            const __m128i exit_mask = _mm_cmpgt_epi8(succ, fifteen_v);                                                 \
            const __m128i ctrl = _mm_or_si128(succ, _mm_and_si128(exit_mask, high_bit_v));                             \
            rare_path = _mm_or_si128(rare_path, _mm_shuffle_epi8(rare_path, ctrl));                                    \
            const __m128i next_succ = _mm_shuffle_epi8(succ, ctrl);                                                    \
            succ = _mm_blendv_epi8(next_succ, succ, exit_mask);                                                        \
        }                                                                                                              \
        while (0)

        GFX9_AVX2_COMPOSE_SUCCESSOR();
        GFX9_AVX2_COMPOSE_SUCCESSOR();
        GFX9_AVX2_COMPOSE_SUCCESSOR();

#    undef GFX9_AVX2_COMPOSE_SUCCESSOR

        const uint32_t rare_entry_mask =
            static_cast<uint32_t>(~_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_and_si128(rare_path, rare_mask_v), zero_v))) &
            0xFFFFu;
        const uint64_t succ_low = static_cast<uint64_t>(_mm_cvtsi128_si64(succ));
        size_t pos = static_cast<size_t>((succ_low >> (entry * 8)) & 0xFFu);

        if (__builtin_expect(((rare_entry_mask >> entry) & 1u) == 0, 1))
        {
            entry = pos - CHUNK;
            bp += CHUNK;
            continue;
        }

        alignas(16) uint8_t info_buf[CHUNK];
        _mm_store_si128(reinterpret_cast<__m128i*>(info_buf), info_v);

        pos = entry;
        while (pos < CHUNK)
        {
            const uint8_t v = info_buf[pos];
            const unsigned bytes = v & INFO_LEN;

            if (__builtin_expect(v & INFO_RARE, 0))
            {
                uint64_t contents;
                std::memcpy(&contents, buf + bp + pos, 8);
                out[n_out++] = QuickToken{contents, static_cast<uint64_t>(buf[bp + pos] & 0x0F), bp + pos};
                if (n_out >= out_cap) goto done;
            }

            pos += bytes ? bytes : 2;
        }

        entry = pos - CHUNK;
        bp += CHUNK;
    }

    bp += entry;

done:
    while (bp < size && n_out < out_cap)
    {
        const uint8_t nibble = buf[bp] & 0x0F;
        const uint8_t v = T.info_lut[nibble];
        const unsigned bytes = v & INFO_LEN;

        if (__builtin_expect(v & INFO_RARE, 0))
        {
            uint64_t contents = 0;
            const size_t avail = size - bp;
            const size_t to_copy = avail < 8 ? avail : 8;
            std::memcpy(&contents, buf + bp, to_copy);
            out[n_out++] = QuickToken{contents, nibble, bp};
            if (n_out >= out_cap) break;
        }

        bp += bytes ? bytes : 2;
    }

    return n_out;
}

#endif // GFX9_QUICK_SCAN_HAS_X86

using ScanFn = size_t (*)(const uint8_t*, size_t, QuickToken*, size_t);

// Returns nullptr when no SIMD path is available — either because the TU
// was compiled for a non-x86 target / a non-GCC compiler, or because the
// running x86 CPU is below the AVX2 bar.
ScanFn select_scanner()
{
#if GFX9_QUICK_SCAN_HAS_X86
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512vbmi") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512f"))
        return &scan_gfx9_avx512vbmi;
    if (__builtin_cpu_supports("avx2")) return &scan_gfx9_avx2;
#endif
    return nullptr;
}

} // namespace

#if GFX9_QUICK_SCAN_HAS_X86
size_t scan_gfx9_avx2_for_testing(const uint8_t* buf, size_t size, QuickToken* __restrict__ out, size_t out_cap)
{
    return scan_gfx9_avx2(buf, size, out, out_cap);
}
#endif

} // namespace gfx9::quick_scan

namespace quick_scan
{
#if GFX9_QUICK_SCAN_HAS_X86
size_t scan_gfx9(const uint8_t* buf, size_t size, QuickToken* __restrict__ out, size_t out_cap)
{
    thread_local const gfx9::quick_scan::ScanFn fn = gfx9::quick_scan::select_scanner();
    if (!fn) throw std::exception();

    return fn(buf, size, out, out_cap);
}
#endif
} // namespace quick_scan
