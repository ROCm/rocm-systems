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

#include "../quick_scan_export.hpp"

#include <cstring>

#include "gfx10/token_types.h"
#include "gfx12parser.h"

// SIMD-only scanner — no scalar fallback. The AVX-512 path uses GCC
// function multiversioning (__attribute__((target("…")))) which MSVC does
// not implement, so MSVC and non-x86 targets compile to a stub that
// returns quick_scan::SCAN_NOT_IMPLEMENTED. x86 CPUs without AVX-512 also
// resolve to the stub at runtime.
#if (defined(__GNUC__) || defined(__clang__)) && (defined(__x86_64__) || defined(__i386__))
#    include <immintrin.h>
#    define GFX12_RARE_SCAN_HAS_X86 1
#else
#    define GFX12_RARE_SCAN_HAS_X86 0
#endif

#if GFX12_RARE_SCAN_HAS_X86

namespace gfx12::quick_scan
{
using ::quick_scan::QuickToken;
using ::quick_scan::ScanState;

namespace
{

// `info` drives the scalar walks. `jump` and `len` are the vector LUTs:
// jump[key] is 0 for rare keys and the token length in nibbles otherwise, so
// that index + jump[key] is the jump function g described below; len[key] is
// the token length for every key. Keys >= 128 alias key & 0x7F except 0xE1,
// which is patched separately.
//
// WAVE_START_EXT is an ordinary fixed-length (40-bit) token here; it is not
// followed by extension nibbles.
struct Tables
{
    uint8_t info[256];
    uint8_t rare_type[256];
    alignas(64) uint8_t jump[128];
    alignas(64) uint8_t len[128];
};

constexpr uint8_t INFO_LEN = 0x1F;
constexpr uint8_t INFO_RARE = 0x20;

constexpr uint8_t MAX_TOKEN_NIBBLES = 18;
constexpr unsigned RARE_END = 5;

static_assert(
    RdnaType::UNKNOWN == 0 && RdnaType::EVENT == 1 && RdnaType::EVENT_SYNC == 2 && RdnaType::REG == 3 &&
        RdnaType::REG_INIT == 4,
    "Rare-token cluster must occupy positions 0..4"
);

uint8_t jump_of(uint8_t info) { return (info & INFO_RARE) ? 0 : (info & INFO_LEN); }

const Tables& tables()
{
    static const Tables t = []
    {
        Tables out{};
        gfx12::TokenLookupTable lk{};
        for (int key = 0; key < 256; ++key)
        {
            const auto& token_info = lk.lookup(static_cast<uint8_t>(key));
            const unsigned nibbles = static_cast<unsigned>(token_info.length) >> 2;
            const unsigned type = token_info.type;
            if ((token_info.length & 3u) != 0 || nibbles > MAX_TOKEN_NIBBLES) __builtin_trap();

            uint8_t v = static_cast<uint8_t>((nibbles ? nibbles : 1) & INFO_LEN);
            if (type < RARE_END)
            {
                v |= INFO_RARE;
                out.rare_type[key] = static_cast<uint8_t>(type);
            }
            out.info[key] = v;
        }
        for (int key = 128; key < 256; ++key)
        {
            if (key != 0xE1 && out.info[key] != out.info[key & 0x7F]) __builtin_trap();
        }
        for (int key = 0; key < 128; ++key)
        {
            out.jump[key] = jump_of(out.info[key]);
            out.len[key] = out.info[key] & INFO_LEN;
        }
        return out;
    }();
    return t;
}

uint8_t load_key(const uint8_t* buf, size_t size, size_t nibble_pos)
{
    const size_t byte_pos = nibble_pos >> 1;
    if (byte_pos >= size) return 0;

    uint16_t window = buf[byte_pos];
    if (byte_pos + 1 < size) window |= static_cast<uint16_t>(buf[byte_pos + 1]) << 8;
    return static_cast<uint8_t>((window >> ((nibble_pos & 1u) * 4u)) & 0xFFu);
}

uint64_t load_contents64(const uint8_t* buf, size_t size, size_t nibble_pos)
{
    const size_t byte_pos = nibble_pos >> 1;
    if (byte_pos >= size) return 0;

    uint64_t contents = 0;
    const size_t avail = size - byte_pos;
    const size_t to_copy = avail < sizeof(contents) ? avail : sizeof(contents);
    std::memcpy(&contents, buf + byte_pos, to_copy);

    if ((nibble_pos & 1u) == 0) return contents;

    const uint64_t carry =
        byte_pos + sizeof(contents) < size ? static_cast<uint64_t>(buf[byte_pos + sizeof(contents)]) << 60 : 0;
    return (contents >> 4) | carry;
}

#    define GFX12_SCAN_TARGET __attribute__((target("avx512vbmi,avx512bw,avx512f,avx2,bmi2")))

// Key (8 bits starting at in-chunk nibble `pos`). The caller guarantees two
// readable bytes at chunk + (pos >> 1).
inline uint8_t load_key_unchecked(const uint8_t* chunk, unsigned pos)
{
    uint16_t window;
    std::memcpy(&window, chunk + (pos >> 1), sizeof(window));
    return static_cast<uint8_t>(window >> ((pos & 1u) * 4u));
}

// 64-bit contents starting at in-chunk nibble `pos`. The caller guarantees
// nine readable bytes at chunk + (pos >> 1).
inline uint64_t load_contents64_unchecked(const uint8_t* chunk, unsigned pos)
{
    const uint8_t* p = chunk + (pos >> 1);
    uint64_t lo;
    std::memcpy(&lo, p, sizeof(lo));
    const uint64_t odd = 0 - static_cast<uint64_t>(pos & 1u);
    return (lo >> (pos & 1u) * 4u) | ((static_cast<uint64_t>(p[sizeof(lo)]) << 60) & odd);
}

GFX12_SCAN_TARGET inline unsigned lane0(__m512i v)
{ return static_cast<uint8_t>(_mm_cvtsi128_si32(_mm512_castsi512_si128(v))); }

// AVX-512_VBMI path. Per 64-nibble (32-byte) chunk:
//   1. Build the 64 overlapping 8-bit keys (one per nibble) with vpermb +
//      vpmultishiftqb.
//   2. Look up jump[key] and form g[i] = i if a rare token starts at nibble i,
//      else i + len(i). Exit lanes (>= 64) are fixed points through
//      exit_table_v.
//   3. Square g six times. Tokens are at least one nibble long, so g^64[i] is
//      the first rare token on the token-start orbit from i, or
//      64 + the next chunk's entry if there is none.
//   4. Fast path: g^64[entry] leaves the chunk.
//   5. Slow path: capture the rare token, then jump straight to the next one
//      through g^64 sampled just past it.
//
// With `state`, the scan starts at state->pos and, when `out` fills, stops
// right after the last captured token so a later call can resume.
GFX12_SCAN_TARGET size_t scan_gfx12_avx512vbmi(
    const uint8_t* buf, size_t size, QuickToken* __restrict__ out, size_t out_cap, ScanState* state
)
{
    if (!buf || !out || out_cap == 0 || size == 0)
    {
        if (state && (!buf || size == 0)) state->done = true;
        return 0;
    }

    const Tables& T = tables();

    constexpr size_t CHUNK_NIBBLES = 64;
    constexpr size_t CHUNK_BYTES = CHUNK_NIBBLES / 2;
    // Full-zmm load per chunk; also covers the slow path's unchecked reads
    // (key: 2 bytes, contents: 9 bytes, from any in-chunk nibble).
    constexpr size_t LOAD_BYTES = 64;
    static_assert((CHUNK_NIBBLES - 1) + MAX_TOKEN_NIBBLES <= 127, "vpermi2b successor indices must not wrap");
    static_assert(CHUNK_BYTES + 9 <= LOAD_BYTES, "slow-path reads must stay within the loaded window");

    alignas(64) static constexpr uint8_t index_bytes[CHUNK_NIBBLES] = {
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
        22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
        44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};
    // Qword q of the gathered vector holds bytes 4q..4q+7; vpmultishiftqb then
    // extracts the 8 bits at bit offset 4j of that qword into byte j, which is
    // the key at nibble 8q + j.
    alignas(64) static constexpr uint8_t gather_bytes[CHUNK_NIBBLES] = {
        0,  1,  2,  3,  4,  5,  6,  7,  4,  5,  6,  7,  8,  9,  10, 11, 8,  9,  10, 11, 12, 13,
        14, 15, 12, 13, 14, 15, 16, 17, 18, 19, 16, 17, 18, 19, 20, 21, 22, 23, 20, 21, 22, 23,
        24, 25, 26, 27, 24, 25, 26, 27, 28, 29, 30, 31, 28, 29, 30, 31, 32, 33, 34, 35};
    alignas(64) static constexpr uint8_t shift_bytes[CHUNK_NIBBLES] = {
        0,  4,  8,  12, 16, 20, 24, 28, 0,  4,  8,  12, 16, 20, 24, 28, 0,  4,  8,  12, 16, 20,
        24, 28, 0,  4,  8,  12, 16, 20, 24, 28, 0,  4,  8,  12, 16, 20, 24, 28, 0,  4,  8,  12,
        16, 20, 24, 28, 0,  4,  8,  12, 16, 20, 24, 28, 0,  4,  8,  12, 16, 20, 24, 28};

    const __m512i jump_lo_v = _mm512_load_si512(reinterpret_cast<const __m512i*>(T.jump));
    const __m512i jump_hi_v = _mm512_load_si512(reinterpret_cast<const __m512i*>(T.jump + 64));
    const __m512i event_sync_jump_v = _mm512_set1_epi8(static_cast<char>(jump_of(T.info[0xE1])));
    const __m512i event_sync_key_v = _mm512_set1_epi8(static_cast<char>(0xE1));
    const __m512i len_lo_v = _mm512_load_si512(reinterpret_cast<const __m512i*>(T.len));
    const __m512i len_hi_v = _mm512_load_si512(reinterpret_cast<const __m512i*>(T.len + 64));
    const __m512i event_sync_len_v = _mm512_set1_epi8(static_cast<char>(T.info[0xE1] & INFO_LEN));
    const __m512i index_v = _mm512_load_si512(reinterpret_cast<const __m512i*>(index_bytes));
    const __m512i chunk_nibbles_v = _mm512_set1_epi8(static_cast<char>(CHUNK_NIBBLES));
    const __m512i exit_table_v = _mm512_add_epi8(index_v, chunk_nibbles_v);
    const __m512i gather_v = _mm512_load_si512(reinterpret_cast<const __m512i*>(gather_bytes));
    const __m512i shift_v = _mm512_load_si512(reinterpret_cast<const __m512i*>(shift_bytes));

    const size_t start = state ? state->pos : 0;
    size_t bp = start >> 1;
    __m512i entry_v = _mm512_set1_epi8(static_cast<char>(start & 1u));
    size_t n_out = 0;

    // `out` is full: record the resume point just past the captured token.
    const auto stop_after = [&](size_t next_nibble)
    {
        if (state) *state = ScanState{next_nibble, false};
        return n_out;
    };

    while (bp + LOAD_BYTES <= size)
    {
        const uint8_t* chunk = buf + bp;
        const __m512i bytes_v = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(chunk));
        const __m512i keys_v = _mm512_multishift_epi64_epi8(shift_v, _mm512_permutexvar_epi8(gather_v, bytes_v));

        __m512i jump_v = _mm512_permutex2var_epi8(jump_lo_v, keys_v, jump_hi_v);
        const __mmask64 event_sync_mask = _mm512_cmpeq_epi8_mask(keys_v, event_sync_key_v);
        jump_v = _mm512_mask_mov_epi8(jump_v, event_sync_mask, event_sync_jump_v);

        __m512i g = _mm512_add_epi8(index_v, jump_v);
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
        unsigned pos = lane0(pos_v);

        if (__builtin_expect(pos >= CHUNK_NIBBLES, 1))
        {
            entry_v = _mm512_sub_epi8(pos_v, chunk_nibbles_v);
            bp += CHUNK_BYTES;
            continue;
        }

        // after_rare[r] is the next rare token (or exit) following the rare
        // token at r: g^64 sampled just past that token.
        __m512i len_v = _mm512_permutex2var_epi8(len_lo_v, keys_v, len_hi_v);
        len_v = _mm512_mask_mov_epi8(len_v, event_sync_mask, event_sync_len_v);
        const __m512i after = _mm512_permutex2var_epi8(g, _mm512_add_epi8(index_v, len_v), exit_table_v);

        alignas(64) uint8_t key_buf[CHUNK_NIBBLES];
        alignas(64) uint8_t after_rare[CHUNK_NIBBLES];
        _mm512_store_si512(reinterpret_cast<__m512i*>(key_buf), keys_v);
        _mm512_store_si512(reinterpret_cast<__m512i*>(after_rare), after);

        // pos is always a rare token start here.
        while (pos < CHUNK_NIBBLES)
        {
            const uint8_t key = key_buf[pos];
            out[n_out++] = QuickToken{load_contents64_unchecked(chunk, pos), T.rare_type[key], bp + (pos >> 1)};
            if (n_out == out_cap) return stop_after(bp * 2 + pos + (T.info[key] & INFO_LEN));
            pos = after_rare[pos];
        }

        entry_v = _mm512_set1_epi8(static_cast<char>(pos - CHUNK_NIBBLES));
        bp += CHUNK_BYTES;
    }

    size_t tail_pos = bp * 2 + lane0(entry_v);
    const size_t size_nibbles = size * 2;
    while (tail_pos < size_nibbles)
    {
        const uint8_t key = load_key(buf, size, tail_pos);
        const uint8_t v = T.info[key];
        const unsigned nibbles = v & INFO_LEN;

        if (__builtin_expect(v & INFO_RARE, 0))
        {
            out[n_out++] = QuickToken{load_contents64(buf, size, tail_pos), T.rare_type[key], tail_pos >> 1};
            if (n_out == out_cap) return stop_after(tail_pos + nibbles);
        }

        tail_pos += nibbles;
    }

    if (state) *state = ScanState{tail_pos, true};
    return n_out;
}

#    undef GFX12_SCAN_TARGET

using ScanFn = size_t (*)(const uint8_t*, size_t, QuickToken*, size_t, ScanState*);

// Returns nullptr if the x86 CPU lacks AVX-512.
ScanFn select_scanner()
{
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512vbmi") && __builtin_cpu_supports("avx512bw") &&
        __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx2"))
        return &scan_gfx12_avx512vbmi;
    return nullptr;
}

} // namespace
} // namespace gfx12::quick_scan

#endif // GFX12_RARE_SCAN_HAS_X86

namespace quick_scan
{
#if GFX12_RARE_SCAN_HAS_X86
size_t scan_gfx12(const uint8_t* buf, size_t size, QuickToken* __restrict__ out, size_t out_cap, ScanState* state)
{
    static const gfx12::quick_scan::ScanFn fn = gfx12::quick_scan::select_scanner();
    if (!fn) throw std::exception();

    return fn(buf, size, out, out_cap, state);
}
#endif
}; // namespace quick_scan
