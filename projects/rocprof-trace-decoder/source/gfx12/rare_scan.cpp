#include "rare_scan.h"

#include <cstring>

#include "gfx10/token_types.h"
#include "gfx12parser.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define GFX12_RARE_SCAN_HAS_X86 1
#else
#define GFX12_RARE_SCAN_HAS_X86 0
#endif

namespace gfx12::rare_scan
{
namespace
{

struct Tables
{
    uint8_t info[256];
    uint8_t rare_type[256];
};

constexpr uint8_t INFO_LEN  = 0x1F;
constexpr uint8_t INFO_RARE = 0x20;
constexpr uint8_t INFO_EXT  = 0x40;
constexpr uint8_t INFO_NOP  = 0x80;

constexpr uint8_t MAX_TOKEN_NIBBLES = 18;
constexpr unsigned RARE_END = 5;

static_assert(RdnaType::UNKNOWN == 0 && RdnaType::EVENT == 1 && RdnaType::EVENT_SYNC == 2 &&
                  RdnaType::REG == 3 && RdnaType::REG_INIT == 4,
              "Rare-token cluster must occupy positions 0..4");

const Tables& tables()
{
    static const Tables t = [] {
        Tables out{};
        gfx12::TokenLookupTable lk{};
        for (int key = 0; key < 256; ++key)
        {
            const auto&    token_info = lk.lookup(static_cast<uint8_t>(key));
            const unsigned nibbles    = static_cast<unsigned>(token_info.length) >> 2;
            const unsigned type       = token_info.type;
            if ((token_info.length & 3u) != 0 || nibbles > MAX_TOKEN_NIBBLES) __builtin_trap();

            uint8_t v = static_cast<uint8_t>(nibbles & INFO_LEN);
            if (type < RARE_END)
            {
                v |= INFO_RARE;
                out.rare_type[key] = static_cast<uint8_t>(type);
            }
            if (type == RdnaType::WAVE_START_EXT) v |= INFO_EXT;
            if (type == RdnaType::NOP) v |= INFO_NOP;
            out.info[key] = v;
        }
        return out;
    }();
    return t;
}

uint8_t
load_key(const uint8_t* buf, size_t size, size_t nibble_pos)
{
    const size_t byte_pos = nibble_pos >> 1;
    if (byte_pos >= size) return 0;

    uint16_t window = buf[byte_pos];
    if (byte_pos + 1 < size) window |= static_cast<uint16_t>(buf[byte_pos + 1]) << 8;
    return static_cast<uint8_t>((window >> ((nibble_pos & 1u) * 4u)) & 0xFFu);
}

uint64_t
load_contents64(const uint8_t* buf, size_t size, size_t nibble_pos)
{
    const size_t byte_pos = nibble_pos >> 1;
    if (byte_pos >= size) return 0;

    uint64_t contents = 0;
    const size_t avail = size - byte_pos;
    const size_t to_copy = avail < sizeof(contents) ? avail : sizeof(contents);
    std::memcpy(&contents, buf + byte_pos, to_copy);

    if ((nibble_pos & 1u) == 0) return contents;

    const uint64_t carry = byte_pos + sizeof(contents) < size
                               ? static_cast<uint64_t>(buf[byte_pos + sizeof(contents)]) << 60
                               : 0;
    return (contents >> 4) | carry;
}

size_t
scan_gfx12_scalar(const uint8_t* buf, size_t size, RareToken* __restrict__ out, size_t out_cap)
{
    if (!buf || !out || out_cap == 0 || size == 0) return 0;

    const Tables& T = tables();
    const size_t  size_nibbles = size * 2;
    size_t        pos = 0;
    size_t        n_out = 0;
    bool          ext = false;

    while (pos < size_nibbles && n_out < out_cap)
    {
        const uint8_t key = load_key(buf, size, pos);

        if (ext && (key & 1u))
        {
            pos += 2;
            continue;
        }

        const uint8_t  v = T.info[key];
        const unsigned nibbles = v & INFO_LEN;

        if (__builtin_expect(v & INFO_RARE, 0))
        {
            out[n_out++] = RareToken{load_contents64(buf, size, pos), T.rare_type[key]};
            if (n_out >= out_cap) break;
        }

        if ((v & INFO_NOP) == 0) ext = (v & INFO_EXT) != 0;
        pos += nibbles ? nibbles : 1;
    }

    return n_out;
}

#if GFX12_RARE_SCAN_HAS_X86

__attribute__((target("avx2"), always_inline)) inline size_t
skip_zero_nibbles_avx2(const uint8_t* buf, size_t size, size_t pos)
{
    const size_t size_nibbles = size * 2;
    if (pos >= size_nibbles) return pos;

    if (pos & 1u)
    {
        const size_t byte_pos = pos >> 1;
        if (buf[byte_pos] & 0xF0u) return pos;
        ++pos;
    }

    size_t byte_pos = pos >> 1;

    const __m256i zero_v       = _mm256_setzero_si256();
    const __m256i nibble_mask  = _mm256_set1_epi8(0x0F);

    while (byte_pos + 32 <= size)
    {
        const __m256i bytes_v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(buf + byte_pos));
        const __m256i low_v   = _mm256_and_si256(bytes_v, nibble_mask);
        const __m256i high_v  = _mm256_and_si256(_mm256_srli_epi16(bytes_v, 4), nibble_mask);

        const uint32_t low_zero_mask =
            static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(low_v, zero_v)));
        const uint32_t high_zero_mask =
            static_cast<uint32_t>(_mm256_movemask_epi8(_mm256_cmpeq_epi8(high_v, zero_v)));
        const uint32_t low_nonzero_mask  = ~low_zero_mask;
        const uint32_t high_nonzero_mask = ~high_zero_mask;
        const uint32_t byte_nonzero_mask = low_nonzero_mask | high_nonzero_mask;

        if (byte_nonzero_mask == 0)
        {
            byte_pos += 32;
            continue;
        }

        const unsigned first = static_cast<unsigned>(__builtin_ctz(byte_nonzero_mask));
        pos = (byte_pos + first) * 2;
        if (((low_nonzero_mask >> first) & 1u) == 0) ++pos;
        return pos;
    }

    while (byte_pos < size)
    {
        const uint8_t b = buf[byte_pos];
        if (b & 0x0Fu) return byte_pos * 2;
        if (b & 0xF0u) return byte_pos * 2 + 1;
        ++byte_pos;
    }

    return size_nibbles;
}

__attribute__((target("avx2")))
size_t
scan_gfx12_avx2(const uint8_t* buf, size_t size, RareToken* __restrict__ out, size_t out_cap)
{
    if (!buf || !out || out_cap == 0 || size == 0) return 0;

    const Tables& T = tables();
    const size_t  size_nibbles = size * 2;
    size_t        pos = 0;
    size_t        n_out = 0;
    bool          ext = false;

    while (pos < size_nibbles && n_out < out_cap)
    {
        pos = skip_zero_nibbles_avx2(buf, size, pos);
        if (pos >= size_nibbles) break;

        const uint8_t key = load_key(buf, size, pos);

        if (ext && (key & 1u))
        {
            pos += 2;
            continue;
        }

        const uint8_t  v = T.info[key];
        const unsigned nibbles = v & INFO_LEN;

        if (__builtin_expect(v & INFO_RARE, 0))
        {
            out[n_out++] = RareToken{load_contents64(buf, size, pos), T.rare_type[key]};
            if (n_out >= out_cap) break;
        }

        if ((v & INFO_NOP) == 0) ext = (v & INFO_EXT) != 0;
        pos += nibbles ? nibbles : 1;
    }

    return n_out;
}

__attribute__((target("avx512vbmi,avx512bw,avx512f,avx2,bmi2")))
size_t
scan_gfx12_avx512vbmi(const uint8_t* buf, size_t size, RareToken* __restrict__ out, size_t out_cap)
{
    if (!buf || !out || out_cap == 0 || size == 0) return 0;

    const Tables& T = tables();

    const __m512i lut0 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(T.info));
    const __m512i lut1 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(T.info + 64));
    const __m512i lut2 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(T.info + 128));
    const __m512i lut3 = _mm512_loadu_si512(reinterpret_cast<const __m512i*>(T.info + 192));

    constexpr size_t  CHUNK_NIBBLES = 64;
    constexpr size_t  CHUNK_BYTES = CHUNK_NIBBLES / 2;
    constexpr size_t  TAIL_GUARD = 1;
    constexpr uint8_t SUCCESSOR_CAP = (CHUNK_NIBBLES - 1) + MAX_TOKEN_NIBBLES;
    static_assert(SUCCESSOR_CAP <= 127, "vpermi2b successor indices must not wrap");

    alignas(64) static constexpr uint8_t index_bytes[CHUNK_NIBBLES] = {
        0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
        16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
        48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};

    alignas(64) static constexpr uint8_t interleave_bytes[CHUNK_NIBBLES] = {
        0,  32, 1,  33, 2,  34, 3,  35, 4,  36, 5,  37, 6,  38, 7,  39,
        8,  40, 9,  41, 10, 42, 11, 43, 12, 44, 13, 45, 14, 46, 15, 47,
        16, 48, 17, 49, 18, 50, 19, 51, 20, 52, 21, 53, 22, 54, 23, 55,
        24, 56, 25, 57, 26, 58, 27, 59, 28, 60, 29, 61, 30, 62, 31, 63};

    const __m512i index_v       = _mm512_load_si512(index_bytes);
    const __m512i interleave_v  = _mm512_load_si512(interleave_bytes);
    const __m512i exit_table_v  = _mm512_add_epi8(index_v, _mm512_set1_epi8(64));
    const __m512i len_mask_v    = _mm512_set1_epi8(static_cast<char>(INFO_LEN));
    const __m512i len_one_v     = _mm512_set1_epi8(1);
    const __m512i succ_cap_v    = _mm512_set1_epi8(static_cast<char>(SUCCESSOR_CAP));
    const __m512i attention_v   = _mm512_set1_epi8(static_cast<char>(INFO_RARE | INFO_EXT));
    const __m256i low_nibble_y  = _mm256_set1_epi8(0x0F);
    const __m256i high_nibble_y = _mm256_set1_epi8(static_cast<char>(0xF0));
    const __m512i zero_v        = _mm512_setzero_si512();

    size_t bp = 0;
    size_t entry = 0;
    size_t n_out = 0;
    bool   ext = false;

    while (bp + CHUNK_BYTES + TAIL_GUARD <= size && n_out < out_cap)
    {
        const __m256i bytes_y = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(buf + bp));
        const __m256i next_y  = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(buf + bp + 1));

        const __m256i high = _mm256_and_si256(_mm256_srli_epi16(bytes_y, 4), low_nibble_y);
        const __m256i low_next = _mm256_and_si256(_mm256_slli_epi16(next_y, 4), high_nibble_y);
        const __m256i odd_lo = _mm256_or_si256(high, low_next);

        const __m256i even_lo = bytes_y;
        const __m512i key_halves = _mm512_inserti64x4(_mm512_castsi256_si512(even_lo), odd_lo, 1);
        const __m512i keys_v = _mm512_permutexvar_epi8(interleave_v, key_halves);

        const __m512i   lo_lookup = _mm512_permutex2var_epi8(lut0, keys_v, lut1);
        const __m512i   hi_lookup = _mm512_permutex2var_epi8(lut2, keys_v, lut3);
        const __mmask64 hi_mask   = _mm512_movepi8_mask(keys_v);
        const __m512i   info_v    = _mm512_mask_blend_epi8(hi_mask, lo_lookup, hi_lookup);

        __m512i len_v = _mm512_and_si512(info_v, len_mask_v);
        const __mmask64 zero_len_mask = _mm512_cmpeq_epi8_mask(len_v, zero_v);
        len_v = _mm512_mask_mov_epi8(len_v, zero_len_mask, len_one_v);

        __m512i succ = _mm512_add_epi8(index_v, len_v);
        succ = _mm512_min_epu8(succ, succ_cap_v);

        __m512i attention_path = _mm512_and_si512(info_v, attention_v);

        attention_path = _mm512_or_si512(attention_path, _mm512_permutex2var_epi8(attention_path, succ, zero_v));
        succ           = _mm512_permutex2var_epi8(succ, succ, exit_table_v);
        attention_path = _mm512_or_si512(attention_path, _mm512_permutex2var_epi8(attention_path, succ, zero_v));
        succ           = _mm512_permutex2var_epi8(succ, succ, exit_table_v);
        attention_path = _mm512_or_si512(attention_path, _mm512_permutex2var_epi8(attention_path, succ, zero_v));
        succ           = _mm512_permutex2var_epi8(succ, succ, exit_table_v);
        attention_path = _mm512_or_si512(attention_path, _mm512_permutex2var_epi8(attention_path, succ, zero_v));
        succ           = _mm512_permutex2var_epi8(succ, succ, exit_table_v);
        attention_path = _mm512_or_si512(attention_path, _mm512_permutex2var_epi8(attention_path, succ, zero_v));
        succ           = _mm512_permutex2var_epi8(succ, succ, exit_table_v);

        const __mmask64 attention_entry_mask = _mm512_test_epi8_mask(attention_path, attention_v);
        const __m128i   succ_0_15            = _mm512_castsi512_si128(succ);
        size_t          pos;
        bool            has_attention;
        if (__builtin_expect(entry == 0, 1))
        {
            pos           = static_cast<size_t>(_mm_cvtsi128_si32(succ_0_15) & 0xFFu);
            has_attention = (attention_entry_mask & 1u) != 0;
        }
        else
        {
            const __m128i succ_16_31 = _mm512_extracti32x4_epi32(succ, 1);
            const uint64_t succ_word =
                entry < 8    ? static_cast<uint64_t>(_mm_cvtsi128_si64(succ_0_15))
                : entry < 16 ? static_cast<uint64_t>(_mm_cvtsi128_si64(_mm_srli_si128(succ_0_15, 8)))
                             : static_cast<uint64_t>(_mm_cvtsi128_si64(succ_16_31));
            pos           = static_cast<size_t>((succ_word >> ((entry & 7u) * 8)) & 0xFFu);
            has_attention = ((attention_entry_mask >> entry) & 1u) != 0;
        }

        if (__builtin_expect(!ext && !has_attention, 1))
        {
            if (__builtin_expect(pos >= CHUNK_NIBBLES, 1))
            {
                entry = pos - CHUNK_NIBBLES;
                bp += CHUNK_BYTES;
                continue;
            }

            attention_path = _mm512_or_si512(attention_path, _mm512_permutex2var_epi8(attention_path, succ, zero_v));
            succ           = _mm512_permutex2var_epi8(succ, succ, exit_table_v);

            const __mmask64 attention64_entry_mask = _mm512_test_epi8_mask(attention_path, attention_v);
            const __m128i   succ64_0_15            = _mm512_castsi512_si128(succ);
            bool            has_attention64;
            if (__builtin_expect(entry == 0, 1))
            {
                pos             = static_cast<size_t>(_mm_cvtsi128_si32(succ64_0_15) & 0xFFu);
                has_attention64 = (attention64_entry_mask & 1u) != 0;
            }
            else
            {
                const __m128i succ64_16_31 = _mm512_extracti32x4_epi32(succ, 1);
                const uint64_t succ64_word =
                    entry < 8    ? static_cast<uint64_t>(_mm_cvtsi128_si64(succ64_0_15))
                    : entry < 16 ? static_cast<uint64_t>(_mm_cvtsi128_si64(_mm_srli_si128(succ64_0_15, 8)))
                                 : static_cast<uint64_t>(_mm_cvtsi128_si64(succ64_16_31));
                pos             = static_cast<size_t>((succ64_word >> ((entry & 7u) * 8)) & 0xFFu);
                has_attention64 = ((attention64_entry_mask >> entry) & 1u) != 0;
            }

            if (!has_attention64)
            {
                entry = pos - CHUNK_NIBBLES;
                bp += CHUNK_BYTES;
                continue;
            }
        }

        alignas(64) uint8_t info_buf[CHUNK_NIBBLES];
        _mm512_store_si512(reinterpret_cast<__m512i*>(info_buf), info_v);

        pos = entry;
        while (pos < CHUNK_NIBBLES)
        {
            const size_t global_nibble = bp * 2 + pos;
            const uint8_t key = load_key(buf, size, global_nibble);

            if (ext && (key & 1u))
            {
                pos += 2;
                continue;
            }

            const uint8_t  v = info_buf[pos];
            const unsigned nibbles = v & INFO_LEN;

            if (__builtin_expect(v & INFO_RARE, 0))
            {
                out[n_out++] = RareToken{load_contents64(buf, size, global_nibble), T.rare_type[key]};
                if (n_out >= out_cap) goto done;
            }

            if ((v & INFO_NOP) == 0) ext = (v & INFO_EXT) != 0;
            pos += nibbles ? nibbles : 1;
        }

        entry = pos - CHUNK_NIBBLES;
        bp += CHUNK_BYTES;
    }

done:
    size_t tail_pos = bp * 2 + entry;
    const size_t size_nibbles = size * 2;
    while (tail_pos < size_nibbles && n_out < out_cap)
    {
        const uint8_t key = load_key(buf, size, tail_pos);

        if (ext && (key & 1u))
        {
            tail_pos += 2;
            continue;
        }

        const uint8_t  v = T.info[key];
        const unsigned nibbles = v & INFO_LEN;

        if (__builtin_expect(v & INFO_RARE, 0))
        {
            out[n_out++] = RareToken{load_contents64(buf, size, tail_pos), T.rare_type[key]};
            if (n_out >= out_cap) break;
        }

        if ((v & INFO_NOP) == 0) ext = (v & INFO_EXT) != 0;
        tail_pos += nibbles ? nibbles : 1;
    }

    return n_out;
}

#endif // GFX12_RARE_SCAN_HAS_X86

using ScanFn = size_t (*)(const uint8_t*, size_t, RareToken*, size_t);

ScanFn select_scanner()
{
#if GFX12_RARE_SCAN_HAS_X86
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512vbmi") && __builtin_cpu_supports("avx512bw") &&
        __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx2"))
        return &scan_gfx12_avx512vbmi;
    if (__builtin_cpu_supports("avx2")) return &scan_gfx12_avx2;
#endif
    return &scan_gfx12_scalar;
}

} // namespace

size_t
scan_gfx12(const uint8_t* buf, size_t size, RareToken* __restrict__ out, size_t out_cap)
{
    static const ScanFn fn = select_scanner();
    return fn(buf, size, out, out_cap);
}

} // namespace gfx12::rare_scan
