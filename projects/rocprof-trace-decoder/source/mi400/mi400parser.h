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
#include <cstring>
#include "gfx12/gfx12parser.h"
#include "mi400token.h"

namespace mi400
{

class TokenLookupTable : public gfx12::TokenLookupTable
{
public:
    TokenLookupTable();

    int64_t getTime(const token_info_t& info, uint64_t contents, int64_t cur_time, bool& PL, int64_t& rt)
    {
        if (info.type == RdnaType::TIMESTAMP)
        {
            gfx12::timestamp_type stamp{.raw = contents};
            PL |= bool(stamp.pl && !stamp.rt);
            if (stamp.rt == 0) return stamp.time + cur_time;

            if (stamp.pl == 0) rt = stamp.time;
            return cur_time;
        }
        else if (info.type == RdnaType::TIME) { cur_time += 1; }
        return getDelta(info, contents) + cur_time;
    };

private:
    // MI400 omits the +4 cycle adjustment that gfx10/11/12 apply to TIME tokens; the
    // +1 above already covers it. Hides gfx12::TokenLookupTable::getDelta via name lookup.
    static int64_t getDelta(const token_info_t& info, uint64_t contents)
    {
        uint64_t mask = (1ull << (info.time_end - info.time_begin)) - 1;
        return ((contents >> info.time_begin) & mask);
    };
};

class TokenGenerator : public NaviTokenGenerator
{
public:
    TokenGenerator(const uint8_t* _buffer, size_t size, int64_t _globaltime, int64_t _base_time);
    gfx10::Token next() final;

    inline uint64_t getBuffer400() { return buffer[byte_ptr]; };

    inline void advanceByte(uint64_t value)
    {
        current = (current >> 8) | (value << 56);
        byte_ptr++;
        bits_toread -= 8;
    }

    inline void readOne_unsafe400()
    {
        while (bits_toread > 0) advanceByte(getBuffer400());
    }

    inline void readOne_safe400()
    {
        while (bits_toread > 0) advanceByte(byte_ptr < BUFFER_SIZE ? getBuffer400() : 0);
    }

    void update_fifo(int wave);
    int get_valu_inst_mi400();

    // Fast type-only scanner used by tokens-mode (bench / IterateTokens_internal).
    //
    // What we skip vs the full next() path:
    //   - globaltime / TIME / TIMESTAMP / addRealtime accounting
    //   - VALU_INST FIFO update + INST inst-range FIFO update
    //   - per-token Token{} value + INST_PC trailing-byte read
    //   - virtual dispatch (this is a template member — visitor inlines)
    //
    // What stays identical to next() so the histogram matches:
    //   - NOP / TIME / TIMESTAMP are silently consumed (no on_token call)
    //   - WAVE_START_EXT extension byte is silently consumed
    //   - bIsExt state machine
    //
    // The "bit trick" — load 8 bytes once via unaligned uint64_t, decode up to 4
    // tokens by shifting the window. mi400 tokens are byte-aligned and only the
    // low byte selects the lookup entry, so the inner loop is a tight chain of
    // independent loads/lookups that the OoO core can pipeline. INST_PC (9 bytes)
    // and TIMESTAMP (8 bytes) hit the break-and-reload path; everything else
    // (the bulk: INST=3, VALU_INST=1, IMM_ONE=2, WAVE_END=3, MISC=3) packs 3-8
    // tokens into one window.
    // Rare-token capture: REG / REG_INIT / EVENT / EVENT_SYNC are uncommon
    // (typically << 1% of stream) but downstream consumers want their full
    // 64-bit contents alongside the type. The scan loop bitmask-checks the
    // type against RARE_MASK; a hit pushes onto `rare_out` (preallocated by
    // caller, e.g. .reserve(32768)). All four types are <= 8 bytes long, so
    // `lo` always holds the complete token at the moment of capture.
    struct QuickToken
    {
        uint64_t contents;
        uint32_t type;
    };

    // The rare-token cluster lives at positions 0..4 in RdnaType
    // (gfx10/token_types.h): UNKNOWN, EVENT, EVENT_SYNC, REG, REG_INIT.
    // The check then reduces to `unsigned(type) < RARE_END` — a single
    // unsigned compare, with no subtract.
    static constexpr unsigned RARE_END = 5;
    static_assert(
        RdnaType::UNKNOWN == 0 && RdnaType::EVENT == 1 && RdnaType::EVENT_SYNC == 2 && RdnaType::REG == 3 &&
            RdnaType::REG_INIT == 4,
        "Rare-token cluster must occupy positions 0..4 — update if enum reordered"
    );

    // Public entrypoints: zero-overhead overload set. CaptureRare is a
    // compile-time switch so the rare-token branch is dead-code-eliminated
    // when the caller doesn't ask for capture.
    template <class F> void scan_types(F&& on_token)
    {
        std::vector<QuickToken>* unused = nullptr;
        scan_types_impl<false>(std::forward<F>(on_token), unused);
    }

    template <class F> void scan_types(F&& on_token, std::vector<QuickToken>& rare_out)
    {
        scan_types_impl<true>(std::forward<F>(on_token), &rare_out);
    }

private:
    template <bool CaptureRare, class F> void scan_types_impl(F&& on_token, std::vector<QuickToken>* rare_out)
    {
        // Need a 16-byte tail guard so the unaligned 8-byte load + the longest
        // token (INST_PC = 9 bytes) can never run off the end of buffer.
        constexpr size_t TAIL_GUARD = 16;

        while (byte_ptr + TAIL_GUARD < BUFFER_SIZE)
        {
            uint64_t window;
            std::memcpy(&window, buffer + byte_ptr, sizeof(window));
            size_t consumed = 0;

            // Up to 4 tokens per loaded window. Total bytes consumed may exceed
            // 8 by up to (max_token_len - 1); when it does the inner loop breaks
            // and the next outer iteration reloads from the new byte_ptr.
            for (int i = 0; i < 4; ++i)
            {
                uint8_t b = static_cast<uint8_t>(window);

                // bIsExt && low-bit-set means the previous WAVE_START_EXT has an
                // extension byte at this position. Skip 1 byte; bIsExt stays true
                // (matches the `continue` in next()).
                if (bIsExt && (b & 1u))
                {
                    window >>= 8;
                    consumed += 1;
                    if (consumed >= 8) break;
                    continue;
                }

                const auto& info = lookupbits.lookup(b);
                RdnaType type = static_cast<RdnaType>(info.type);

                if (type == RdnaType::NOP)
                {
                    window >>= 8;
                    consumed += 1;
                    if (consumed >= 8) break;
                    continue;
                }

                bIsExt = (type == RdnaType::WAVE_START_EXT);

                // Rare-token capture (compile-time elided when CaptureRare=false).
                // Single unsigned compare against the cluster boundary.
                //
                // We re-fetch 8 fresh bytes from the buffer rather than reusing
                // `window`. After prior inner iterations, `window` has been
                // right-shifted by `consumed*8` bits, zero-padding the high
                // `consumed` bytes. REG / REG_INIT are 8 bytes wide, so every
                // bit is a valid field; the shifted view would corrupt the
                // upper payload. EVENT (3B) / EVENT_SYNC (4B) only care about
                // the low bytes, but the cold-path cost of a single extra
                // 8-byte load is negligible (~50 rare tokens per ~5M).
                // TAIL_GUARD = 16 ensures buffer[byte_ptr + consumed + 8) is
                // in-bounds (consumed <= 7 here).
                if constexpr (CaptureRare)
                {
                    if (static_cast<unsigned>(type) < RARE_END)
                    {
                        uint64_t contents;
                        std::memcpy(&contents, buffer + byte_ptr + consumed, sizeof(contents));
                        rare_out->push_back(QuickToken{contents, static_cast<uint32_t>(type)});
                    }
                }

                // TIME and TIMESTAMP are consumed silently in next() — preserve
                // that so the bench histogram totals match the slow path.
                if (type != RdnaType::TIMESTAMP && type != RdnaType::TIME) on_token(static_cast<int>(type));

                unsigned bytes = static_cast<unsigned>(info.length) >> 3;
                consumed += bytes;
                if (consumed >= 8) break; // window exhausted (or 9-byte INST_PC)
                window >>= bytes * 8;     // bytes <= 7 here, no UB
            }

            byte_ptr += consumed;
        }

        // Tail: hand off to the legacy next() path for the last bytes where the
        // window approach can't safely peek 16 bytes ahead. Sync the bit-stream
        // cursor inherited from NaviTokenGenerator so readOne_*400() refills
        // `current` from the current byte position.
        bit_ptr = byte_ptr * 8;
        bits_toread = 64;
        current = 0;

        try
        {
            while (nextValid())
            {
                gfx10::Token tok = next();
                // next() returns the {0,0,TIMESTAMP} sentinel only when both
                // unsafe and safe loops fail to produce a token; nextValid()
                // already filters most of that, but TIMESTAMP is also silently
                // consumed in the fast path so suppressing it here is consistent.
                if (tok.type == RdnaType::TIMESTAMP || tok.type == RdnaType::TIME || tok.type == RdnaType::NOP)
                    continue;

                if constexpr (CaptureRare)
                {
                    if (static_cast<unsigned>(tok.type) < RARE_END)
                        rare_out->push_back(QuickToken{tok.contents, static_cast<uint32_t>(tok.type)});
                }
                on_token(static_cast<int>(tok.type));
            }
        }
        catch (const std::exception&)
        {}
    }

protected:
    std::array<int, 6> FIFO = {-1, -1, -1, -1, -1, -1};

private:
    size_t byte_ptr = 0;
    bool bIsExt = false;
    TokenLookupTable lookupbits{};
};

} // namespace mi400
