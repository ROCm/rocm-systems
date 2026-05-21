// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace rocprofsys
{
namespace core
{
// Trace.packets framing wire tag: field 1, wire type 2 (length-delimited).
inline constexpr std::uint8_t TRACE_PACKETS_TAG = 0x0A;

// TracePacket.trusted_packet_sequence_id wire tag: field 10, wire type 0 (varint).
inline constexpr std::uint8_t TRUSTED_SEQ_ID_TAG = 0x50;

// Number of bytes a Trace.packets field tag occupies in the encoded stream.
inline constexpr std::size_t TRACE_PACKETS_TAG_BYTES = 1;

// Protobuf varint encoding: low 7 bits carry payload, high bit signals a
// continuation byte. A 64-bit value occupies at most 10 bytes.
inline constexpr std::uint8_t  VARINT_PAYLOAD_MASK     = 0x7F;
inline constexpr std::uint8_t  VARINT_CONTINUATION_BIT = 0x80;
inline constexpr std::uint32_t VARINT_SHIFT_BITS       = 7;
inline constexpr std::uint32_t VARINT_MAX_SHIFT_BITS   = 64;
inline constexpr std::size_t   VARINT_MAX_BYTES        = 10;

// Protobuf wire-type encoding: low 3 bits of the field tag select the wire
// type. Fixed64 and Fixed32 payloads occupy a constant byte width.
inline constexpr std::uint32_t WIRE_TYPE_MASK          = 0x7;
inline constexpr std::size_t   WIRE_TYPE_FIXED64_BYTES = 8;
inline constexpr std::size_t   WIRE_TYPE_FIXED32_BYTES = 4;

// Decodes a protobuf varint from data[pos..size). Advances pos past the
// last byte read. Returns true on success, false on truncated input or
// >64-bit overflow.
inline bool
read_varint(const char* data, std::size_t size, std::size_t& pos,
            std::uint64_t& out) noexcept
{
    out                 = 0;
    std::uint32_t shift = 0;
    while(pos < size)
    {
        auto b = static_cast<std::uint8_t>(data[pos++]);
        out |= static_cast<std::uint64_t>(b & VARINT_PAYLOAD_MASK) << shift;
        if((b & VARINT_CONTINUATION_BIT) == 0) return true;
        shift += VARINT_SHIFT_BITS;
        if(shift >= VARINT_MAX_SHIFT_BITS) return false;
    }
    return false;
}

// Appends `v` to `dst` in protobuf varint encoding.
inline void
append_varint(std::vector<char>& dst, std::uint64_t v)
{
    while(v >= VARINT_CONTINUATION_BIT)
    {
        dst.push_back(
            static_cast<char>((v & VARINT_PAYLOAD_MASK) | VARINT_CONTINUATION_BIT));
        v >>= VARINT_SHIFT_BITS;
    }
    dst.push_back(static_cast<char>(v));
}

// Walks one TracePacket payload, copies every field verbatim EXCEPT
// trusted_packet_sequence_id (field 10), then appends a fresh field 10
// with `new_seq_id`. Wraps the rewritten payload in the Trace.packets
// length-delimited frame and appends it to `dst`.
//
// Returns false on malformed input (truncated varint, length overflow,
// or unknown wire type). On false return, `dst` may have been partially
// modified — the caller should drop the remainder of the source's bytes
// rather than risk emitting garbage.
inline bool
rewrite_trace_packet(std::vector<char>& dst, const char* packet, std::size_t size,
                     std::uint32_t new_seq_id)
{
    std::vector<char> rewritten;
    rewritten.reserve(size + 5);

    std::size_t pos = 0;
    while(pos < size)
    {
        std::size_t   tag_start = pos;
        std::uint64_t tag       = 0;
        if(!read_varint(packet, size, pos, tag)) return false;
        const std::uint32_t wire = tag & WIRE_TYPE_MASK;

        std::size_t value_end = pos;
        switch(wire)
        {
            case 0:  // varint
            {
                std::uint64_t v = 0;
                if(!read_varint(packet, size, pos, v)) return false;
                value_end = pos;
                break;
            }
            case 2:  // length-delimited
            {
                std::uint64_t len = 0;
                if(!read_varint(packet, size, pos, len)) return false;
                if(len > size - pos) return false;
                pos += static_cast<std::size_t>(len);
                value_end = pos;
                break;
            }
            case 1:  // fixed64 — bounds-check before increment so pos cannot wrap
                if(WIRE_TYPE_FIXED64_BYTES > size - pos) return false;
                pos += WIRE_TYPE_FIXED64_BYTES;
                value_end = pos;
                break;
            case 5:  // fixed32 — same bounds-check discipline as fixed64
                if(WIRE_TYPE_FIXED32_BYTES > size - pos) return false;
                pos += WIRE_TYPE_FIXED32_BYTES;
                value_end = pos;
                break;
            default: return false;  // group/unknown wire types
        }
        if(value_end > size) return false;

        if(tag == TRUSTED_SEQ_ID_TAG) continue;  // re-emitted below
        rewritten.insert(rewritten.end(), packet + tag_start, packet + value_end);
    }

    rewritten.push_back(static_cast<char>(TRUSTED_SEQ_ID_TAG));
    append_varint(rewritten, new_seq_id);

    dst.push_back(static_cast<char>(TRACE_PACKETS_TAG));
    append_varint(dst, rewritten.size());
    dst.insert(dst.end(), rewritten.begin(), rewritten.end());
    return true;
}
}  // namespace core
}  // namespace rocprofsys
