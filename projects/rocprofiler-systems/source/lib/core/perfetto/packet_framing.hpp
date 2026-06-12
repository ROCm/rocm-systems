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

// Decodes a protobuf varint from data[offset..size). Advances offset past the
// last byte read. Returns true on success, false on truncated input or
// >64-bit overflow.
inline bool
read_varint(const char* data, std::size_t size, std::size_t& offset,
            std::uint64_t& decoded_value) noexcept
{
    decoded_value       = 0;
    std::uint32_t shift = 0;
    while(offset < size)
    {
        // Reject before OR-ing payload bits so an attacker-controlled 10th
        // byte cannot smuggle bits into the high end of `decoded_value`.
        if(shift >= VARINT_MAX_SHIFT_BITS) return false;
        auto byte = static_cast<std::uint8_t>(data[offset++]);
        if(shift == VARINT_MAX_SHIFT_BITS - 1 && (byte & VARINT_PAYLOAD_MASK) > 1)
            return false;
        decoded_value |= static_cast<std::uint64_t>(byte & VARINT_PAYLOAD_MASK) << shift;
        if((byte & VARINT_CONTINUATION_BIT) == 0) return true;
        shift += VARINT_SHIFT_BITS;
    }
    return false;
}

// Appends `value` to `output` in protobuf varint encoding.
inline void
append_varint(std::vector<char>& output, std::uint64_t value)
{
    while(value >= VARINT_CONTINUATION_BIT)
    {
        output.push_back(
            static_cast<char>((value & VARINT_PAYLOAD_MASK) | VARINT_CONTINUATION_BIT));
        value >>= VARINT_SHIFT_BITS;
    }
    output.push_back(static_cast<char>(value));
}

// Walks one TracePacket payload, copies every field verbatim EXCEPT
// trusted_packet_sequence_id (field 10), then appends a fresh field 10
// carrying `seq_id_offset + original_seq_id` (or just `seq_id_offset`
// if the input had no seq_id field). Wraps the rewritten payload in
// the Trace.packets length-delimited frame and appends it to `output`.
//
// Offset (not replace) is the correct semantics because Perfetto's
// interned-data namespace (event_categories[].iid, event_names[].iid,
// debug_annotation_names[].iid) is keyed PER trusted_packet_sequence_id.
// Collapsing distinct input seq_ids into one output value would merge
// independent iid namespaces and silently misresolve later definitions.
// Adding a per-source base offset preserves the original seq_id
// grouping while shifting the whole range into a disjoint window of
// the merged namespace.
//
// Returns false on malformed input (truncated varint, length overflow,
// or unknown wire type). On false return, `output` may have been partially
// modified — the caller should drop the remainder of the source's bytes
// rather than risk emitting garbage.
inline bool
rewrite_trace_packet(std::vector<char>& output, const char* packet, std::size_t size,
                     std::uint32_t seq_id_offset)
{
    std::vector<char> rewritten_packet;
    rewritten_packet.reserve(size + 5);

    std::uint64_t original_seq_id = 0;

    std::size_t offset = 0;
    while(offset < size)
    {
        std::size_t   field_start = offset;
        std::uint64_t field_tag   = 0;
        if(!read_varint(packet, size, offset, field_tag)) return false;
        const std::uint32_t wire_type = field_tag & WIRE_TYPE_MASK;

        std::size_t field_end = offset;
        switch(wire_type)
        {
            case 0:  // varint
            {
                std::uint64_t field_value = 0;
                if(!read_varint(packet, size, offset, field_value)) return false;
                field_end = offset;
                if(field_tag == TRUSTED_SEQ_ID_TAG) original_seq_id = field_value;
                break;
            }
            case 2:  // length-delimited
            {
                std::uint64_t field_length = 0;
                if(!read_varint(packet, size, offset, field_length)) return false;
                if(field_length > size - offset) return false;
                offset += static_cast<std::size_t>(field_length);
                field_end = offset;
                break;
            }
            case 1:  // fixed64 -- bounds-check before increment so offset cannot wrap
                if(WIRE_TYPE_FIXED64_BYTES > size - offset) return false;
                offset += WIRE_TYPE_FIXED64_BYTES;
                field_end = offset;
                break;
            case 5:  // fixed32 -- same bounds-check discipline as fixed64
                if(WIRE_TYPE_FIXED32_BYTES > size - offset) return false;
                offset += WIRE_TYPE_FIXED32_BYTES;
                field_end = offset;
                break;
            default: return false;  // group/unknown wire types
        }
        if(field_end > size) return false;

        if(field_tag == TRUSTED_SEQ_ID_TAG) continue;  // re-emitted below
        rewritten_packet.insert(rewritten_packet.end(), packet + field_start,
                                packet + field_end);
    }

    rewritten_packet.push_back(static_cast<char>(TRUSTED_SEQ_ID_TAG));
    append_varint(rewritten_packet, original_seq_id + seq_id_offset);

    output.push_back(static_cast<char>(TRACE_PACKETS_TAG));
    append_varint(output, rewritten_packet.size());
    output.insert(output.end(), rewritten_packet.begin(), rewritten_packet.end());
    return true;
}
}  // namespace core
}  // namespace rocprofsys
