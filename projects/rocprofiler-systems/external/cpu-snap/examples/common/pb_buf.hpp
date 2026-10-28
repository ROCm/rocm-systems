#pragma once

/**
 * @file pb_buf.hpp
 * @brief Minimal protobuf wire encoder used by perfetto_proto_writer.
 *
 * Encodes only what the bundled demo writer needs: varint, tag +
 * payload, length-delimited string +
 * submessage, and a raw bytes accessor for the surrounding std::ofstream
 * writer.
 *
 * Header-only. Not part of the cpu_snap library install surface.
 */

#include <cstdint>
#include <string>

namespace cs_examples::perfetto
{

// NOLINTBEGIN(readability-identifier-naming): codebase uses k_*-prefixed
//   lower_case for global constants; the .clang-tidy GlobalConstantCase
//   rule defaults to UPPER_CASE but the entire project ships with k_*.
namespace detail
{
/** Protobuf varint continuation-bit mask: bytes with bit-7 set carry more. */
inline constexpr std::uint8_t k_pb_varint_more   = 0x80;
/** Protobuf varint payload mask: low 7 bits per byte. */
inline constexpr std::uint8_t k_pb_varint_chunk  = 0x7f;
/** Protobuf varint shift per byte. */
inline constexpr std::uint32_t k_pb_varint_shift = 7;
/** Wire types: varint (0), length-delimited (2). */
inline constexpr std::uint8_t k_pb_wire_varint   = 0;
inline constexpr std::uint8_t k_pb_wire_lendelim = 2;
/** Protobuf field-number shift in tag encoding (3 bits reserved for wire type). */
inline constexpr std::uint32_t k_pb_tag_shift = 3;
}  // namespace detail
// NOLINTEND(readability-identifier-naming)

namespace pb
{

/**
 * @brief Append-only protobuf byte buffer.
 *
 * Backed by a single std::string; all writes are append. Each call
 * either emits a raw varint or a tag + payload pair. Submessages are
 * encoded by buffering the inner message in a separate buf and
 * embedding it length-prefixed via tag_submsg.
 */
class buf
{
public:
    void varint(std::uint64_t value)
    {
        while(value > detail::k_pb_varint_chunk)
        {
            m_data.push_back(static_cast<char>((value & detail::k_pb_varint_chunk) |
                                               detail::k_pb_varint_more));
            value >>= detail::k_pb_varint_shift;
        }
        m_data.push_back(static_cast<char>(value & detail::k_pb_varint_chunk));
    }

    void tag(std::uint32_t field, std::uint8_t wire)
    {
        varint((static_cast<std::uint64_t>(field) << detail::k_pb_tag_shift) | wire);
    }

    void tag_varint(std::uint32_t field, std::uint64_t value)
    {
        tag(field, detail::k_pb_wire_varint);
        varint(value);
    }
    void tag_int32(std::uint32_t field, std::int32_t value)
    {
        // negative int32 is encoded as 10-byte varint
        tag(field, detail::k_pb_wire_varint);
        if(value >= 0)
        {
            varint(static_cast<std::uint64_t>(value));
        }
        else
        {
            varint(static_cast<std::uint64_t>(static_cast<std::int64_t>(value)));
        }
    }

    void tag_string(std::uint32_t field, const std::string& str)
    {
        tag(field, detail::k_pb_wire_lendelim);
        varint(str.size());
        m_data.append(str);
    }

    void tag_submsg(std::uint32_t field, const buf& inner)
    {
        tag(field, detail::k_pb_wire_lendelim);
        varint(inner.size());
        m_data.append(inner.m_data);
    }

    [[nodiscard]] const std::string& bytes() const { return m_data; }
    [[nodiscard]] std::size_t        size() const { return m_data.size(); }

private:
    std::string m_data;
};

}  // namespace pb

}  // namespace cs_examples::perfetto
