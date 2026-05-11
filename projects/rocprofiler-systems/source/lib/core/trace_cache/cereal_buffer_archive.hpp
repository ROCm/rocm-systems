// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <timemory/tpls/cereal/cereal/archives/binary.hpp>

#include <cstdint>
#include <cstring>
#include <istream>
#include <ostream>
#include <streambuf>

namespace rocprofsys
{
namespace trace_cache
{

// Streambuf that writes into a preallocated raw byte buffer.  Used to drive a
// cereal BinaryOutputArchive over the trace_cache ring slot without any
// intermediate allocation.
class fixed_buffer_outbuf : public std::streambuf
{
public:
    fixed_buffer_outbuf(std::uint8_t* buffer, std::size_t capacity)
    : m_begin(buffer)
    , m_capacity(capacity)
    {
        auto* base = reinterpret_cast<char*>(buffer);
        setp(base, base + capacity);
    }

    [[nodiscard]] std::size_t bytes_written() const
    {
        return static_cast<std::size_t>(pptr() - reinterpret_cast<char*>(m_begin));
    }

protected:
    std::streamsize xsputn(const char_type* s, std::streamsize n) override
    {
        const auto remaining = static_cast<std::streamsize>(epptr() - pptr());
        if(n > remaining)
        {
            // Caller is expected to size the buffer via the counting pass; a
            // short write here means buffer reservation is too small.
            return 0;
        }
        std::memcpy(pptr(), s, static_cast<std::size_t>(n));
        pbump(static_cast<int>(n));
        return n;
    }

    int_type overflow(int_type ch) override
    {
        if(ch == traits_type::eof() || pptr() >= epptr())
        {
            return traits_type::eof();
        }
        *pptr() = static_cast<char>(ch);
        pbump(1);
        return ch;
    }

private:
    std::uint8_t* m_begin;
    std::size_t   m_capacity;
};

// Counts bytes that would be written.  Used for the size-precompute pass
// before reserving space in the ring buffer.
class counting_outbuf : public std::streambuf
{
public:
    [[nodiscard]] std::size_t bytes_written() const { return m_count; }

protected:
    std::streamsize xsputn(const char_type*, std::streamsize n) override
    {
        m_count += static_cast<std::size_t>(n);
        return n;
    }

    int_type overflow(int_type ch) override
    {
        if(ch == traits_type::eof())
        {
            return traits_type::eof();
        }
        ++m_count;
        return ch;
    }

private:
    std::size_t m_count{ 0 };
};

// Streambuf that reads from a raw byte cursor and advances it.  Mirrors the
// existing deserialize<T>(uint8_t*&) contract: the cursor passed in is
// advanced past the consumed bytes once the archive goes out of scope.
class cursor_inbuf : public std::streambuf
{
public:
    explicit cursor_inbuf(std::uint8_t*& cursor)
    : m_cursor(cursor)
    {
        auto* base = reinterpret_cast<char*>(m_cursor);
        // Set a large gettable area so cereal's xsgetn / sbumpc can run without
        // hitting underflow; we rely on the framing header's sample_size to
        // bound reads.  The end pointer is not used to validate.
        setg(base, base, base + static_cast<std::size_t>(-1) / 2);
    }

    ~cursor_inbuf() override { m_cursor = reinterpret_cast<std::uint8_t*>(gptr()); }

protected:
    std::streamsize xsgetn(char_type* s, std::streamsize n) override
    {
        std::memcpy(s, gptr(), static_cast<std::size_t>(n));
        gbump(static_cast<int>(n));
        return n;
    }

private:
    std::uint8_t*& m_cursor;
};

template <typename T>
inline std::size_t
cereal_serialized_size(const T& value)
{
    counting_outbuf buf;
    std::ostream    os(&buf);
    {
        tim::cereal::BinaryOutputArchive archive(os);
        archive(value);
    }
    return buf.bytes_written();
}

template <typename T>
inline void
cereal_serialize_to(std::uint8_t* buffer, std::size_t capacity, const T& value)
{
    fixed_buffer_outbuf buf(buffer, capacity);
    std::ostream        os(&buf);
    {
        tim::cereal::BinaryOutputArchive archive(os);
        archive(value);
    }
}

template <typename T>
inline T
cereal_deserialize_from(std::uint8_t*& cursor)
{
    T            value{};
    cursor_inbuf buf(cursor);
    std::istream is(&buf);
    {
        tim::cereal::BinaryInputArchive archive(is);
        archive(value);
    }
    return value;
}

}  // namespace trace_cache
}  // namespace rocprofsys
