// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Cereal-style archive: each type opts in by defining a single
//   template <class Archive> void serialize(Archive&)
// member that lists its fields with `ar(field, ...)`. Bytes go directly
// to / from a uint8_t cursor with no intermediate streambuf and no
// virtual dispatch.
//
// Wire layout:
//   POD / arithmetic / enum: native-endian memcpy(sizeof T)
//   std::string             : uint64 byte length + raw bytes
//   std::vector<T>          : uint64 byte count + bulk memcpy (trivial T)
//                              or per-element recursion
//   std::optional<T>        : uint8 present flag + recurse if present
//   class with member serialize<Archive>: dispatched recursively
//   types specialised on treat_as_blob<T>: raw memcpy(sizeof T)
//
// CRITICAL dispatch ordering: the member-`serialize` check MUST come
// BEFORE the `is_trivially_copyable_v` shortcut for class types, otherwise
// padded structs would memcpy with padding included and break wire parity
// against any handrolled field-by-field writer.

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

namespace rocprofsys
{
namespace trace_cache
{

class output_archive;
class input_archive;
class size_archive;

namespace archive_detail
{

template <typename T>
struct is_std_vector : std::false_type
{};

template <typename U, typename A>
struct is_std_vector<std::vector<U, A>> : std::true_type
{};

template <typename T>
inline constexpr bool is_std_vector_v = is_std_vector<T>::value;

template <typename T>
struct is_std_optional : std::false_type
{};

template <typename U>
struct is_std_optional<std::optional<U>> : std::true_type
{};

template <typename T>
inline constexpr bool is_std_optional_v = is_std_optional<T>::value;

template <typename T>
inline constexpr bool is_std_string_v = std::is_same_v<std::decay_t<T>, std::string>;

template <typename T>
inline constexpr bool dependent_false_v = false;

}  // namespace archive_detail

// Opt-in: types specialised to true are memcpy'd whole even when they
// are class types (and might otherwise look like they expose a member
// `serialize`). Use sparingly, only when the legacy wire bytes include
// alignment padding that field-by-field recursion would drop.
template <typename T>
struct treat_as_blob : std::false_type
{};

template <typename T>
inline constexpr bool treat_as_blob_v = treat_as_blob<T>::value;

// Detect a member `void serialize(Archive&)` reachable on T. Has to be
// SFINAE-friendly so the `if constexpr` chain does not hard-error on
// types that lack the member.
template <typename Archive, typename T, typename = void>
struct has_member_serialize : std::false_type
{};

template <typename Archive, typename T>
struct has_member_serialize<
    Archive, T,
    std::void_t<decltype(std::declval<T&>().serialize(std::declval<Archive&>()))>>
: std::true_type
{};

template <typename Archive, typename T>
inline constexpr bool has_member_serialize_v = has_member_serialize<Archive, T>::value;

// ---------------------------------------------------------------- output_archive

class output_archive
{
public:
    explicit output_archive(std::uint8_t* dst) noexcept
    : m_cur{ dst }
    , m_begin{ dst }
    {}

    template <typename... Ts>
    output_archive& operator()(const Ts&... vals)
    {
        (write_one(vals), ...);
        return *this;
    }

    [[nodiscard]] std::size_t bytes_written() const noexcept
    {
        return static_cast<std::size_t>(m_cur - m_begin);
    }

private:
    template <typename T>
    void write_one(const T& v)
    {
        using D = std::decay_t<T>;
        if constexpr(treat_as_blob_v<D>)
        {
            std::memcpy(m_cur, &v, sizeof(D));
            m_cur += sizeof(D);
        }
        else if constexpr(archive_detail::is_std_string_v<D>)
        {
            const std::uint64_t n = v.size();
            std::memcpy(m_cur, &n, sizeof(n));
            m_cur += sizeof(n);
            if(n != 0)
            {
                std::memcpy(m_cur, v.data(), n);
                m_cur += n;
            }
        }
        else if constexpr(archive_detail::is_std_vector_v<D>)
        {
            using elem_t          = typename D::value_type;
            const std::uint64_t n = v.size();
            std::memcpy(m_cur, &n, sizeof(n));
            m_cur += sizeof(n);
            if(n != 0)
            {
                if constexpr(std::is_trivially_copyable_v<elem_t>)
                {
                    const std::size_t bytes = n * sizeof(elem_t);
                    std::memcpy(m_cur, v.data(), bytes);
                    m_cur += bytes;
                }
                else
                {
                    for(const auto& e : v)
                    {
                        write_one(e);
                    }
                }
            }
        }
        else if constexpr(archive_detail::is_std_optional_v<D>)
        {
            const std::uint8_t flag = v.has_value() ? 1 : 0;
            *m_cur++                = flag;
            if(v.has_value())
            {
                write_one(*v);
            }
        }
        else if constexpr(std::is_class_v<D> && has_member_serialize_v<output_archive, D>)
        {
            // Field-by-field. Must take precedence over the trivially_copyable
            // memcpy shortcut so structs with internal alignment padding match
            // the legacy field-by-field wire bytes.
            const_cast<D&>(v).serialize(*this);
        }
        else if constexpr(std::is_arithmetic_v<D> || std::is_enum_v<D>)
        {
            std::memcpy(m_cur, &v, sizeof(D));
            m_cur += sizeof(D);
        }
        else
        {
            static_assert(archive_detail::dependent_false_v<D>,
                          "output_archive: unsupported type. Add a member "
                          "serialize<Archive>(Archive&) or specialise "
                          "treat_as_blob<T>::value = true.");
        }
    }

    std::uint8_t* m_cur;
    std::uint8_t* m_begin;
};

// ----------------------------------------------------------------- input_archive

class input_archive
{
public:
    explicit input_archive(std::uint8_t*& cursor) noexcept
    : m_cur{ cursor }
    {}

    template <typename... Ts>
    input_archive& operator()(Ts&... vals)
    {
        (read_one(vals), ...);
        return *this;
    }

private:
    template <typename T>
    void read_one(T& out)
    {
        using D = std::decay_t<T>;
        if constexpr(treat_as_blob_v<D>)
        {
            std::memcpy(&out, m_cur, sizeof(D));
            m_cur += sizeof(D);
        }
        else if constexpr(archive_detail::is_std_string_v<D>)
        {
            std::uint64_t n{};
            std::memcpy(&n, m_cur, sizeof(n));
            m_cur += sizeof(n);
            out.assign(reinterpret_cast<const char*>(m_cur), n);
            m_cur += n;
        }
        else if constexpr(archive_detail::is_std_vector_v<D>)
        {
            using elem_t = typename D::value_type;
            std::uint64_t n{};
            std::memcpy(&n, m_cur, sizeof(n));
            m_cur += sizeof(n);
            out.resize(n);
            if(n != 0)
            {
                if constexpr(std::is_trivially_copyable_v<elem_t>)
                {
                    const std::size_t bytes = n * sizeof(elem_t);
                    std::memcpy(out.data(), m_cur, bytes);
                    m_cur += bytes;
                }
                else
                {
                    for(auto& e : out)
                    {
                        read_one(e);
                    }
                }
            }
        }
        else if constexpr(archive_detail::is_std_optional_v<D>)
        {
            const std::uint8_t flag = *m_cur++;
            if(flag != 0)
            {
                out.emplace();
                read_one(*out);
            }
            else
            {
                out.reset();
            }
        }
        else if constexpr(std::is_class_v<D> && has_member_serialize_v<input_archive, D>)
        {
            out.serialize(*this);
        }
        else if constexpr(std::is_arithmetic_v<D> || std::is_enum_v<D>)
        {
            std::memcpy(&out, m_cur, sizeof(D));
            m_cur += sizeof(D);
        }
        else
        {
            static_assert(archive_detail::dependent_false_v<D>,
                          "input_archive: unsupported type. Add a member "
                          "serialize<Archive>(Archive&) or specialise "
                          "treat_as_blob<T>::value = true.");
        }
    }

    std::uint8_t*& m_cur;
};

// ------------------------------------------------------------------ size_archive

class size_archive
{
public:
    template <typename... Ts>
    size_archive& operator()(const Ts&... vals)
    {
        (count_one(vals), ...);
        return *this;
    }

    [[nodiscard]] std::size_t total() const noexcept { return m_total; }

private:
    template <typename T>
    void count_one(const T& v)
    {
        using D = std::decay_t<T>;
        if constexpr(treat_as_blob_v<D>)
        {
            m_total += sizeof(D);
        }
        else if constexpr(archive_detail::is_std_string_v<D>)
        {
            m_total += sizeof(std::uint64_t) + v.size();
        }
        else if constexpr(archive_detail::is_std_vector_v<D>)
        {
            using elem_t = typename D::value_type;
            m_total += sizeof(std::uint64_t);
            if constexpr(std::is_trivially_copyable_v<elem_t>)
            {
                m_total += v.size() * sizeof(elem_t);
            }
            else
            {
                for(const auto& e : v)
                {
                    count_one(e);
                }
            }
        }
        else if constexpr(archive_detail::is_std_optional_v<D>)
        {
            m_total += sizeof(std::uint8_t);
            if(v.has_value())
            {
                count_one(*v);
            }
        }
        else if constexpr(std::is_class_v<D> && has_member_serialize_v<size_archive, D>)
        {
            const_cast<D&>(v).serialize(*this);
        }
        else if constexpr(std::is_arithmetic_v<D> || std::is_enum_v<D>)
        {
            m_total += sizeof(D);
        }
        else
        {
            static_assert(archive_detail::dependent_false_v<D>,
                          "size_archive: unsupported type. Add a member "
                          "serialize<Archive>(Archive&) or specialise "
                          "treat_as_blob<T>::value = true.");
        }
    }

    std::size_t m_total{ 0 };
};

// ------------------------------------------------------------------ free helpers

template <typename T>
inline std::size_t
serialized_size(const T& v)
{
    size_archive sa;
    sa(v);
    return sa.total();
}

template <typename T>
inline void
serialize_to(std::uint8_t* dst, const T& v)
{
    output_archive oa{ dst };
    oa(v);
}

template <typename T>
inline T
deserialize_from(std::uint8_t*& cursor)
{
    T             out{};
    input_archive ia{ cursor };
    ia(out);
    return out;
}

}  // namespace trace_cache
}  // namespace rocprofsys
