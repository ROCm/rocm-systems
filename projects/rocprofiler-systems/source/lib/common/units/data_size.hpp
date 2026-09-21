// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <fmt/base.h>

#include <compare>
#include <cstdint>
#include <ratio>
#include <string_view>
#include <type_traits>

#include "common/units/quantity.hpp"

namespace rocprofsys::inline common::units
{

/**
 * A data-size value, modelled on std::chrono::duration.
 *
 * Stores a count of @p Scale bytes, where @p Scale is a std::ratio relative to
 * one byte, so `data_size<double, std::ratio<1024>>{1}` represents 1024 bytes.
 *
 * Both unit families are available: decimal SI (@ref kilobytes, 1 KB = 1000 B)
 * and binary IEC (@ref kibibytes, 1 KiB = 1024 B). They are distinct types and
 * print distinct suffixes, so pick whichever the consumer of the value expects.
 *
 * @tparam Rep   underlying arithmetic representation (defaults to double so that
 *               scaling never silently truncates).
 * @tparam Scale std::ratio giving this unit's size in bytes.
 */
template <typename Rep, typename Scale = std::ratio<1>>
class data_size : public detail::quantity<Rep, Scale>
{
public:
    using rep   = Rep;
    using scale = Scale;

    using detail::quantity<Rep, Scale>::quantity;

    /**
     * This size as a raw count of bytes (35000 for `35_kb`, 35840 for `35_kib`).
     *
     * Explicit on purpose: a typed size never decays into a number implicitly
     * so the unit cannot be dropped by accident.
     *
     * @warning The result has type @p Rep (double for the built-in aliases).
     *          Storing it in a narrower integer truncates, and a value outside
     *          that integer's range is undefined behaviour - prefer a 64-bit or
     *          floating-point target for large sizes. Because the conversion is
     *          explicit, that narrowing happens at the call site where
     *          `-Wconversion` can see it.
     * @return the size in bytes.
     */
    [[nodiscard]] constexpr Rep to_bytes() const noexcept
    {
        using calc = std::common_type_t<Rep, std::intmax_t>;
        return static_cast<Rep>(static_cast<calc>(this->count()) *
                                static_cast<calc>(Scale::num) /
                                static_cast<calc>(Scale::den));
    }
};

namespace detail
{
inline constexpr std::intmax_t k_bytes_per_kib = 1024LL;
}  // namespace detail

using bytes = data_size<double, std::ratio<1>>;

/// Decimal (SI) sizes: 1 KB = 1000 B. Suffixes "KB", "MB", "GB", "TB".
using kilobytes = data_size<double, std::kilo>;
using megabytes = data_size<double, std::mega>;
using gigabytes = data_size<double, std::giga>;
using terabytes = data_size<double, std::tera>;

/// Binary (IEC) sizes: 1 KiB = 1024 B. Suffixes "KiB", "MiB", "GiB", "TiB".
using kibibytes = data_size<double, std::ratio<detail::k_bytes_per_kib>>;
using mebibytes =
    data_size<double, std::ratio<detail::k_bytes_per_kib * detail::k_bytes_per_kib>>;
using gibibytes =
    data_size<double, std::ratio<detail::k_bytes_per_kib * detail::k_bytes_per_kib *
                                 detail::k_bytes_per_kib>>;
using tebibytes =
    data_size<double, std::ratio<detail::k_bytes_per_kib * detail::k_bytes_per_kib *
                                 detail::k_bytes_per_kib * detail::k_bytes_per_kib>>;

namespace detail
{

template <typename T>
struct is_data_size : std::false_type
{};

template <typename Rep, typename Scale>
struct is_data_size<data_size<Rep, Scale>> : std::true_type
{};

}  // namespace detail

/** Satisfied by any instantiation of @ref data_size. */
template <typename T>
concept data_size_like = detail::is_data_size<std::remove_cvref_t<T>>::value;

/**
 * Printable suffix for a data-size @p Scale (e.g. "MB" for 1024*1024 bytes).
 *
 * Left undefined for unknown scales so that printing an unregistered unit is a
 * compile error; specialise it to add a new unit's suffix.
 */
template <typename Scale>
struct data_size_suffix;

template <>
struct data_size_suffix<std::ratio<1>>
{
    static constexpr std::string_view k_value = "B";
};
template <>
struct data_size_suffix<std::kilo>
{
    static constexpr std::string_view k_value = "KB";
};
template <>
struct data_size_suffix<std::mega>
{
    static constexpr std::string_view k_value = "MB";
};
template <>
struct data_size_suffix<std::giga>
{
    static constexpr std::string_view k_value = "GB";
};
template <>
struct data_size_suffix<std::tera>
{
    static constexpr std::string_view k_value = "TB";
};
template <>
struct data_size_suffix<std::ratio<detail::k_bytes_per_kib>>
{
    static constexpr std::string_view k_value = "KiB";
};
template <>
struct data_size_suffix<std::ratio<detail::k_bytes_per_kib * detail::k_bytes_per_kib>>
{
    static constexpr std::string_view k_value = "MiB";
};
template <>
struct data_size_suffix<std::ratio<detail::k_bytes_per_kib * detail::k_bytes_per_kib *
                                   detail::k_bytes_per_kib>>
{
    static constexpr std::string_view k_value = "GiB";
};
template <>
struct data_size_suffix<std::ratio<detail::k_bytes_per_kib * detail::k_bytes_per_kib *
                                   detail::k_bytes_per_kib * detail::k_bytes_per_kib>>
{
    static constexpr std::string_view k_value = "TiB";
};

/**
 * Convert a data-size to another data-size unit at compile time.
 *
 * Mirrors std::chrono::duration_cast: the factor is the exact rational
 * @c From::scale / @c To::scale, applied in @c To::rep.
 *
 * @tparam To   target data-size type.
 * @tparam From source data-size type (deduced).
 * @param  from value to convert.
 * @return @p from expressed in @p To's unit.
 */
template <data_size_like To, data_size_like From>
[[nodiscard]] constexpr To
data_size_cast(const From& from) noexcept
{
    using from_t = std::remove_cvref_t<From>;
    return To{ detail::quantity_cast_count<typename To::rep, typename To::scale,
                                           typename from_t::rep, typename from_t::scale>(
        from.count()) };
}

/**
 * Equality across any two data-size units.
 *
 * Both operands are converted to bytes before comparing, so `1024_b == 1_kb`.
 */
template <data_size_like LHS, data_size_like RHS>
[[nodiscard]] constexpr bool
operator==(const LHS& lhs, const RHS& rhs) noexcept
{
    using rep  = std::common_type_t<typename LHS::rep, typename RHS::rep>;
    using base = data_size<rep, std::ratio<1>>;
    return data_size_cast<base>(lhs).count() == data_size_cast<base>(rhs).count();
}

/**
 * Ordering across any two data-size units.
 *
 * Both operands are normalised to bytes before comparing, so `1_kb < 2048_b`
 * is well-defined. Returns std::partial_ordering because the underlying rep is
 * floating-point.
 */
template <data_size_like LHS, data_size_like RHS>
[[nodiscard]] constexpr auto
operator<=>(const LHS& lhs, const RHS& rhs) noexcept
{
    using rep  = std::common_type_t<typename LHS::rep, typename RHS::rep>;
    using base = data_size<rep, std::ratio<1>>;
    return data_size_cast<base>(lhs).count() <=> data_size_cast<base>(rhs).count();
}

namespace literals
{

// clang-format off
constexpr bytes     operator""_b (long double value)        noexcept { return bytes{static_cast<double>(value)}; }
constexpr bytes     operator""_b (unsigned long long value) noexcept { return bytes{static_cast<double>(value)}; }
constexpr kilobytes operator""_kb(long double value)        noexcept { return kilobytes{static_cast<double>(value)}; }
constexpr kilobytes operator""_kb(unsigned long long value) noexcept { return kilobytes{static_cast<double>(value)}; }
constexpr megabytes operator""_mb(long double value)        noexcept { return megabytes{static_cast<double>(value)}; }
constexpr megabytes operator""_mb(unsigned long long value) noexcept { return megabytes{static_cast<double>(value)}; }
constexpr gigabytes operator""_gb(long double value)        noexcept { return gigabytes{static_cast<double>(value)}; }
constexpr gigabytes operator""_gb(unsigned long long value) noexcept { return gigabytes{static_cast<double>(value)}; }
constexpr terabytes operator""_tb(long double value)        noexcept { return terabytes{static_cast<double>(value)}; }
constexpr terabytes operator""_tb(unsigned long long value) noexcept { return terabytes{static_cast<double>(value)}; }

constexpr kibibytes operator""_kib(long double value)        noexcept { return kibibytes{static_cast<double>(value)}; }
constexpr kibibytes operator""_kib(unsigned long long value) noexcept { return kibibytes{static_cast<double>(value)}; }
constexpr mebibytes operator""_mib(long double value)        noexcept { return mebibytes{static_cast<double>(value)}; }
constexpr mebibytes operator""_mib(unsigned long long value) noexcept { return mebibytes{static_cast<double>(value)}; }
constexpr gibibytes operator""_gib(long double value)        noexcept { return gibibytes{static_cast<double>(value)}; }
constexpr gibibytes operator""_gib(unsigned long long value) noexcept { return gibibytes{static_cast<double>(value)}; }
constexpr tebibytes operator""_tib(long double value)        noexcept { return tebibytes{static_cast<double>(value)}; }
constexpr tebibytes operator""_tib(unsigned long long value) noexcept { return tebibytes{static_cast<double>(value)}; }
// clang-format on

}  // namespace literals

}  // namespace rocprofsys::inline common::units

// ---------------------------------------------------------------------------
// fmt::formatter<rocprofsys::common::units::data_size<Rep, Scale>>
//
// Default: fmt::format("{}", 2_mb)  -> "2 MB"
//          fmt::format("{}", 2_mib) -> "2 MiB"
// ---------------------------------------------------------------------------
template <typename Rep, typename Scale>
struct fmt::formatter<rocprofsys::common::units::data_size<Rep, Scale>>
: fmt::formatter<Rep>
{
    template <typename FormatContext>
    auto format(const rocprofsys::common::units::data_size<Rep, Scale>& value,
                FormatContext&                                          ctx) const
    {
        using rocprofsys::common::units::data_size_suffix;
        auto out = fmt::formatter<Rep>::format(value.count(), ctx);
        return fmt::format_to(out, " {}", data_size_suffix<Scale>::k_value);
    }
};
