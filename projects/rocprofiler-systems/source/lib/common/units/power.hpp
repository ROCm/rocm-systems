// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <compare>
#include <ratio>
#include <string_view>
#include <type_traits>

#include "common/units/quantity.hpp"

namespace rocprofsys::inline common::units
{

/**
 * A power value, modelled on std::chrono::duration.
 *
 * Stores a count of @p Period watts, where @p Period is a std::ratio relative
 * to one watt. `power<double, std::milli>{500}` therefore represents 500 mW.
 *
 * @tparam Rep    underlying arithmetic representation (defaults to double so
 *                that scaling never silently truncates).
 * @tparam Period std::ratio giving this unit's size in watts.
 */
template <typename Rep, typename Period = std::ratio<1>>
class power : public detail::quantity<Rep, Period>
{
public:
    using rep    = Rep;
    using period = Period;

    using detail::quantity<Rep, Period>::quantity;
};

using nanowatt  = power<double, std::nano>;
using microwatt = power<double, std::micro>;
using milliwatt = power<double, std::milli>;
using watt      = power<double, std::ratio<1>>;
using kilowatt  = power<double, std::kilo>;

namespace detail
{

template <typename T>
struct is_power : std::false_type
{};

template <typename Rep, typename Period>
struct is_power<power<Rep, Period>> : std::true_type
{};

}  // namespace detail

/** Satisfied by any instantiation of @ref power. */
template <typename T>
concept power_like = detail::is_power<std::remove_cvref_t<T>>::value;

/**
 * Printable suffix for a power @p Period (e.g. "kW" for std::kilo).
 *
 * Left undefined for unknown periods so that printing an unregistered unit is a
 * compile error; specialise it to add a new unit's suffix.
 */
template <typename Period>
struct power_suffix;

template <>
struct power_suffix<std::nano>
{
    static constexpr std::string_view k_value = "nW";
};
template <>
struct power_suffix<std::micro>
{
    static constexpr std::string_view k_value = "uW";
};
template <>
struct power_suffix<std::milli>
{
    static constexpr std::string_view k_value = "mW";
};
template <>
struct power_suffix<std::ratio<1>>
{
    static constexpr std::string_view k_value = "W";
};
template <>
struct power_suffix<std::kilo>
{
    static constexpr std::string_view k_value = "kW";
};

/**
 * Convert a power value to another power unit at compile time.
 *
 * Mirrors std::chrono::duration_cast: the factor is the exact rational
 * @c From::period / @c To::period, applied in @c To::rep.
 *
 * @tparam To   target power type.
 * @tparam From source power type (deduced).
 * @param  from value to convert.
 * @return @p from expressed in @p To's unit.
 */
template <power_like To, power_like From>
[[nodiscard]] constexpr To
power_cast(const From& from) noexcept
{
    using from_t = std::remove_cvref_t<From>;
    return To{ detail::quantity_cast_count<typename To::rep, typename To::period,
                                           typename from_t::rep, typename from_t::period>(
        from.count()) };
}

/**
 * Equality across any two power units.
 *
 * Both operands are converted to watts before comparing, so `1000_mw == 1_w`.
 */
template <power_like LHS, power_like RHS>
[[nodiscard]] constexpr bool
operator==(const LHS& lhs, const RHS& rhs) noexcept
{
    using rep  = std::common_type_t<typename LHS::rep, typename RHS::rep>;
    using base = power<rep, std::ratio<1>>;
    return power_cast<base>(lhs).count() == power_cast<base>(rhs).count();
}

/**
 * Ordering across any two power units.
 *
 * Both operands are normalised to watts before comparing. Returns
 * std::partial_ordering because the underlying rep is floating-point.
 */
template <power_like LHS, power_like RHS>
[[nodiscard]] constexpr auto
operator<=>(const LHS& lhs, const RHS& rhs) noexcept
{
    using rep  = std::common_type_t<typename LHS::rep, typename RHS::rep>;
    using base = power<rep, std::ratio<1>>;
    return power_cast<base>(lhs).count() <=> power_cast<base>(rhs).count();
}

namespace literals
{

// clang-format off
constexpr nanowatt  operator""_nw(long double value)        noexcept { return nanowatt{static_cast<double>(value)}; }
constexpr nanowatt  operator""_nw(unsigned long long value) noexcept { return nanowatt{static_cast<double>(value)}; }
constexpr microwatt operator""_uw(long double value)        noexcept { return microwatt{static_cast<double>(value)}; }
constexpr microwatt operator""_uw(unsigned long long value) noexcept { return microwatt{static_cast<double>(value)}; }
constexpr milliwatt operator""_mw(long double value)        noexcept { return milliwatt{static_cast<double>(value)}; }
constexpr milliwatt operator""_mw(unsigned long long value) noexcept { return milliwatt{static_cast<double>(value)}; }
constexpr watt      operator""_w (long double value)        noexcept { return watt{static_cast<double>(value)}; }
constexpr watt      operator""_w (unsigned long long value) noexcept { return watt{static_cast<double>(value)}; }
constexpr kilowatt  operator""_kw(long double value)        noexcept { return kilowatt{static_cast<double>(value)}; }
constexpr kilowatt  operator""_kw(unsigned long long value) noexcept { return kilowatt{static_cast<double>(value)}; }
// clang-format on

}  // namespace literals

}  // namespace rocprofsys::inline common::units
