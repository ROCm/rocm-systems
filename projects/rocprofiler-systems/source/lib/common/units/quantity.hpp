// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <concepts>
#include <cstdint>
#include <limits>
#include <ratio>
#include <type_traits>
#include <utility>

namespace rocprofsys::inline common::units::detail
{

/**
 * Shared storage + construction machinery for frequency, data_size, and power,
 * modelled on std::chrono::duration. Each of those types derives from a
 * quantity<Rep, Ratio> instantiation and inherits its constructors, so this is
 * the one place the narrowing-conversion guard and count() live.
 *
 * @tparam Rep   underlying arithmetic representation (defaults to double so
 *               that scaling never silently truncates).
 * @tparam Ratio std::ratio giving the derived unit's size relative to its base
 *               unit (a frequency::period, data_size::scale, or power::period).
 */
template <std::floating_point Rep, typename Ratio>
class quantity
{
public:
    constexpr quantity() = default;

    /**
     * The constructor is explicit, so a count never converts implicitly into a
     * quantity. On top of that, @p U is admitted only when every value it can
     * hold is exactly representable in @p Rep, which is what the digit-count
     * comparison tests. `int` and `float` therefore pass for a `double` @p Rep,
     * while `long`, `long long` (signed or unsigned) and `long double` do not,
     * and the caller has to write the narrowing cast where it is visible.
     */
    template <typename U>
        requires std::convertible_to<const U&, Rep> &&
                 (std::numeric_limits<std::remove_cvref_t<U>>::digits <=
                  std::numeric_limits<Rep>::digits)
    constexpr explicit quantity(U&& value) noexcept
    : m_count{ static_cast<Rep>(std::forward<U>(value)) }
    {}

    /** @return the raw count in units of @p Ratio. */
    [[nodiscard]] constexpr Rep count() const noexcept { return m_count; }

private:
    Rep m_count{};
};

/**
 * Shared arithmetic core for the frequency_cast/data_size_cast/power_cast
 * helpers: converts a raw count from one ratio to another.
 *
 * Mirrors std::chrono::duration_cast: the factor is the exact rational
 * @p FromRatio / @p ToRatio, applied in @p ToRep.
 */
template <typename ToRep, typename ToRatio, typename FromRep, typename FromRatio>
[[nodiscard]] constexpr ToRep
quantity_cast_count(FromRep from_count) noexcept
{
    using factor_t     = std::ratio_divide<FromRatio, ToRatio>;
    using calc_t       = std::common_type_t<ToRep, FromRep, std::intmax_t>;
    const auto count_t = static_cast<calc_t>(from_count) *
                         static_cast<calc_t>(factor_t::num) /
                         static_cast<calc_t>(factor_t::den);
    return static_cast<ToRep>(count_t);
}

}  // namespace rocprofsys::inline common::units::detail
