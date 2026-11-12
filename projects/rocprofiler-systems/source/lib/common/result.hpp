// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

namespace rocprofsys::inline common
{
/**
 * @brief Tags a value as the error side, so `result<T, E>` stays unambiguous when `T`
 *        and `E` are convertible to each other.
 */
template <typename E>
struct failure
{
    E error;
};

template <typename E>
failure(E) -> failure<E>;

/**
 * @brief Holds either a success value `T` or an error `E`, never both.
 *
 * Use it for operations that can fail with a reason the caller may need. `T` and `E`
 * must be nothrow move-constructible.
 */
template <typename T, typename E>
class [[nodiscard]] result
{
    // A throwing move during assignment between alternatives would leave the variant
    // valueless, making is_ok() and is_error() both false.
    static_assert(std::is_nothrow_move_constructible_v<T> &&
                      std::is_nothrow_move_constructible_v<E>,
                  "result<T, E> requires nothrow move-constructible T and E");

public:
    constexpr explicit result(T value)
    : m_data{ std::in_place_index<0>, std::move(value) }
    {}

    constexpr explicit result(failure<E> fail)
    : m_data{ std::in_place_index<1>, std::move(fail.error) }
    {}

    [[nodiscard]] constexpr bool is_ok() const noexcept { return m_data.index() == 0; }
    [[nodiscard]] constexpr bool is_error() const noexcept { return m_data.index() == 1; }

    /// @throws std::logic_error if this holds an error.
    [[nodiscard]] constexpr T& value() &
    {
        if(is_error())
        {
            throw std::logic_error{ "result::value() called on a failure" };
        }
        return std::get<0>(m_data);
    }

    /// @throws std::logic_error if this holds an error.
    [[nodiscard]] constexpr const T& value() const&
    {
        if(is_error())
        {
            throw std::logic_error{ "result::value() called on a failure" };
        }
        return std::get<0>(m_data);
    }

    /// @throws std::logic_error if this holds a value.
    [[nodiscard]] constexpr const E& error() const&
    {
        if(is_ok())
        {
            throw std::logic_error{ "result::error() called on a success" };
        }
        return std::get<1>(m_data);
    }

    template <typename U>
    [[nodiscard]] constexpr T value_or(U&& fallback) const&
    {
        return is_ok() ? std::get<0>(m_data) : static_cast<T>(std::forward<U>(fallback));
    }

private:
    std::variant<T, E> m_data;
};
}  // namespace rocprofsys::inline common
