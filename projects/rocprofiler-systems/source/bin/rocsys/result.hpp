// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "diagnostic.hpp"

#include <utility>
#include <variant>

namespace rocprofsys::rocsys
{

/// @brief A value or a @ref diagnostic. Becomes `std::expected` once the project is
/// C++23.
template <typename T>
class result
{
public:
    result(T value)
    : m_storage(std::move(value))
    {}

    result(diagnostic error)
    : m_storage(std::move(error))
    {}

    [[nodiscard]] bool has_value() const noexcept
    {
        return std::holds_alternative<T>(m_storage);
    }

    [[nodiscard]] explicit operator bool() const noexcept { return has_value(); }

    [[nodiscard]] T&       value() & { return std::get<T>(m_storage); }
    [[nodiscard]] const T& value() const& { return std::get<T>(m_storage); }

    [[nodiscard]] const diagnostic& error() const
    {
        return std::get<diagnostic>(m_storage);
    }

private:
    std::variant<T, diagnostic> m_storage;
};

}  // namespace rocprofsys::rocsys
